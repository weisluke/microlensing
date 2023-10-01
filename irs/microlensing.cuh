#pragma once

#include "binomial_coefficients.cuh"
#include "complex.cuh"
#include "fmm.cuh"
#include "irs_microlensing.cuh"
#include "mass_function.cuh"
#include "star.cuh"
#include "stopwatch.hpp"
#include "tree_node.cuh"
#include "util.hpp"

#include <curand_kernel.h>

#include <algorithm> //for std::min and std::max
#include <chrono> //for setting random seed with clock
#include <cmath>
#include <fstream> //for writing files
#include <iostream>
#include <limits> //for std::numeric_limits
#include <string>


/******************************************************************************
number of stars to use directly when shooting rays
this helps determine the size of the tree
******************************************************************************/
const int MAX_NUM_STARS_DIRECT = 32;


template <typename T>
class Microlensing
{

public:

	/******************************************************************************
	variables for kernel threads and blocks
	******************************************************************************/
	dim3 threads;
	dim3 blocks;

	/******************************************************************************
	stopwatch for timing purposes
	******************************************************************************/
	Stopwatch stopwatch;


	const T PI = static_cast<T>(3.1415926535898);

	/******************************************************************************
	default variables
	******************************************************************************/
	T kappa_tot = 0.3;
	T shear = 0.3;
	T smooth_fraction = 0.1;
	T kappa_star = 0.27;
	T theta_e = 1;
	std::string mass_function_str = "equal";
	T m_solar = 1;
	T m_lower = 0.01;
	T m_upper = 50;
	T light_loss = 0.01;
	int rectangular = 1;
	int approx = 1;
	T safety_scale = 1.37;
	std::string starfile = "";
	T half_length_source = 5;
	int num_pixels = 1000;
	int num_rays_source = 100;
	int random_seed = 0;
	int write_maps = 1;
	int write_parities = 0;
	int write_histograms = 1;
	std::string outfile_type = ".bin";
	std::string outfile_prefix = "./";

	/******************************************************************************
	derived variables
	******************************************************************************/
	massfunctions::massfunction mass_function;
	T mean_mass;
	T mean_mass2;

	int num_stars;
	T kappa_star_actual;
	T m_lower_actual;
	T m_upper_actual;
	T mean_mass_actual;
	T mean_mass2_actual;

	T mu_ave;
	T num_rays_lens;
	T ray_sep;
	Complex<T> half_length_image;
	Complex<T> corner;
	int taylor_smooth;

	int tree_size;
	int tree_levels;
	int expansion_order;

	/******************************************************************************
	dynamic memory
	******************************************************************************/
	curandState* states = nullptr;
	star<T>* stars = nullptr;
	star<T>* temp_stars = nullptr;

	int* binomial_coeffs = nullptr;
	TreeNode<T>* tree = nullptr;

	int* pixels = nullptr;
	int* pixels_minima = nullptr;
	int* pixels_saddles = nullptr;

	int* min_rays = nullptr;
	int* max_rays = nullptr;
	int histogram_length = 0;
	int* histogram = nullptr;
	int* histogram_minima = nullptr;
	int* histogram_saddles = nullptr;


	Microlensing()
	{

	}

	bool calculate_derived_params(bool verbose)
	{
		double t_elapsed;

		std::cout << "Calculating derived parameters...\n";
		stopwatch.start();

		/******************************************************************************
		determine mass function, <m>, and <m^2>
		******************************************************************************/
		mass_function = massfunctions::MASS_FUNCTIONS.at(mass_function_str);
		set_param("mean_mass", mean_mass, MassFunction<T>(mass_function).mean_mass(m_solar, m_lower, m_upper), verbose);
		set_param("mean_mass2", mean_mass2, MassFunction<T>(mass_function).mean_mass2(m_solar, m_lower, m_upper), verbose, starfile != "");

		/******************************************************************************
		if star file is specified, check validity of values and set num_stars,
		m_lower_actual, m_upper_actual, mean_mass_actual, and mean_mass2_actual based
		on star information
		******************************************************************************/
		if (starfile != "")
		{
			std::cout << "Calculating some parameter values based on star input file " << starfile << "\n";

			if (!read_star_file<T>(num_stars, rectangular, corner, theta_e, stars, 
				kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual, mean_mass2_actual, starfile))
			{
				std::cerr << "Error. Unable to read star field parameters from file " << starfile << "\n";
				return false;
			}

			std::cout << "Done calculating some parameter values based on star input file " << starfile << "\n\n";

			set_param("m_lower", m_lower, m_lower_actual, verbose);
			set_param("m_upper", m_upper, m_upper_actual, verbose);
			set_param("mean_mass", mean_mass, mean_mass_actual, verbose);
			set_param("mean_mass2", mean_mass2, mean_mass2_actual, verbose);
		}

		/******************************************************************************
		average magnification of the system
		******************************************************************************/
		set_param("mu_ave", mu_ave, 1 / ((1 - kappa_tot) * (1 - kappa_tot) - shear * shear), verbose);

		/******************************************************************************
		number density of rays in the lens plane
		uses the fact that for a given user specified number density of rays in the
		source plane, further subdivisions are made that multiply the effective number
		of rays in the image plane by 27^2
		******************************************************************************/
		set_param("num_rays_lens", num_rays_lens, 
			num_rays_source / std::abs(mu_ave) * num_pixels * num_pixels / (2 * half_length_source * 2 * half_length_source) / (NUM_RESAMPLED_RAYS * NUM_RESAMPLED_RAYS),
			verbose);
		
		/******************************************************************************
		average separation between rays in one dimension is 1/sqrt(number density)
		******************************************************************************/
		set_param("ray_sep", ray_sep, 1 / std::sqrt(num_rays_lens), verbose);

		/******************************************************************************
		shooting region is greater than outer boundary for macro-mapping by the size of
		the region of images visible for a macro-image which on average loses no more
		than the desired amount of flux
		******************************************************************************/
		half_length_image = Complex<T>(
			(half_length_source + theta_e * std::sqrt(kappa_star * mean_mass2 / (mean_mass * light_loss))) / std::abs(1 - kappa_tot + shear),
			(half_length_source + theta_e * std::sqrt(kappa_star * mean_mass2 / (mean_mass * light_loss))) / std::abs(1 - kappa_tot - shear)
			);

		/******************************************************************************
		make shooting region a multiple of the ray separation
		******************************************************************************/
		set_param("half_length_image", half_length_image, Complex<T>(ray_sep) * (Complex<int>(half_length_image / ray_sep) + Complex<int>(1, 1)), verbose);

		/******************************************************************************
		if stars are not drawn from external file, calculate final number of stars to
		use
		******************************************************************************/
		if (starfile == "")
		{
			if (rectangular)
			{
				set_param("num_stars", num_stars, static_cast<int>((safety_scale * 2 * half_length_image.re) * (safety_scale * 2 * half_length_image.im)
					* kappa_star / (PI * theta_e * theta_e * mean_mass)) + 1, verbose);

				set_param("corner", corner,
					std::sqrt(PI * theta_e * theta_e * num_stars * mean_mass / (4 * kappa_star))
					* Complex<T>(
						std::sqrt(std::abs((1 - kappa_tot - shear) / (1 - kappa_tot + shear))),
						std::sqrt(std::abs((1 - kappa_tot + shear) / (1 - kappa_tot - shear)))
						),
					verbose);
			}
			else
			{
				set_param("num_stars", num_stars, static_cast<int>(safety_scale * safety_scale * half_length_image.abs() * half_length_image.abs()
					* kappa_star / (theta_e * theta_e * mean_mass)) + 1, verbose);

				set_param("corner", corner,
					std::sqrt(theta_e * theta_e * num_stars * mean_mass / (kappa_star * 2 * ((1 - kappa_tot) * (1 - kappa_tot) + shear * shear)))
					* Complex<T>(
						std::sqrt(std::abs(1 - kappa_tot - shear)),
						std::sqrt(std::abs(1 - kappa_tot + shear))
						),
					verbose);
			}
		}

		set_param("taylor_smooth", taylor_smooth,
			std::max(
				static_cast<int>(std::log(2 * kappa_star * corner.abs() / (2 * half_length_source / num_pixels * PI)) / std::log(safety_scale)),
				1),
			verbose && rectangular && approx);

		if (rectangular)
		{
			set_param("tree_levels", tree_levels, 
				static_cast<int>(
					std::log2(num_stars * 9 / MAX_NUM_STARS_DIRECT * (corner.re > corner.im ? corner.re / corner.im : corner.im / corner.re)) / 2
					) + 1,
				verbose);
		}
		else
		{
			set_param("tree_levels", tree_levels, static_cast<int>(std::log2(num_stars * 9 / MAX_NUM_STARS_DIRECT / PI * 4) / 2) + 1, verbose);
		}

		set_param("tree_size", tree_size, ((1 << (2 * tree_levels + 2)) - 1) / 3, verbose);

		set_param("expansion_order", expansion_order,
			static_cast<int>(std::log2(theta_e * theta_e * m_upper * num_pixels / (2 * half_length_source) * MAX_NUM_STARS_DIRECT / 9)) + 1, verbose);
		if (expansion_order > treenode::MAX_EXPANSION_ORDER)
		{
			std::cerr << "Error. Maximum allowed expansion order is " << treenode::MAX_EXPANSION_ORDER << "\n";
			return false;
		}

		t_elapsed = stopwatch.stop();
		std::cout << "Done calculating derived parameters. Elapsed time: " << t_elapsed << " seconds.\n\n";

		return true;
	}

	bool allocate_initialize_memory(bool verbose)
	{
		double t_elapsed;

		std::cout << "Allocating memory...\n";
		stopwatch.start();

		/******************************************************************************
		allocate memory for stars
		******************************************************************************/
		cudaMallocManaged(&states, num_stars * sizeof(curandState));
		if (cuda_error("cudaMallocManaged(*states)", false, __FILE__, __LINE__)) return false;
		if (starfile == "")
		{
			cudaMallocManaged(&stars, num_stars * sizeof(star<T>));
			if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return false;
		}
		cudaMallocManaged(&temp_stars, num_stars * sizeof(star<T>));
		if (cuda_error("cudaMallocManaged(*temp_stars)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		allocate memory for tree
		******************************************************************************/
		cudaMallocManaged(&binomial_coeffs, 2 * expansion_order * (2 * expansion_order + 3) / 2 * sizeof(int));
		if (cuda_error("cudaMallocManaged(*binomial_coeffs)", false, __FILE__, __LINE__)) return false;
		cudaMallocManaged(&tree, tree_size * sizeof(TreeNode<T>));
		if (cuda_error("cudaMallocManaged(*tree)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		allocate memory for pixels
		******************************************************************************/
		cudaMallocManaged(&pixels, num_pixels * num_pixels * sizeof(int));
		if (cuda_error("cudaMallocManaged(*pixels)", false, __FILE__, __LINE__)) return false;
		if (write_parities)
		{
			cudaMallocManaged(&pixels_minima, num_pixels * num_pixels * sizeof(int));
			if (cuda_error("cudaMallocManaged(*pixels_minima)", false, __FILE__, __LINE__)) return false;
			cudaMallocManaged(&pixels_saddles, num_pixels * num_pixels * sizeof(int));
			if (cuda_error("cudaMallocManaged(*pixels_saddles)", false, __FILE__, __LINE__)) return false;
		}

		t_elapsed = stopwatch.stop();
		std::cout << "Done allocating memory. Elapsed time: " << t_elapsed << " seconds.\n\n";


		/******************************************************************************
		initialize pixel values
		******************************************************************************/
		std::cout << "Initializing pixel values...\n";
		stopwatch.start();

		initialize_pixels_kernel<T> <<<blocks, threads>>> (pixels, num_pixels);
		if (cuda_error("initialize_pixels_kernel", true, __FILE__, __LINE__)) return false;
		if (write_parities)
		{
			initialize_pixels_kernel<T> <<<blocks, threads>>> (pixels_minima, num_pixels);
			if (cuda_error("initialize_pixels_kernel", true, __FILE__, __LINE__)) return false;
			initialize_pixels_kernel<T> <<<blocks, threads>>> (pixels_saddles, num_pixels);
			if (cuda_error("initialize_pixels_kernel", true, __FILE__, __LINE__)) return false;
		}

		t_elapsed = stopwatch.stop();
		std::cout << "Done initializing pixel values. Elapsed time: " << t_elapsed << " seconds.\n\n";

		return true;
	}

	bool populate_star_array(bool verbose) 
	{
		/******************************************************************************
		BEGIN populating star array
		******************************************************************************/

		if (starfile == "")
		{
			std::cout << "Generating star field...\n";
			stopwatch.start();

			/******************************************************************************
			if random seed was not provided, get one based on the time
			******************************************************************************/
			while (random_seed == 0)
			{
				set_param("random_seed", random_seed, static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count()), verbose);
			}

			/******************************************************************************
			generate random star field if no star file has been given
			******************************************************************************/
			initialize_curand_states_kernel<T> <<<blocks, threads>>> (states, num_stars, random_seed);
			if (cuda_error("initialize_curand_states_kernel", true, __FILE__, __LINE__)) return false;
			generate_star_field_kernel<T> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, mass_function, m_solar, m_lower, m_upper);
			if (cuda_error("generate_star_field_kernel", true, __FILE__, __LINE__)) return false;

			double t_elapsed = stopwatch.stop();
			std::cout << "Done generating star field. Elapsed time: " << t_elapsed << " seconds.\n\n";

			/******************************************************************************
			calculate kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual,
			and mean_mass2_actual based on star information
			******************************************************************************/
			calculate_star_params<T>(num_stars, rectangular, corner, theta_e, stars, 
				kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual, mean_mass2_actual);
		}
		else
		{
			/******************************************************************************
			ensure random seed is 0 to denote that stars come from external file
			******************************************************************************/
			set_param("random_seed", random_seed, 0, verbose);

			std::cout << "Reading star field from file " << starfile << "\n";

			/******************************************************************************
			reading star field from external file
			******************************************************************************/
			if (!read_star_file<T>(num_stars, rectangular, corner, theta_e, stars, 
				kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual, mean_mass2_actual, starfile))
			{
				std::cerr << "Error. Unable to read star field from file " << starfile << "\n";
				return false;
			}

			std::cout << "Done reading star field from file " << starfile << "\n\n";
		}

		/******************************************************************************
		END populating star array
		******************************************************************************/

		return true;
	}

	bool create_tree(bool verbose)
	{
		/******************************************************************************
		number of threads per block, and number of blocks per grid
		uses 512 for number of threads in x dimension, as 1024 is the maximum allowable
		number of threads per block but is too large for some memory allocation, and
		512 is next power of 2 smaller
		******************************************************************************/
		set_threads(threads, 512);
		set_blocks(threads, blocks, num_stars);


		/******************************************************************************
		BEGIN create root node, then create children and sort stars
		******************************************************************************/
		T root_half_length;
		if (rectangular)
		{
			root_half_length = corner.re > corner.im ? corner.re : corner.im;
		}
		else
		{
			root_half_length = corner.abs();
		}
		root_half_length /= (1 << tree_levels);
		root_half_length = ray_sep * (static_cast<int>(root_half_length / ray_sep) + 1);
		set_param("root_half_length", root_half_length, root_half_length * (1 << tree_levels), verbose);
		tree[0] = TreeNode<T>(Complex<T>(0, 0), root_half_length, 0, 0);
		tree[0].numstars = num_stars;

		int* max_num_stars_in_level;
		int* min_num_stars_in_level;
		cudaMallocManaged(&max_num_stars_in_level, sizeof(int));
		if (cuda_error("cudaMallocManaged(*max_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		cudaMallocManaged(&min_num_stars_in_level, sizeof(int));
		if (cuda_error("cudaMallocManaged(*min_num_stars_in_level)", false, __FILE__, __LINE__)) return false;

		print_verbose("Creating children and sorting stars...\n", verbose);
		stopwatch.start();
		set_threads(threads, 512);
		for (int i = 0; i < tree_levels; i++)
		{
			print_verbose("Loop " + std::to_string(i + 1) + " /  " + std::to_string(tree_levels) + "\n", verbose);

			set_blocks(threads, blocks, treenode::get_num_nodes(i));
			treenode::create_children_kernel<T> <<<blocks, threads>>> (tree, i);
			if (cuda_error("create_tree_kernel", true, __FILE__, __LINE__)) return false;

			set_blocks(threads, blocks, 512 * treenode::get_num_nodes(i));
			treenode::sort_stars_kernel<T> <<<blocks, threads>>> (tree, i, stars, temp_stars);
			if (cuda_error("sort_stars_kernel", true, __FILE__, __LINE__)) return false;


			*max_num_stars_in_level = 0;
			*min_num_stars_in_level = num_stars;

			set_blocks(threads, blocks, treenode::get_num_nodes(i));
			treenode::get_min_max_stars_kernel<T> <<<blocks, threads>>> (tree, i + 1, min_num_stars_in_level, max_num_stars_in_level);
			if (cuda_error("get_min_max_stars_kernel", true, __FILE__, __LINE__)) return false;

			if (*max_num_stars_in_level <= MAX_NUM_STARS_DIRECT)
			{
				print_verbose("Necessary recursion limit reached.\n", verbose);
				print_verbose("Maximum number of stars in a node and its neighbors is " + std::to_string(*max_num_stars_in_level) + "\n", verbose);
				print_verbose("Minimum number of stars in a node and its neighbors is " + std::to_string(*min_num_stars_in_level) + "\n", verbose);
				set_param("tree_levels", tree_levels, i + 1, verbose);
				break;
			}
			else
			{
				print_verbose("Maximum number of stars in a node and its neighbors is " + std::to_string(*max_num_stars_in_level) + "\n", verbose);
				print_verbose("Minimum number of stars in a node and its neighbors is " + std::to_string(*min_num_stars_in_level) + "\n", verbose);
			}
		}
		print_verbose("Done creating children and sorting stars. Elapsed time: " + std::to_string(stopwatch.stop()) + " seconds.\n\n", verbose);

		set_threads(threads, 512);
		for (int i = 0; i <= tree_levels; i++)
		{
			set_blocks(threads, blocks, treenode::get_num_nodes(i));
			treenode::set_neighbors_kernel<T> <<<blocks, threads>>> (tree, i);
			if (cuda_error("set_neighbors_kernel", true, __FILE__, __LINE__)) return false;
		}

		/******************************************************************************
		END create root node, then create children and sort stars
		******************************************************************************/

		print_verbose("Calculating binomial coefficients...\n", verbose);
		calculate_binomial_coeffs(binomial_coeffs, 2 * expansion_order);
		print_verbose("Done calculating binomial coefficients.\n\n", verbose);


		print_verbose("Calculating multipole and local coefficients...\n", verbose);
		stopwatch.start();

		set_threads(threads, expansion_order + 1);
		set_blocks(threads, blocks, (expansion_order + 1) * treenode::get_num_nodes(tree_levels));
		fmm::calculate_multipole_coeffs_kernel<T> <<<blocks, threads, (expansion_order + 1) * sizeof(Complex<T>)>>> (tree, tree_levels, expansion_order, stars);

		set_threads(threads, expansion_order + 1, 4);
		for (int i = tree_levels - 1; i >= 0; i--)
		{
			set_blocks(threads, blocks, (expansion_order + 1) * treenode::get_num_nodes(i), 4);
			fmm::calculate_M2M_coeffs_kernel<T> <<<blocks, threads, 4 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree, i, expansion_order, binomial_coeffs);
		}

		for (int i = 2; i <= tree_levels; i++)
		{
			set_threads(threads, expansion_order + 1);
			set_blocks(threads, blocks, (expansion_order + 1) * treenode::get_num_nodes(i));
			fmm::calculate_L2L_coeffs_kernel<T> <<<blocks, threads, (expansion_order + 1) * sizeof(Complex<T>)>>> (tree, i, expansion_order, binomial_coeffs);

			set_threads(threads, expansion_order + 1, 27);
			set_blocks(threads, blocks, (expansion_order + 1) * treenode::get_num_nodes(i), 27);
			fmm::calculate_M2L_coeffs_kernel<T> <<<blocks, threads, 27 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree, i, expansion_order, binomial_coeffs);
		}
		if (cuda_error("calculate_coeffs_kernels", true, __FILE__, __LINE__)) return false;

		print_verbose("Done calculating multipole and local coefficients. Elapsed time: " + std::to_string(stopwatch.stop()) + " seconds.\n\n", verbose);

		return true;
	}

	bool create_histograms(bool verbose)
	{
		/******************************************************************************
		create histograms of pixel values
		******************************************************************************/


		if (write_histograms)
		{
			std::cout << "Creating histograms...\n";
			stopwatch.start();

			cudaMallocManaged(&min_rays, sizeof(int));
			if (cuda_error("cudaMallocManaged(*min_rays)", false, __FILE__, __LINE__)) return false;
			cudaMallocManaged(&max_rays, sizeof(int));
			if (cuda_error("cudaMallocManaged(*max_rays)", false, __FILE__, __LINE__)) return false;

			*min_rays = std::numeric_limits<int>::max();
			*max_rays = 0;

			/******************************************************************************
			redefine thread and block size to maximize parallelization
			******************************************************************************/
			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels, num_pixels);

			histogram_min_max_kernel<T> <<<blocks, threads>>> (pixels, num_pixels, min_rays, max_rays);
			if (cuda_error("histogram_min_max_kernel", true, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				histogram_min_max_kernel<T> <<<blocks, threads>>> (pixels_minima, num_pixels, min_rays, max_rays);
				if (cuda_error("histogram_min_max_kernel", true, __FILE__, __LINE__)) return false;
				histogram_min_max_kernel<T> <<<blocks, threads>>> (pixels_saddles, num_pixels, min_rays, max_rays);
				if (cuda_error("histogram_min_max_kernel", true, __FILE__, __LINE__)) return false;
			}

			histogram_length = *max_rays - *min_rays + 1;

			cudaMallocManaged(&histogram, histogram_length * sizeof(int));
			if (cuda_error("cudaMallocManaged(*histogram)", false, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				cudaMallocManaged(&histogram_minima, histogram_length * sizeof(int));
				if (cuda_error("cudaMallocManaged(*histogram_minima)", false, __FILE__, __LINE__)) return false;
				cudaMallocManaged(&histogram_saddles, histogram_length * sizeof(int));
				if (cuda_error("cudaMallocManaged(*histogram_saddles)", false, __FILE__, __LINE__)) return false;
			}

			/******************************************************************************
			redefine thread and block size to maximize parallelization
			******************************************************************************/
			set_threads(threads, 512);
			set_blocks(threads, blocks, histogram_length);

			initialize_histogram_kernel<T> <<<blocks, threads>>> (histogram, histogram_length);
			if (cuda_error("initialize_histogram_kernel", true, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				initialize_histogram_kernel<T> <<<blocks, threads>>> (histogram_minima, histogram_length);
				if (cuda_error("initialize_histogram_kernel", true, __FILE__, __LINE__)) return false;
				initialize_histogram_kernel<T> <<<blocks, threads>>> (histogram_saddles, histogram_length);
				if (cuda_error("initialize_histogram_kernel", true, __FILE__, __LINE__)) return false;
			}

			/******************************************************************************
			redefine thread and block size to maximize parallelization
			******************************************************************************/
			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels, num_pixels);

			histogram_kernel<T> <<<blocks, threads>>> (pixels, num_pixels, *min_rays, histogram);
			if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				histogram_kernel<T> <<<blocks, threads>>> (pixels_minima, num_pixels, *min_rays, histogram_minima);
				if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return false;
				histogram_kernel<T> <<<blocks, threads>>> (pixels_saddles, num_pixels, *min_rays, histogram_saddles);
				if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return false;
			}

			std::cout << "Done creating histograms. Elapsed time: " + std::to_string(stopwatch.stop()) + " seconds.\n\n";
		}
		/******************************************************************************
		done creating histograms of pixel values
		******************************************************************************/

		return true;
	}

	bool write_files(bool verbose, double t_ray_shoot)
	{
		/******************************************************************************
		stream for writing output files
		set precision to 9 digits
		******************************************************************************/
		std::ofstream outfile;
		outfile.precision(9);
		std::string fname;


		std::cout << "Writing parameter info...\n";
		fname = outfile_prefix + "irs_parameter_info.txt";
		outfile.open(fname);
		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		outfile << "kappa_tot " << kappa_tot << "\n";
		outfile << "shear " << shear << "\n";
		outfile << "mu_ave " << mu_ave << "\n";
		outfile << "smooth_fraction " << smooth_fraction << "\n";
		outfile << "kappa_star " << kappa_star << "\n";
		if (starfile == "")
		{
			outfile << "kappa_star_actual " << kappa_star_actual << "\n";
		}
		outfile << "theta_e " << theta_e << "\n";
		if (starfile == "")
		{
			outfile << "mass_function " << mass_function_str << "\n";
			if (mass_function_str == "salpeter" || mass_function_str == "kroupa")
			{
				outfile << "m_solar " << m_solar << "\n";
			}
			if (mass_function_str != "equal")
			{
				outfile << "m_lower " << m_lower << "\n";
				outfile << "m_upper " << m_upper << "\n";
				outfile << "m_lower_actual " << m_lower_actual << "\n";
				outfile << "m_upper_actual " << m_upper_actual << "\n";
			}
			outfile << "mean_mass " << mean_mass << "\n";
			outfile << "mean_mass2 " << mean_mass2 << "\n";
			if (mass_function_str != "equal")
			{
				outfile << "mean_mass_actual " << mean_mass_actual << "\n";
				outfile << "mean_mass2_actual " << mean_mass2_actual << "\n";
			}
		}
		else
		{
			outfile << "m_lower_actual " << m_lower_actual << "\n";
			outfile << "m_upper_actual " << m_upper_actual << "\n";
			outfile << "mean_mass_actual " << mean_mass_actual << "\n";
			outfile << "mean_mass2_actual " << mean_mass2_actual << "\n";
		}
		outfile << "light_loss " << light_loss << "\n";
		outfile << "num_stars " << num_stars << "\n";
		if (rectangular)
		{
			outfile << "corner_x1 " << corner.re << "\n";
			outfile << "corner_x2 " << corner.im << "\n";
			if (approx)
			{
				outfile << "taylor_smooth " << taylor_smooth << "\n";
			}
		}
		else
		{
			outfile << "rad " << corner.abs() << "\n";
		}
		outfile << "safety_scale " << safety_scale << "\n";
		outfile << "half_length " << half_length_source << "\n";
		outfile << "num_pixels " << num_pixels << "\n";
		outfile << "mean_rays_per_pixel " << num_rays_source << "\n";
		outfile << "random_seed " << random_seed << "\n";
		outfile << "lens_hl_x1 " << half_length_image.re << "\n";
		outfile << "lens_hl_x2 " << half_length_image.im << "\n";
		outfile << "ray_sep " << ray_sep << "\n";
		outfile << "t_ray_shoot " << t_ray_shoot << "\n";
		outfile.close();
		std::cout << "Done writing parameter info to file " << fname << "\n\n";


		std::cout << "Writing star info...\n";
		fname = outfile_prefix + "irs_stars" + outfile_type;
		if (!write_star_file<T>(num_stars, rectangular, corner, theta_e, stars, fname))
		{
			std::cerr << "Error. Unable to write star info to file " << fname << "\n";
			return false;
		}
		std::cout << "Done writing star info to file " << fname << "\n\n";


		/******************************************************************************
		histograms of magnification maps
		******************************************************************************/
		if (write_histograms)
		{
			std::cout << "Writing magnification histograms...\n";

			fname = outfile_prefix + "irs_numrays_numpixels.txt";
			if (!write_histogram<T>(histogram, histogram_length, *min_rays, fname))
			{
				std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
				return false;
			}
			std::cout << "Done writing magnification histogram to file " << fname << "\n";
			if (write_parities)
			{
				fname = outfile_prefix + "irs_numrays_numpixels_minima.txt";
				if (!write_histogram<T>(histogram_minima, histogram_length, *min_rays, fname))
				{
					std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
					return false;
				}
				std::cout << "Done writing magnification histogram to file " << fname << "\n";

				fname = outfile_prefix + "irs_numrays_numpixels_saddles.txt";
				if (!write_histogram<T>(histogram_saddles, histogram_length, *min_rays, fname))
				{
					std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
					return false;
				}
				std::cout << "Done writing magnification histogram to file " << fname << "\n";
			}
			std::cout << "\n";
		}


		/******************************************************************************
		write magnifications for minima, saddle, and combined maps
		******************************************************************************/
		if (write_maps)
		{
			std::cout << "Writing magnifications...\n";

			fname = outfile_prefix + "irs_magnifications" + outfile_type;
			if (!write_array<int>(pixels, num_pixels, num_pixels, fname))
			{
				std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
				return false;
			}
			std::cout << "Done writing magnifications to file " << fname << "\n";
			if (write_parities)
			{
				fname = outfile_prefix + "irs_magnifications_minima" + outfile_type;
				if (!write_array<int>(pixels_minima, num_pixels, num_pixels, fname))
				{
					std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
					return false;
				}
				std::cout << "Done writing magnifications to file " << fname << "\n";

				fname = outfile_prefix + "irs_magnifications_saddles" + outfile_type;
				if (!write_array<int>(pixels_saddles, num_pixels, num_pixels, fname))
				{
					std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
					return false;
				}
				std::cout << "Done writing magnifications to file " << fname << "\n";
			}
			std::cout << "\n";
		}

		return true;
	}
};
