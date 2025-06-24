#pragma once

#include "array_functions.cuh"
#include "binomial_coefficients.cuh"
#include "complex.cuh"
#include "fmm.cuh"
#include "mif_functions.cuh"
#include "mass_functions.cuh"
#include "mass_functions/equal.cuh"
#include "mass_functions/kroupa.cuh"
#include "mass_functions/salpeter.cuh"
#include "mass_functions/uniform.cuh"
#include "star.cuh"
#include "stopwatch.hpp"
#include "tree_node.cuh"
#include "util/math_util.cuh"
#include "util/util.cuh"

#include <curand_kernel.h>
#include <thrust/execution_policy.h> //for thrust::device
#include <thrust/extrema.h> //for thrust::min_element, thrust::max_element
#include <thrust/fill.h> //for thrust::fill
#include <thrust/functional.h> //for thrust::plus
#include <thrust/transform.h> //for thrust::transform
#include <thrust/universal_vector.h> //for thrust::universal_vector

#include <algorithm> //for std::min and std::max
#include <chrono> //for setting random seed with clock
#include <cmath>
#include <fstream> //for writing files
#include <iostream>
#include <limits> //for std::numeric_limits
#include <memory> //for std::shared_ptr
#include <numbers>
#include <numeric> //for std::reduce
#include <string>
#include <vector>


template <typename T>
class MIF
{

public:
	/******************************************************************************
	default input variables
	******************************************************************************/
	T kappa_tot = static_cast<T>(0.3);
	T shear = static_cast<T>(0.3);
	T kappa_star = static_cast<T>(0.27);
	T theta_star = static_cast<T>(1);
	std::string mass_function_str = "equal";
	T m_solar = static_cast<T>(1);
	T m_lower = static_cast<T>(0.01);
	T m_upper = static_cast<T>(50);
	T light_loss = static_cast<T>(0.001); //average fraction of light lost due to scatter by the microlenses in the large deflection angle limit
	int rectangular = 0; //whether star field is rectangular or circular
	int approx = 1; //whether terms for alpha_smooth are exact or approximate
	T safety_scale = static_cast<T>(1.37); //ratio of the size of the star field to the radius of convergence for alpha_smooth
	std::string starfile = "";
	Complex<T> w0 = Complex<T>();
	Complex<T> v = Complex<T>(2, 3);
	int random_seed = 0;
	int write_stars = 1;
	int write_images = 1;
	int write_image_line = 0;
	int write_magnifications = 1;
	std::string outfile_prefix = "./";


private:
	/******************************************************************************
	constant variables
	******************************************************************************/
	const std::string outfile_type = ".bin";
	const int MAX_TAYLOR_SMOOTH = 101; //arbitrary limit to the expansion order to avoid numerical precision loss from high degree polynomials

	/******************************************************************************
	variables for cuda device, kernel threads, and kernel blocks
	******************************************************************************/
	cudaDeviceProp cuda_device_prop;
	dim3 threads;
	dim3 blocks;

	/******************************************************************************
	stopwatch for timing purposes
	******************************************************************************/
	Stopwatch stopwatch;
	double t_elapsed;

	/******************************************************************************
	derived variables
	******************************************************************************/
	std::shared_ptr<massfunctions::MassFunction<T>> mass_function;
	T mean_mass;
	T mean_mass2;
	T mean_mass2_ln_mass;

	int num_stars;
	T kappa_star_actual;
	T m_lower_actual;
	T m_upper_actual;
	T mean_mass_actual;
	T mean_mass2_actual;
	T mean_mass2_ln_mass_actual;

	T mu_ave;
	/******************************************************************************
	maximum source plane size of the region of images visible for a macro-image
	which on average loses no more than the desired amount of flux
	******************************************************************************/
	T max_r;
	Complex<T> corner;
	int taylor_smooth;

	T alpha_error; //error in the deflection angle

	int expansion_order;

	T root_half_length;
	int tree_levels;
	std::vector<TreeNode<T>*> tree; //members of the tree will need their memory freed
	std::vector<int> num_nodes;

	/******************************************************************************
	dynamic memory
	******************************************************************************/
	curandState* states = nullptr;
	star<T>* stars = nullptr;
	star<T>* temp_stars = nullptr;

	int* binomial_coeffs = nullptr;

	Complex<T>* image_line = nullptr;
	Complex<T>* source_line = nullptr;
	T* magnifications = nullptr;
	std::vector<int> num_images;

	std::vector<Complex<T>> images;
	std::vector<Complex<T>> image_mags;



	bool set_cuda_devices(int verbose)
	{
		print_verbose("Setting device...\n", verbose, 3);

		/******************************************************************************
		check that a CUDA capable device is present
		******************************************************************************/
		int n_devices = 0;

		cudaGetDeviceCount(&n_devices);
		if (cuda_error("cudaGetDeviceCount", false, __FILE__, __LINE__)) return false;

		if (n_devices < 1)
		{
			std::cerr << "Error. No CUDA capable devices detected.\n";
			return false;
		}

		if (verbose >= 3)
		{
			std::cout << "Available CUDA capable devices:\n\n";

			for (int i = 0; i < n_devices; i++)
			{
				cudaDeviceProp prop;
				cudaGetDeviceProperties(&prop, i);
				if (cuda_error("cudaGetDeviceProperties", false, __FILE__, __LINE__)) return false;

				show_device_info(i, prop);
			}
		}

		if (n_devices > 1)
		{
			print_verbose("More than one CUDA capable device detected. Defaulting to first device.\n\n", verbose, 2);
		}
		cudaSetDevice(0);
		if (cuda_error("cudaSetDevice", false, __FILE__, __LINE__)) return false;
		cudaGetDeviceProperties(&cuda_device_prop, 0);
		if (cuda_error("cudaGetDeviceProperties", false, __FILE__, __LINE__)) return false;

		print_verbose("Done setting device.\n\n", verbose, 3);
		return true;
	}

	//optional return or not, so memory can be cleared in destructor without error checking
	bool clear_memory(int verbose, bool return_on_error = true)
	{
		print_verbose("Clearing memory...\n", verbose, 3);
		
		/******************************************************************************
		free memory and set variables to nullptr
		******************************************************************************/

		cudaFree(states);
		if (return_on_error && cuda_error("cudaFree(*states)", false, __FILE__, __LINE__)) return false;
		states = nullptr;
		
		cudaFree(stars);
		if (return_on_error && cuda_error("cudaFree(*stars)", false, __FILE__, __LINE__)) return false;
		stars = nullptr;
		
		cudaFree(temp_stars);
		if (return_on_error && cuda_error("cudaFree(*temp_stars)", false, __FILE__, __LINE__)) return false;
		temp_stars = nullptr;
		
		cudaFree(binomial_coeffs);
		if (return_on_error && cuda_error("cudaFree(*binomial_coeffs)", false, __FILE__, __LINE__)) return false;
		binomial_coeffs = nullptr;
		
		for	(int i = 0; i < tree.size(); i++) //for every level in the tree, free the memory for the nodes
		{
			cudaFree(tree[i]);
			if (return_on_error && cuda_error("cudaFree(*tree[i])", false, __FILE__, __LINE__)) return false;
			tree[i] = nullptr;
		}
		
		cudaFree(image_line);
		if (return_on_error && cuda_error("cudaFree(*image_line)", false, __FILE__, __LINE__)) return false;
		image_line = nullptr;
		
		cudaFree(source_line);
		if (return_on_error && cuda_error("cudaFree(*source_line)", false, __FILE__, __LINE__)) return false;
		source_line = nullptr;
		
		cudaFree(magnifications);
		if (return_on_error && cuda_error("cudaFree(*magnifications)", false, __FILE__, __LINE__)) return false;
		magnifications = nullptr;

		num_images.clear();
		num_images.shrink_to_fit();

		images.clear();
		images.shrink_to_fit();

		image_mags.clear();
		image_mags.shrink_to_fit();

		print_verbose("Done clearing memory.\n\n", verbose, 3);
		return true;
	}

	bool check_input_params(int verbose)
	{
		print_verbose("Checking input parameters...\n", verbose, 3);


		if (kappa_tot < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. kappa_tot must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (kappa_star < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. kappa_star must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}
		if (starfile == "" && kappa_star > kappa_tot)
		{
			std::cerr << "Error. kappa_star must be <= kappa_tot\n";
			return false;
		}

		if (starfile == "" && theta_star < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. theta_star must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (starfile == "" && !massfunctions::MASS_FUNCTIONS<T>.count(mass_function_str))
		{
			std::cerr << "Error. mass_function must be equal, uniform, Salpeter, Kroupa, or optical_depth.\n";
			return false;
		}

		if (starfile == "" && m_solar < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. m_solar must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (starfile == "" && m_lower < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. m_lower must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (starfile == "" && m_upper < m_lower)
		{
			std::cerr << "Error. m_upper must be >= m_lower.\n";
			return false;
		}

		if (light_loss < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. light_loss must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}
		else if (light_loss > 0.01)
		{
			std::cerr << "Error. light_loss must be <= 0.01\n";
			return false;
		}

		if (starfile == "" && rectangular != 0 && rectangular != 1)
		{
			std::cerr << "Error. rectangular must be 1 (rectangular) or 0 (circular).\n";
			return false;
		}

		if (approx != 0 && approx != 1)
		{
			std::cerr << "Error. approx must be 1 (approximate) or 0 (exact).\n";
			return false;
		}

		/******************************************************************************
		if the alpha_smooth comes from a rectangular mass sheet, finding the caustics
		requires a Taylor series approximation to alpha_smooth. a bound on the error of
		that series necessitates having some minimum cutoff here for the ratio of the
		size of the star field to the size of the shooting rectangle
		******************************************************************************/
		if (safety_scale < 1.1)
		{
			std::cerr << "Error. safety_scale must be >= 1.1\n";
			return false;
		}

		if (write_stars != 0 && write_stars != 1)
		{
			std::cerr << "Error. write_stars must be 1 (true) or 0 (false).\n";
			return false;
		}

		if (write_images != 0 && write_images != 1)
		{
			std::cerr << "Error. write_images must be 1 (true) or 0 (false).\n";
			return false;
		}

		if (write_image_line != 0 && write_image_line != 1)
		{
			std::cerr << "Error. write_image_line must be 1 (true) or 0 (false).\n";
			return false;
		}

		if (write_magnifications != 0 && write_magnifications != 1)
		{
			std::cerr << "Error. write_magnifications must be 1 (true) or 0 (false).\n";
			return false;
		}


		print_verbose("Done checking input parameters.\n\n", verbose, 3);

		return true;
	}

	bool calculate_derived_params(int verbose)
	{
		print_verbose("Calculating derived parameters...\n", verbose, 3);
		stopwatch.start();

		/******************************************************************************
		if star file is not specified, set the mass function, mean_mass, and
		mean_mass2
		******************************************************************************/
		if (starfile == "")
		{
			if (mass_function_str == "equal")
			{
				set_param("m_lower", m_lower, 1, verbose);
				set_param("m_upper", m_upper, 1, verbose);
			}
			else
			{
				set_param("m_lower", m_lower, m_lower * m_solar, verbose);
				set_param("m_upper", m_upper, m_upper * m_solar, verbose);
			}

			/******************************************************************************
			determine mass function, <m>, <m^2>, and <m^2 * ln(m)>
			******************************************************************************/
			mass_function = massfunctions::MASS_FUNCTIONS<T>.at(mass_function_str);
			set_param("mean_mass", mean_mass, mass_function->mean_mass(m_lower, m_upper, m_solar), verbose);
			set_param("mean_mass2", mean_mass2, mass_function->mean_mass2(m_lower, m_upper, m_solar), verbose);
			set_param("mean_mass2_ln_mass", mean_mass2_ln_mass, mass_function->mean_mass2_ln_mass(m_lower, m_upper, m_solar), verbose);
		}
		/******************************************************************************
		if star file is specified, check validity of values and set num_stars,
		rectangular, corner, theta_star, stars, kappa_star, m_lower, m_upper,
		mean_mass, and mean_mass2 based on star information
		******************************************************************************/
		else
		{
			print_verbose("Calculating some parameter values based on star input file " << starfile << "\n", verbose, 3);

			if (!read_star_file<T>(num_stars, rectangular, corner, theta_star, stars,
				kappa_star, m_lower, m_upper, mean_mass, mean_mass2, mean_mass2_ln_mass, starfile))
			{
				std::cerr << "Error. Unable to read star field parameters from file " << starfile << "\n";
				return false;
			}

			set_param("num_stars", num_stars, num_stars, verbose);
			set_param("rectangular", rectangular, rectangular, verbose);
			set_param("corner", corner, corner, verbose);
			set_param("theta_star", theta_star, theta_star, verbose);
			set_param("kappa_star", kappa_star, kappa_star, verbose);
			if (kappa_star > kappa_tot)
			{
				std::cerr << "Warning. kappa_star > kappa_tot\n";
			}
			set_param("m_lower", m_lower, m_lower, verbose);
			set_param("m_upper", m_upper, m_upper, verbose);
			set_param("mean_mass", mean_mass, mean_mass, verbose);
			set_param("mean_mass2", mean_mass2, mean_mass2, verbose);
			set_param("mean_mass2_ln_mass", mean_mass2_ln_mass, mean_mass2_ln_mass, verbose);

			print_verbose("Done calculating some parameter values based on star input file " << starfile << "\n", verbose, 3);
		}

		/******************************************************************************
		average magnification of the system
		******************************************************************************/
		set_param("mu_ave", mu_ave, 1 / ((1 - kappa_tot) * (1 - kappa_tot) - shear * shear), verbose);

		set_param("max_r", max_r, theta_star * std::sqrt(kappa_star * mean_mass2 / (mean_mass * light_loss)), verbose);


		/******************************************************************************
		if stars are not drawn from external file, calculate final number of stars to
		use and corner of the star field
		******************************************************************************/
		if (starfile == "")
		{
			corner = Complex<T>((std::abs(w0.re) + max_r) / std::abs(1 - kappa_tot + shear), 
								(std::abs(w0.im) + max_r) / std::abs(1 - kappa_tot - shear));

			if (rectangular)
			{
				num_stars = std::ceil(
					safety_scale * 2 * corner.re
					* safety_scale * 2 * corner.im
					* kappa_star / (std::numbers::pi_v<T> * theta_star * theta_star * mean_mass)
					);
				set_param("num_stars", num_stars, num_stars, verbose);

				corner = Complex<T>(std::sqrt(corner.re / corner.im), std::sqrt(corner.im / corner.re));
				corner *= std::sqrt(std::numbers::pi_v<T> * theta_star * theta_star * num_stars * mean_mass / (4 * kappa_star));
				set_param("corner", corner, corner, verbose);
			}
			else
			{
				num_stars = std::ceil(
					safety_scale * corner.abs()
					* safety_scale * corner.abs()
					* kappa_star / (theta_star * theta_star * mean_mass)
					);
				set_param("num_stars", num_stars, num_stars, verbose);

				corner = corner / corner.abs();
				corner *= std::sqrt(theta_star * theta_star * num_stars * mean_mass / kappa_star);
				set_param("corner", corner, corner, verbose);
			}
		}
		/******************************************************************************
		otherwise, check that the star file actually has a large enough field of stars
		******************************************************************************/
		else
		{
			Complex<T> tmp_corner = Complex<T>((std::abs(w0.re) + max_r) / std::abs(1 - kappa_tot + shear), 
											   (std::abs(w0.im) + max_r) / std::abs(1 - kappa_tot - shear));

			if (!rectangular)
			{
				corner = tmp_corner / tmp_corner.abs() * corner.abs();
				set_param("corner", corner, corner, verbose);
			}

			if (
				(rectangular && (corner.re < safety_scale * tmp_corner.re || corner.im < safety_scale * tmp_corner.im)) ||
				(!rectangular && (corner.abs() < safety_scale * tmp_corner.abs()))
				)
			{
				std::cerr << "Error. The provided star field is not large enough to cover the necessary image plane region.\n";
				std::cerr << "Try decreasing the safety_scale, or providing a larger field of stars.\n";
				return false;
			}
		}

		//error is 10^-7 einstein radii
		set_param("alpha_error", alpha_error, theta_star * 0.0000001, verbose, !(rectangular && approx) && verbose < 3);

		taylor_smooth = 1;
		while ((kappa_star * std::numbers::inv_pi_v<T> * 4 / (taylor_smooth + 1) * corner.abs() * (safety_scale + 1) / (safety_scale - 1)
				* std::pow(1 / safety_scale, taylor_smooth + 1) > alpha_error)
				&& taylor_smooth <= MAX_TAYLOR_SMOOTH)
		{
			taylor_smooth += 2;
		}
		/******************************************************************************
		if phase * (taylor_smooth - 1), a term in the approximation of alpha_smooth, is
		not in the correct fractional range of pi, increase taylor_smooth
		this is due to NOT wanting cos(phase * (taylor_smooth - 1)) = 0, within errors
		******************************************************************************/
		while ((std::fmod(corner.arg() * (taylor_smooth - 1), std::numbers::pi_v<T>) < 0.1 * std::numbers::pi_v<T> 
				|| std::fmod(corner.arg() * (taylor_smooth - 1), std::numbers::pi_v<T>) > 0.9 * std::numbers::pi_v<T>)
				&& taylor_smooth <= MAX_TAYLOR_SMOOTH)
		{
			taylor_smooth += 2;
		}
		set_param("taylor_smooth", taylor_smooth, taylor_smooth, verbose * (rectangular && approx), verbose < 3);
		if (rectangular && taylor_smooth > MAX_TAYLOR_SMOOTH)
		{
			std::cerr << "Error. taylor_smooth must be <= " << MAX_TAYLOR_SMOOTH << "\n";
			return false;
		}

		t_elapsed = stopwatch.stop();
		print_verbose("Done calculating derived parameters. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 3);

		return true;
	}

	bool allocate_initialize_memory(int verbose)
	{
		print_verbose("Allocating memory...\n", verbose, 3);
		stopwatch.start();

		/******************************************************************************
		allocate memory for stars
		******************************************************************************/
		cudaMallocManaged(&states, num_stars * sizeof(curandState));
		if (cuda_error("cudaMallocManaged(*states)", false, __FILE__, __LINE__)) return false;
		if (stars == nullptr) //if memory wasn't allocated already due to reading a star file
		{
			cudaMallocManaged(&stars, num_stars * sizeof(star<T>));
			if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return false;
		}
		cudaMallocManaged(&temp_stars, num_stars * sizeof(star<T>));
		if (cuda_error("cudaMallocManaged(*temp_stars)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		allocate memory for binomial coefficients
		******************************************************************************/
		cudaMallocManaged(&binomial_coeffs, (2 * treenode::MAX_EXPANSION_ORDER * (2 * treenode::MAX_EXPANSION_ORDER + 3) / 2 + 1) * sizeof(int));
		if (cuda_error("cudaMallocManaged(*binomial_coeffs)", false, __FILE__, __LINE__)) return false;

		t_elapsed = stopwatch.stop();
		print_verbose("Done allocating memory. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 3);

		return true;
	}

	bool populate_star_array(int verbose)
	{
		/******************************************************************************
		BEGIN populating star array
		******************************************************************************/

		set_threads(threads, 512);
		set_blocks(threads, blocks, num_stars);

		if (starfile == "")
		{
			print_verbose("Generating star field...\n", verbose, 1);
			stopwatch.start();

			/******************************************************************************
			if random seed was not provided, get one based on the time
			******************************************************************************/
			while (random_seed == 0) //in case it randomly chooses 0, try again
			{
				set_param("random_seed", random_seed, std::chrono::system_clock::now().time_since_epoch().count(), verbose);
			}

			/******************************************************************************
			generate random star field if no star file has been given
			******************************************************************************/
			initialize_curand_states_kernel<T> <<<blocks, threads>>> (states, num_stars, random_seed);
			if (cuda_error("initialize_curand_states_kernel", true, __FILE__, __LINE__)) return false;

			/******************************************************************************
			mass function must be a template for the kernel due to polymorphism
			we therefore must check all possible options
			******************************************************************************/
			if (mass_function_str == "equal")
			{
				generate_star_field_kernel<T, massfunctions::Equal<T>> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, m_lower, m_upper, m_solar);
			}
			else if (mass_function_str == "uniform")
			{
				generate_star_field_kernel<T, massfunctions::Uniform<T>> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, m_lower, m_upper, m_solar);
			}
			else if (mass_function_str == "salpeter")
			{
				generate_star_field_kernel<T, massfunctions::Salpeter<T>> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, m_lower, m_upper, m_solar);
			}
			else if (mass_function_str == "kroupa")
			{
				generate_star_field_kernel<T, massfunctions::Kroupa<T>> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, m_lower, m_upper, m_solar);
			}
			else if (mass_function_str == "optical_depth")
			{
				generate_star_field_kernel<T, massfunctions::OpticalDepth<T>> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, m_lower, m_upper, m_solar);
			}
			else
			{
				std::cerr << "Error. mass_function must be equal, uniform, Salpeter, Kroupa, or optical_depth.\n";
				return false;
			}
			if (cuda_error("generate_star_field_kernel", true, __FILE__, __LINE__)) return false;

			t_elapsed = stopwatch.stop();
			print_verbose("Done generating star field. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 1);
		}
		else
		{
			/******************************************************************************
			ensure random seed is 0 to denote that stars come from external file
			******************************************************************************/
			set_param("random_seed", random_seed, 0, verbose);
		}

		/******************************************************************************
		calculate kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual,
		mean_mass2_actual, and mean_mass2_ln_mass_actual based on star information
		******************************************************************************/
		calculate_star_params<T>(num_stars, rectangular, corner, theta_star, stars,
			kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual, mean_mass2_actual, mean_mass2_ln_mass_actual);

		set_param("kappa_star_actual", kappa_star_actual, kappa_star_actual, verbose);
		set_param("m_lower_actual", m_lower_actual, m_lower_actual, verbose);
		set_param("m_upper_actual", m_upper_actual, m_upper_actual, verbose);
		set_param("mean_mass_actual", mean_mass_actual, mean_mass_actual, verbose);
		set_param("mean_mass2_actual", mean_mass2_actual, mean_mass2_actual, verbose);
		set_param("mean_mass2_ln_mass_actual", mean_mass2_ln_mass_actual, mean_mass2_ln_mass_actual, verbose, starfile != "");

		if (starfile == "")
		{
			if (rectangular)
			{
				corner = Complex<T>(std::sqrt(corner.re / corner.im), std::sqrt(corner.im / corner.re));
				corner *= std::sqrt(std::numbers::pi_v<T> * theta_star * theta_star * num_stars * mean_mass_actual / (4 * kappa_star));
				set_param("corner", corner, corner, verbose, true);
			}
			else
			{
				corner = corner / corner.abs();
				corner *= std::sqrt(theta_star * theta_star * num_stars * mean_mass_actual / kappa_star);
				set_param("corner", corner, corner, verbose, true);
			}
		}

		/******************************************************************************
		END populating star array
		******************************************************************************/

		return true;
	}

	bool create_tree(int verbose)
	{
		/******************************************************************************
		BEGIN create root node, then create children and sort stars
		******************************************************************************/

		if (rectangular)
		{
			root_half_length = corner.re > corner.im ? corner.re : corner.im;
		}
		else
		{
			root_half_length = corner.abs();
		}
		set_param("root_half_length", root_half_length, root_half_length * 1.1, verbose, true); //slight buffer for containing all the stars

		//initialize variables
		tree_levels = 0;
		tree = {};
		num_nodes = {};

		/******************************************************************************
		push empty pointer into tree, add 1 to number of nodes, and allocate memory
		******************************************************************************/
		tree.push_back(nullptr);
		num_nodes.push_back(1);
		cudaMallocManaged(&tree.back(), num_nodes.back() * sizeof(TreeNode<T>));
		if (cuda_error("cudaMallocManaged(*tree)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		initialize root node
		******************************************************************************/
		tree[0][0] = TreeNode<T>(Complex<T>(0, 0), root_half_length, 0);
		tree[0][0].numstars = num_stars;


		int* max_num_stars_in_level = nullptr;
		int* min_num_stars_in_level = nullptr;
		int* num_nonempty_nodes = nullptr;
		cudaMallocManaged(&max_num_stars_in_level, sizeof(int));
		if (cuda_error("cudaMallocManaged(*max_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		cudaMallocManaged(&min_num_stars_in_level, sizeof(int));
		if (cuda_error("cudaMallocManaged(*min_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		cudaMallocManaged(&num_nonempty_nodes, sizeof(int));
		if (cuda_error("cudaMallocManaged(*num_nonempty_nodes)", false, __FILE__, __LINE__)) return false;

		print_verbose("Creating children and sorting stars...\n", verbose, 1);
		stopwatch.start();

		do
		{
			print_verbose("\nProcessing level " << tree_levels << "\n", verbose, 3);

			*max_num_stars_in_level = 0;
			*min_num_stars_in_level = num_stars;
			*num_nonempty_nodes = 0;

			set_threads(threads, 512);
			set_blocks(threads, blocks, num_nodes[tree_levels]);
			treenode::get_node_star_info_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels],
				num_nonempty_nodes, min_num_stars_in_level, max_num_stars_in_level);
			if (cuda_error("get_node_star_info_kernel", true, __FILE__, __LINE__)) return false;

			print_verbose("Maximum number of stars in a node and its neighbors is " << *max_num_stars_in_level << "\n", verbose, 3);
			print_verbose("Minimum number of stars in a node and its neighbors is " << *min_num_stars_in_level << "\n", verbose, 3);

			if (*max_num_stars_in_level > treenode::MAX_NUM_STARS_DIRECT)
			{
				print_verbose("Number of non-empty children: " << *num_nonempty_nodes * treenode::MAX_NUM_CHILDREN << "\n", verbose, 3);

				print_verbose("Allocating memory for children...\n", verbose, 3);
				tree.push_back(nullptr);
				num_nodes.push_back(*num_nonempty_nodes * treenode::MAX_NUM_CHILDREN);
				cudaMallocManaged(&tree.back(), num_nodes.back() * sizeof(TreeNode<T>));
				if (cuda_error("cudaMallocManaged(*tree)", false, __FILE__, __LINE__)) return false;

				print_verbose("Creating children...\n", verbose, 3);
				(*num_nonempty_nodes)--; //subtract one since value is size of array, and instead needs to be the first allocatable element
				set_threads(threads, 512);
				set_blocks(threads, blocks, num_nodes[tree_levels]);
				treenode::create_children_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels], num_nonempty_nodes, tree[tree_levels + 1]);
				if (cuda_error("create_children_kernel", true, __FILE__, __LINE__)) return false;

				print_verbose("Sorting stars...\n", verbose, 3);
				set_threads(threads, std::ceil(1.0 * 512 / *max_num_stars_in_level), std::min(512, *max_num_stars_in_level));
				set_blocks(threads, blocks, num_nodes[tree_levels], std::min(512, *max_num_stars_in_level));
				treenode::sort_stars_kernel<T> <<<blocks, threads, (threads.x + threads.x + threads.x * treenode::MAX_NUM_CHILDREN) * sizeof(int)>>> (tree[tree_levels], num_nodes[tree_levels], stars, temp_stars);
				if (cuda_error("sort_stars_kernel", true, __FILE__, __LINE__)) return false;

				tree_levels++;

				print_verbose("Setting neighbors...\n", verbose, 3);
				set_threads(threads, 512);
				set_blocks(threads, blocks, num_nodes[tree_levels]);
				treenode::set_neighbors_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels]);
				if (cuda_error("set_neighbors_kernel", true, __FILE__, __LINE__)) return false;
			}
		} while (*max_num_stars_in_level > treenode::MAX_NUM_STARS_DIRECT);
		print_verbose("\n", verbose, 3);
		set_param("tree_levels", tree_levels, tree_levels, verbose, verbose > 2);


		cudaFree(max_num_stars_in_level);
		if (cuda_error("cudaFree(*max_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		max_num_stars_in_level = nullptr;
		
		cudaFree(min_num_stars_in_level);
		if (cuda_error("cudaFree(*min_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		min_num_stars_in_level = nullptr;
		
		cudaFree(num_nonempty_nodes);
		if (cuda_error("cudaFree(*num_nonempty_nodes)", false, __FILE__, __LINE__)) return false;
		num_nonempty_nodes = nullptr;


		t_elapsed = stopwatch.stop();
		print_verbose("Done creating children and sorting stars. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 1);

		/******************************************************************************
		END create root node, then create children and sort stars
		******************************************************************************/

		expansion_order = std::ceil(2 * std::log2(theta_star)
									+ std::log2(mean_mass2) - std::log2(mean_mass)
									+ tree_levels
									- std::log2(root_half_length) - std::log2(alpha_error));
		set_param("expansion_order", expansion_order, expansion_order, verbose, true);
		if (expansion_order < 3)
		{
			std::cerr << "Error. Expansion order needs to be >= 3\n";
			return false;
		}
		else if (expansion_order > treenode::MAX_EXPANSION_ORDER)
		{
			std::cerr << "Error. Maximum allowed expansion order is " << treenode::MAX_EXPANSION_ORDER << "\n";
			return false;
		}

		print_verbose("Calculating binomial coefficients...\n", verbose, 3);
		calculate_binomial_coeffs(binomial_coeffs, 2 * expansion_order);
		print_verbose("Done calculating binomial coefficients.\n\n", verbose, 3);


		/******************************************************************************
		BEGIN calculating multipole and local coefficients
		******************************************************************************/

		print_verbose("Calculating multipole and local coefficients...\n", verbose, 1);
		stopwatch.start();

		for (int i = tree_levels; i >= 0; i--)
		{
			set_threads(threads, 16, expansion_order + 1);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1));
			fmm::calculate_multipole_coeffs_kernel<T> <<<blocks, threads, 16 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, stars);

			set_threads(threads, 4, expansion_order + 1, treenode::MAX_NUM_CHILDREN);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1), treenode::MAX_NUM_CHILDREN);
			fmm::calculate_M2M_coeffs_kernel<T> <<<blocks, threads, 4 * treenode::MAX_NUM_CHILDREN * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs);
		}

		/******************************************************************************
		local coefficients are non zero only starting at the second level
		******************************************************************************/
		for (int i = 2; i <= tree_levels; i++)
		{
			set_threads(threads, 16, expansion_order + 1);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1));
			fmm::calculate_L2L_coeffs_kernel<T> <<<blocks, threads, 16 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs);

			set_threads(threads, 1, expansion_order + 1, treenode::MAX_NUM_SAME_LEVEL_INTERACTION_LIST);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1), treenode::MAX_NUM_SAME_LEVEL_INTERACTION_LIST);
			fmm::calculate_M2L_coeffs_kernel<T> <<<blocks, threads, 1 * treenode::MAX_NUM_SAME_LEVEL_INTERACTION_LIST * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs);

			set_threads(threads, 4, expansion_order + 1, treenode::MAX_NUM_DIFFERENT_LEVEL_INTERACTION_LIST);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1), treenode::MAX_NUM_DIFFERENT_LEVEL_INTERACTION_LIST);
			fmm::calculate_P2L_coeffs_kernel<T> <<<blocks, threads, 4 * treenode::MAX_NUM_DIFFERENT_LEVEL_INTERACTION_LIST * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs, stars);
		}
		if (cuda_error("calculate_coeffs_kernels", true, __FILE__, __LINE__)) return false;

		t_elapsed = stopwatch.stop();
		print_verbose("Done calculating multipole and local coefficients. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 1);

		/******************************************************************************
		END calculating multipole and local coefficients
		******************************************************************************/

		return true;
	}

	bool find_image_line(int verbose)
	{
		print_verbose("Finding images...\n", verbose, 1);

        std::vector<std::vector<Complex<T>>> tmp_image_line;

        int s = 30; //scale factor for how far true root can be from the tangent estimate
        Complex<T> z, new_z, tmp_dz, dz;
        T max_dz = theta_star * std::sqrt(mean_mass2 / mean_mass) / s;
        T min_dz = max_dz / 1000;
        TreeNode<T>* node;
        T mu1, mu2;
        T dt = 1;

        thrust::universal_vector<int> use_star(num_stars, 1);
        set_threads(threads, 256);
        set_blocks(threads, blocks, num_stars);
		if (write_image_line)
		{
        	use_star_kernel<T> <<<blocks, threads>>> (stars, num_stars, kappa_tot, shear, w0, v, max_r, &use_star[0]);
        	if (cuda_error("use_star_kernel", true, __FILE__, __LINE__)) return false;
		}
		else
		{
			int N_ANGLES = 1000;
			for (int i = 0; i <= N_ANGLES; i++)
			{
        		use_star_kernel<T> <<<blocks, threads>>> (stars, num_stars, kappa_tot, shear, w0, 
														  v * Complex<T>(std::cos(std::numbers::pi_v<T> / (2 * N_ANGLES) * i),
																		 std::sin(std::numbers::pi_v<T> / (2 * N_ANGLES) * i)),
														  max_r, &use_star[0]);
        		if (cuda_error("use_star_kernel", true, __FILE__, __LINE__)) return false;
			}
		}
        
		/******************************************************************************
		BEGIN finding main image line
		******************************************************************************/
		print_verbose("Finding main image line...\n", verbose, 2);
        stopwatch.start();

        //start in a position such that the velocity vector takes you through the field
        z = Complex<T>(-v.re / (1 - kappa_tot + shear), 
            		   -v.im / (1 - kappa_tot - shear));
        z = z / z.abs() * corner.abs() * safety_scale;

        tmp_image_line.push_back(std::vector<Complex<T>>());

        z = find_root<T>(z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
						 rectangular, corner, approx, taylor_smooth, w0, v);
        tmp_image_line.back().push_back(z);

        node = treenode::get_nearest_node(z, tree[0]);
        mu1 = microlensing::mu<T>(z, kappa_tot, shear, theta_star, stars, kappa_star, node,
                        		  rectangular, corner, approx, taylor_smooth);
        do
        {
            new_z = step_tangent<T>(z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                        rectangular, corner, approx, taylor_smooth, w0, v, dt, min_dz, max_dz);
            tmp_dz = (new_z - z);
            new_z = find_root<T>(new_z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                        rectangular, corner, approx, taylor_smooth, w0, v);
            dz = (new_z - z);

            //if actual root is too large relative to the initial tangent step
            if ((dz - tmp_dz).abs() * s > tmp_dz.abs())
            {
                dt /= 2;
                continue;
            }
            //if actual root is very close to the tangent step
            else if ((dz - tmp_dz).abs() * s * s < tmp_dz.abs())
            {
                dt *= 2;
            }

            node = treenode::get_nearest_node(new_z, tree[0]);
            mu2 = microlensing::mu<T>(new_z, kappa_tot, shear, theta_star, stars, kappa_star, node,
                            		  rectangular, corner, approx, taylor_smooth);
            
            if ((mu1 < 0 && mu2 > 0 )
                || (mu1 > 0 && mu2 < 0))
            {
                tmp_dz = dz;
                new_z = z - mu1 / (mu2 - mu1) * dz;
                new_z = find_critical_curve_image<T>(new_z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                            rectangular, corner, approx, taylor_smooth, w0, v);
                dz = (new_z - z);
                dt *= dz.abs() / tmp_dz.abs();
                mu2 = 0;
            }           
            z = new_z;
            mu1 = mu2;
            tmp_image_line.back().push_back(z);

            if (is_near_star(z, stars, tree[0], dz)
                && is_near_star(z - dz, stars, tree[0], dz))
            {
                int index = get_nearest_star(z - dz / 2, stars, tree[0]);
                use_star[index] = 0;
            }
        } while (z.abs() < safety_scale * corner.abs());
        t_elapsed = stopwatch.stop();
        print_verbose("Done finding main image line. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 2);
		/******************************************************************************
		END finding main image line
		******************************************************************************/

        
		/******************************************************************************
		BEGIN finding secondary image loops
		******************************************************************************/
		print_verbose("Finding secondary image loops...\n", verbose, 2);
        stopwatch.start();
        for (int i = 0; i < num_stars; i++)
        {
            if (use_star[i])
            {
                use_star[i] = 0;
                dz = Complex<T>(-v.re / (1 - kappa_tot + shear), 
                                -v.im / (1 - kappa_tot - shear));
                max_dz = theta_star * theta_star * stars[i].mass / std::abs(macro_parametric_image_line(stars[i].position, kappa_tot, shear, w0, v));
                max_dz /= 2 * s; //diameter to radius
                max_dz = std::min(max_dz, theta_star * std::sqrt(mean_mass2 / mean_mass) / s);
                min_dz = max_dz / 1000;
                dz *= min_dz / dz.abs();
                z = stars[i].position + dz;

                tmp_image_line.push_back(std::vector<Complex<T>>());

                z = find_root<T>(z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                                    rectangular, corner, approx, taylor_smooth, w0, v);
                tmp_image_line.back().push_back(z);

                node = treenode::get_nearest_node(z, tree[0]);
                mu1 = microlensing::mu<T>(z, kappa_tot, shear, theta_star, stars, kappa_star, node,
                                		  rectangular, corner, approx, taylor_smooth);
                do
                {
                    new_z = step_tangent<T>(z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                                			rectangular, corner, approx, taylor_smooth, w0, v, dt, min_dz, max_dz);
                    tmp_dz = (new_z - z);

                    if (is_near_star(new_z, stars, tree[0], tmp_dz)
                        && is_near_star(new_z - tmp_dz, stars, tree[0], tmp_dz))
                    {
                        int index = get_nearest_star(new_z - tmp_dz / 2, stars, tree[0]);
                        use_star[index] = 0;
                        if (index == i)
                        {
                            break;
                        }
                    }

                    new_z = find_root<T>(new_z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                                		 rectangular, corner, approx, taylor_smooth, w0, v);
                    dz = (new_z - z);
                    if ((dz - tmp_dz).abs() * s > tmp_dz.abs())
                    {
                        dt /= 2;
                        continue;
                    }
                    else if ((dz - tmp_dz).abs() * s * s < tmp_dz.abs())
                    {
                        dt *= 2;
                    }

                    node = treenode::get_nearest_node(new_z, tree[0]);
                    mu2 = microlensing::mu<T>(new_z, kappa_tot, shear, theta_star, stars, kappa_star, node,
                                    		  rectangular, corner, approx, taylor_smooth);
                    
                    if ((mu1 < 0 && mu2 > 0 )
                        || (mu1 > 0 && mu2 < 0))
                    {
                        tmp_dz = dz;
                        new_z = z - mu1 / (mu2 - mu1) * dz;
                        new_z = find_critical_curve_image<T>(new_z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                                    rectangular, corner, approx, taylor_smooth, w0, v);
                        dz = (new_z - z);
                        dt *= dz.abs() / tmp_dz.abs();
                        mu2 = 0;
                    }           
                    z = new_z;
                    mu1 = mu2;
                    tmp_image_line.back().push_back(z);
                } while (true);
            }
        }
        t_elapsed = stopwatch.stop();
        print_verbose("Done finding secondary image loops. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 2);
		/******************************************************************************
		END finding secondary image loops
		******************************************************************************/

        print_verbose("Done finding images.\n\n", verbose, 1);


        for (int i = 0; i < tmp_image_line.size(); i++)
        {
            num_images.push_back(tmp_image_line[i].size());
        }

        int total_num_images = std::reduce(num_images.begin(), num_images.end(), 0);
        
        cudaMallocManaged(&image_line, total_num_images * sizeof(Complex<T>));
        if (cuda_error("cudaMallocManaged(*image_line)", false, __FILE__, __LINE__)) return false;
        cudaMallocManaged(&source_line, total_num_images * sizeof(Complex<T>));
        if (cuda_error("cudaMallocManaged(*source_line)", false, __FILE__, __LINE__)) return false;
        cudaMallocManaged(&magnifications, total_num_images * sizeof(T));
        if (cuda_error("cudaMallocManaged(*magnifications)", false, __FILE__, __LINE__)) return false;

        print_verbose("Copying image lines...\n", verbose, 2);
        stopwatch.start();
        for (int i = 0; i < tmp_image_line.size(); i++)
        {
            int start = std::reduce(&num_images[0], &num_images[i], 0);
            thrust::copy(tmp_image_line[i].begin(), tmp_image_line[i].end(), &image_line[start]);
        }
        t_elapsed = stopwatch.stop();
        print_verbose("Done copying image lines. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 2);


        set_threads(threads, 256);
        set_blocks(threads, blocks, total_num_images);

        print_verbose("Mapping image lines...\n", verbose, 2);
        stopwatch.start();
        image_to_source_kernel<T> <<<blocks, threads>>> (image_line, total_num_images, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
            rectangular, corner, approx, taylor_smooth, source_line);
        if (cuda_error("image_to_source_kernel", true, __FILE__, __LINE__)) return false;
        t_elapsed = stopwatch.stop();
        print_verbose("Done mapping image lines. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 2);

        print_verbose("Calculating magnifications...\n", verbose, 2);
        stopwatch.start();
        magnifications_kernel<T> <<<blocks, threads>>> (image_line, total_num_images, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
            rectangular, corner, approx, taylor_smooth, magnifications);
        if (cuda_error("magnifications_kernel", true, __FILE__, __LINE__)) return false;
        t_elapsed = stopwatch.stop();
        print_verbose("Done calculating magnifications. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 2);
		
		return true;
	}

	bool find_point_images(int verbose)
	{
		print_verbose("Finding point images...\n", verbose, 1);

		Complex<T> z1, z2, dz;
		T f1, f2;
		TreeNode<T>* node;
		
        stopwatch.start();
		for (int i =0; i < num_images.size(); i++)
		{
			int start = std::reduce(&num_images[0], &num_images[i], 0);

			for (int j = 0; j < num_images[i] - 1; j++)
			{
				z1 = image_line[start + j];
				z2 = image_line[start + j + 1];
				dz = (z2 - z1);

				node = treenode::get_nearest_node(z2, tree[0]);
				f2 = parametric_image_line(z2, kappa_tot, shear, theta_star, stars, kappa_star, node,
										   rectangular, corner, approx, taylor_smooth, w0, v * Complex<T>(0, 1));
				node = treenode::get_nearest_node(z1, tree[0]);
				f1 = parametric_image_line(z1, kappa_tot, shear, theta_star, stars, kappa_star, node,
										   rectangular, corner, approx, taylor_smooth, w0, v * Complex<T>(0, 1));
				
				//if we are crossing a star, the sign changes
				bool is_star = (is_near_star(z1, stars, tree[0], dz) && is_near_star(z2, stars, tree[0], dz));
				
				if (sgn(f1) != sgn(f2) && !is_star)
				{
					T dt = -f1 / (f2 - f1);
					images.push_back(z1 + dz * dt);
				}

			}
		}
		print_verbose("Number of point images: " << images.size() << "\n", verbose, 2);
		for (int i = 0; i < images.size(); i++)
		{
			images[i] = find_point_image(images[i], kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
					rectangular, corner, approx, taylor_smooth, w0, v);
		}
        t_elapsed = stopwatch.stop();
		print_verbose("Done finding point images.\n\n", verbose, 1);
		
		return true;
	}

	bool write_files(int verbose)
	{
		/******************************************************************************
		stream for writing output files
		set precision to 9 digits
		******************************************************************************/
		std::ofstream outfile;
		outfile.precision(9);
		std::string fname;


		print_verbose("Writing parameter info...\n", verbose, 2);
		fname = outfile_prefix + "mif_parameter_info.txt";
		outfile.open(fname);
		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		outfile << "kappa_tot " << kappa_tot << "\n";
		outfile << "shear " << shear << "\n";
		outfile << "mu_ave " << mu_ave << "\n";
		outfile << "smooth_fraction " << (1 - kappa_star / kappa_tot) << "\n";
		outfile << "kappa_star " << kappa_star << "\n";
		if (starfile == "")
		{
			outfile << "kappa_star_actual " << kappa_star_actual << "\n";
		}
		outfile << "theta_star " << theta_star << "\n";
		outfile << "random_seed " << random_seed << "\n";
		if (starfile == "")
		{
			outfile << "mass_function " << mass_function_str << "\n";
			if (mass_function_str == "salpeter" || mass_function_str == "kroupa")
			{
				outfile << "m_solar " << m_solar << "\n";
			}
			outfile << "m_lower " << m_lower << "\n";
			outfile << "m_upper " << m_upper << "\n";
			outfile << "mean_mass " << mean_mass << "\n";
			outfile << "mean_mass2 " << mean_mass2 << "\n";
			outfile << "mean_mass2_ln_mass " << mean_mass2_ln_mass << "\n";
		}
		outfile << "m_lower_actual " << m_lower_actual << "\n";
		outfile << "m_upper_actual " << m_upper_actual << "\n";
		outfile << "mean_mass_actual " << mean_mass_actual << "\n";
		outfile << "mean_mass2_actual " << mean_mass2_actual << "\n";
		outfile << "mean_mass2_ln_mass_actual " << mean_mass2_ln_mass_actual << "\n";
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
		outfile << "w0_1 " << w0.re << "\n";
		outfile << "w0_2 " << w0.im << "\n";
		outfile << "v_1 " << v.re << "\n";
		outfile << "v_2 " << v.im << "\n";
		outfile << "alpha_error " << alpha_error << "\n";
		outfile << "expansion_order " << expansion_order << "\n";
		outfile << "root_half_length " << root_half_length << "\n";
		outfile << "tree_levels " << tree_levels << "\n";
		outfile.close();
		print_verbose("Done writing parameter info to file " << fname << "\n", verbose, 1);
		print_verbose("\n", verbose * (write_stars || write_images || write_image_line), 2);


		if (write_stars)
		{
			print_verbose("Writing star info...\n", verbose, 2);
			fname = outfile_prefix + "mif_stars" + outfile_type;
			if (!write_star_file<T>(num_stars, rectangular, corner, theta_star, stars, fname))
			{
				std::cerr << "Error. Unable to write star info to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing star info to file " << fname << "\n", verbose, 1);
			print_verbose("\n", verbose * (write_images || write_image_line), 2);
		}

		if (write_images)
		{
			print_verbose("Writing point images...\n", verbose, 2);
			fname = outfile_prefix + "mif_images" + outfile_type;
			if (!write_array<Complex<T>>(&images[0], 1, images.size(), fname))
			{
				std::cerr << "Error. Unable to write point images to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing point images to file " << fname << "\n", verbose, 1);
			print_verbose("\n", verbose * write_image_line, 2);
		}

		if (write_image_line)
		{
			print_verbose("Writing image line...\n", verbose, 2);
			fname = outfile_prefix + "mif_image_line" + outfile_type;
			if (!write_ragged_array<Complex<T>>(image_line, num_images, fname))
			{
				std::cerr << "Error. Unable to write images to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing image line to file " << fname << "\n", verbose, 1);

			print_verbose("Writing source line...\n", verbose, 2);
			fname = outfile_prefix + "mif_source_line" + outfile_type;
			if (!write_ragged_array<Complex<T>>(source_line, num_images, fname))
			{
				std::cerr << "Error. Unable to write sources to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing source line to file " << fname << "\n", verbose, 1);

			if (write_magnifications)
			{
				print_verbose("Writing image line magnifications...\n", verbose, 2);
				fname = outfile_prefix + "mif_image_line_magnifications" + outfile_type;
				if (!write_ragged_array<T>(magnifications, num_images, fname))
				{
					std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing image line magnifications to file " << fname << "\n", verbose, 1);
			}
		}

		return true;
	}


public:

	/******************************************************************************
	class initializer is empty
	******************************************************************************/
	MIF()
	{

	}

	/******************************************************************************
	class destructor clears memory with no output or error checking
	******************************************************************************/
	~MIF()
	{
		clear_memory(0, false);
	}

	/******************************************************************************
	copy constructor sets this object's dynamic memory pointers to null
	******************************************************************************/
	MIF(const MIF& other)
	{
		states = nullptr;
		stars = nullptr;
		temp_stars = nullptr;

		binomial_coeffs = nullptr;

		tree = {};
	}

	/******************************************************************************
	copy assignment sets this object's dynamic memory pointers to null
	******************************************************************************/
	MIF& operator=(const MIF& other)
	{
        if (this == &other) return *this;
		
		states = nullptr;
		stars = nullptr;
		temp_stars = nullptr;

		binomial_coeffs = nullptr;

		tree = {};

		return *this;
	}

	bool run(int verbose)
	{
		if (!set_cuda_devices(verbose)) return false;
		if (!clear_memory(verbose)) return false;
		if (!check_input_params(verbose)) return false;
		if (!calculate_derived_params(verbose)) return false;
		if (!allocate_initialize_memory(verbose)) return false;
		if (!populate_star_array(verbose)) return false;
		if (!create_tree(verbose)) return false;
		if (!find_image_line(verbose)) return false;
		if (!find_point_images(verbose)) return false;

		return true;
	}

	bool save(int verbose)
	{
		if (!write_files(verbose)) return false;

		return true;
	}

	int get_num_stars()			{return num_stars;}
	Complex<T> get_corner()		{if (rectangular) {return corner;} else {return Complex<T>(corner.abs(), 0);}}
	star<T>* get_stars()		{return stars;}

};

