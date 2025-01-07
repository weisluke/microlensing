#pragma once

#include "array_functions.cuh"
#include "binomial_coefficients.cuh"
#include "complex.cuh"
#include "fmm.cuh"
#include "ipm_functions.cuh"
#include "mass_functions.cuh"
#include "mass_functions/equal.cuh"
#include "mass_functions/kroupa.cuh"
#include "mass_functions/salpeter.cuh"
#include "mass_functions/uniform.cuh"
#include "star.cuh"
#include "stopwatch.hpp"
#include "tree_node.cuh"
#include "util.cuh"

#include <curand_kernel.h>
#include <thrust/execution_policy.h> //for thrust::device
#include <thrust/extrema.h> //for thrust::min_element, thrust::max_element

#include <algorithm> //for std::min and std::max
#include <chrono> //for setting random seed with clock
#include <cmath>
#include <fstream> //for writing files
#include <iostream>
#include <limits> //for std::numeric_limits
#include <memory> //for std::shared_ptr
#include <numbers>
#include <string>
#include <vector>


template <typename T>
class IPM
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
	T safety_scale = static_cast<T>(1.37); //ratio of the size of the star field to the size of the shooting region
	std::string starfile = "";
	Complex<T> center_y = Complex<T>();
	Complex<T> half_length_y = Complex<T>(5, 5);
	Complex<int> num_pixels_y = Complex<int>(1000, 1000);
	int num_rays_y = 1; //number density of rays per pixel in the source plane
	int random_seed = 0;
	int write_stars = 1;
	int write_maps = 1;
	int write_parities = 0;
	int write_histograms = 1;
	std::string outfile_prefix = "./";


	/******************************************************************************
	class initializer is empty
	******************************************************************************/
	IPM()
	{

	}


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
	double t_shoot_cells;

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
	T num_rays_x; //number density of rays per unit area in the image plane
	Complex<T> ray_half_sep; //distance from center of cell to corner
	Complex<T> center_x;
	Complex<T> half_length_x;
	Complex<int> num_ray_threads; //number of threads in x1 and x2 directions 
	Complex<T> corner;
	int taylor_smooth;

	T alpha_error; //error in the deflection angle

	int expansion_order;

	T root_half_length;
	int tree_levels;
	std::vector<TreeNode<T>*> tree;
	std::vector<int> num_nodes;

	/******************************************************************************
	dynamic memory
	******************************************************************************/
	curandState* states;
	star<T>* stars;
	star<T>* temp_stars;

	int* binomial_coeffs;

	T* pixels;
	T* pixels_minima;
	T* pixels_saddles;

	int min_mag;
	int max_mag;
	int histogram_length;
	int* histogram;
	int* histogram_minima;
	int* histogram_saddles;

	int min_log_mag;
	int max_log_mag;
	int log_histogram_length;
	int* log_histogram;
	int* log_histogram_minima;
	int* log_histogram_saddles;



	bool clear_memory(int verbose)
	{
		print_verbose("Clearing memory...\n", verbose, 3);

		cudaDeviceReset(); //free all previously allocated memory
		if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return false;
		
		//and set variables to nullptr
		states = nullptr;
		stars = nullptr;
		temp_stars = nullptr;

		binomial_coeffs = nullptr;

		pixels = nullptr;
		pixels_minima = nullptr;
		pixels_saddles = nullptr;

		histogram = nullptr;
		histogram_minima = nullptr;
		histogram_saddles = nullptr;

		log_histogram = nullptr;
		log_histogram_minima = nullptr;
		log_histogram_saddles = nullptr;

		print_verbose("Done clearing memory.\n\n", verbose, 3);
		return true;
	}

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

		if (half_length_y.re < std::numeric_limits<T>::min() || half_length_y.im < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. half_length_y1 and half_length_y2 must both be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (num_pixels_y.re < 1 || num_pixels_y.im < 1)
		{
			std::cerr << "Error. num_pixels_y1 and num_pixels_y2 must both be integers > 0\n";
			return false;
		}

		if (num_rays_y < 1)
		{
			std::cerr << "Error. num_rays_y must be an integer > 0\n";
			return false;
		}

		if (write_stars != 0 && write_stars != 1)
		{
			std::cerr << "Error. write_stars must be 1 (true) or 0 (false).\n";
			return false;
		}

		if (write_maps != 0 && write_maps != 1)
		{
			std::cerr << "Error. write_maps must be 1 (true) or 0 (false).\n";
			return false;
		}

		if (write_parities != 0 && write_parities != 1)
		{
			std::cerr << "Error. write_parities must be 1 (true) or 0 (false).\n";
			return false;
		}

		if (write_histograms != 0 && write_histograms != 1)
		{
			std::cerr << "Error. write_histograms must be 1 (true) or 0 (false).\n";
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

		/******************************************************************************
		number density of rays in the lens plane
		******************************************************************************/
		set_param("num_rays_x", num_rays_x,
			1.0 * num_rays_y * num_pixels_y.re * num_pixels_y.im / (2 * half_length_y.re * 2 * half_length_y.im),
			verbose);

		/******************************************************************************
		average area covered by one ray is 1 / number density
		account for potential rectangular pixels and use half_length
		******************************************************************************/
		ray_half_sep = Complex<T>(std::sqrt(half_length_y.re / half_length_y.im * num_pixels_y.im / num_pixels_y.re),
							 std::sqrt(half_length_y.im / half_length_y.re * num_pixels_y.re / num_pixels_y.im));
		ray_half_sep /= (2 * std::sqrt(num_rays_x));
		set_param("ray_half_sep", ray_half_sep, ray_half_sep, verbose);

		/******************************************************************************
		shooting region is greater than outer boundary for macro-mapping by the size of
		the region of images visible for a macro-image which on average loses no more
		than the desired amount of flux
		******************************************************************************/
		half_length_x = half_length_y + theta_star * std::sqrt(kappa_star * mean_mass2 / (mean_mass * light_loss)) * Complex<T>(1, 1);
		half_length_x = Complex<T>(half_length_x.re / std::abs(1 - kappa_tot + shear), half_length_x.im / std::abs(1 - kappa_tot - shear));
		/******************************************************************************
		make shooting region a multiple of the ray separation
		******************************************************************************/
		num_ray_threads = Complex<int>(half_length_x.re / (2 * ray_half_sep.re), half_length_x.im / (2 * ray_half_sep.im)) + Complex<int>(1, 1);
		half_length_x = Complex<T>(2 * ray_half_sep.re * num_ray_threads.re, 2 * ray_half_sep.im * num_ray_threads.im);
		set_param("half_length_x", half_length_x, half_length_x, verbose);
		set_param("num_ray_threads", num_ray_threads, 2 * num_ray_threads, verbose);

		center_x = Complex<T>(center_y.re / (1 - kappa_tot + shear), center_y.im / (1 - kappa_tot - shear));
		set_param("center_x", center_x, center_x, verbose);

		/******************************************************************************
		if stars are not drawn from external file, calculate final number of stars to
		use and corner of the star field
		******************************************************************************/
		if (starfile == "")
		{
			corner = Complex<T>(std::abs(center_x.re) + half_length_x.re, std::abs(center_x.im) + half_length_x.im);

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
			Complex<T> tmp_corner = Complex<T>(std::abs(center_x.re) + half_length_x.re, std::abs(center_x.im) + half_length_x.im);

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
				std::cerr << "Error. The provided star field is not large enough to cover the desired source plane region.\n";
				std::cerr << "Try decreasing the safety_scale, or providing a larger field of stars.\n";
				return false;
			}
		}

		alpha_error = std::min(half_length_y.re / (10 * num_pixels_y.re), 
			half_length_y.im / (10 * num_pixels_y.im)); //error is a circle of radius 1/10 of smallest pixel scale
		set_param("alpha_error", alpha_error, alpha_error, verbose, !(rectangular && approx) && verbose < 3);

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

		/******************************************************************************
		allocate memory for pixels
		******************************************************************************/
		cudaMallocManaged(&pixels, num_pixels_y.re * num_pixels_y.im * sizeof(T));
		if (cuda_error("cudaMallocManaged(*pixels)", false, __FILE__, __LINE__)) return false;
		if (write_parities)
		{
			cudaMallocManaged(&pixels_minima, num_pixels_y.re * num_pixels_y.im * sizeof(T));
			if (cuda_error("cudaMallocManaged(*pixels_minima)", false, __FILE__, __LINE__)) return false;
			cudaMallocManaged(&pixels_saddles, num_pixels_y.re * num_pixels_y.im * sizeof(T));
			if (cuda_error("cudaMallocManaged(*pixels_saddles)", false, __FILE__, __LINE__)) return false;
		}

		t_elapsed = stopwatch.stop();
		print_verbose("Done allocating memory. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 3);


		/******************************************************************************
		initialize pixel values
		******************************************************************************/
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_pixels_y.re, num_pixels_y.im);

		print_verbose("Initializing array values...\n", verbose, 3);
		stopwatch.start();

		initialize_array_kernel<T> <<<blocks, threads>>> (pixels, num_pixels_y.im, num_pixels_y.re);
		if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
		if (write_parities)
		{
			initialize_array_kernel<T> <<<blocks, threads>>> (pixels_minima, num_pixels_y.im, num_pixels_y.re);
			if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
			initialize_array_kernel<T> <<<blocks, threads>>> (pixels_saddles, num_pixels_y.im, num_pixels_y.re);
			if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
		}

		t_elapsed = stopwatch.stop();
		print_verbose("Done initializing array values. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 3);

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


		int* max_num_stars_in_level;
		int* min_num_stars_in_level;
		int* num_nonempty_nodes;
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

		t_elapsed = stopwatch.stop();
		print_verbose("Done creating children and sorting stars. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 1);

		/******************************************************************************
		END create root node, then create children and sort stars
		******************************************************************************/

		expansion_order = std::ceil(2 * std::log2(theta_star) + tree_levels
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

	bool shoot_cells(int verbose)
	{
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_ray_threads.re, num_ray_threads.im);

		unsigned long long int* percentage = nullptr;
		cudaMallocManaged(&percentage, sizeof(unsigned long long int));
		if (cuda_error("cudaMallocManaged(*percentage)", false, __FILE__, __LINE__)) return false;

		*percentage = 1;

		/******************************************************************************
		shoot rays and calculate time taken in seconds
		******************************************************************************/
		print_verbose("Shooting cells...\n", verbose, 1);
		stopwatch.start();
		shoot_cells_kernel<T> <<<blocks, threads>>> (kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
			rectangular, corner, approx, taylor_smooth, ray_half_sep, num_ray_threads, center_x, half_length_x,
			center_y, half_length_y, pixels_minima, pixels_saddles, pixels, num_pixels_y, percentage, verbose);
		if (cuda_error("shoot_rays_kernel", true, __FILE__, __LINE__)) return false;
		t_shoot_cells = stopwatch.stop();
		print_verbose("\nDone shooting cells. Elapsed time: " << t_shoot_cells << " seconds.\n\n", verbose, 1);

		if (write_parities)
		{
			print_verbose("Adding arrays...\n", verbose, 2);
			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels_y.re, num_pixels_y.im);
			add_arrays_kernel<T> <<<blocks, threads>>> (pixels_minima, pixels_saddles, pixels, num_pixels_y.im, num_pixels_y.re);
			if (cuda_error("add_arrays_kernel", true, __FILE__, __LINE__)) return false;
			print_verbose("Done adding arrays.\n\n", verbose, 2);
		}

		return true;
	}

	bool create_histograms(int verbose)
	{
		/******************************************************************************
		create histograms of pixel values
		******************************************************************************/

		if (write_histograms)
		{
			print_verbose("Creating histograms...\n", verbose, 2);
			stopwatch.start();

			/******************************************************************************
			factor by which to multiply values before casting to integers
			in this way, the histogram will be of the value * 1000
			(i.e., accurate to 3 decimals)
			******************************************************************************/
			int factor = 1000;


			/******************************************************************************
			histogram of mu
			******************************************************************************/
			min_mag = std::round(*thrust::min_element(thrust::device, pixels, pixels + num_pixels_y.re * num_pixels_y.im) * factor);
			max_mag = std::round(*thrust::max_element(thrust::device, pixels, pixels + num_pixels_y.re * num_pixels_y.im) * factor);

			T mu_min_theory = 1 / ((1 - kappa_tot + kappa_star) * (1 - kappa_tot + kappa_star));
			T mu_min_actual = 1.0 * min_mag / factor;

			if (mu_ave > 1 && mu_min_actual < mu_min_theory)
			{
				std::cerr << "Warning. Minimum magnification after shooting cells is less than the theoretical minimum.\n";
				std::cerr << "   mu_min_actual = " << mu_min_actual << "\n";
				std::cerr << "   mu_min_theory = 1 / (1 - (kappa_tot - kappa_star))^2\n";
				std::cerr << "                 = 1 / (1 - (" << kappa_tot << " - " << kappa_star << "))^2 = " << mu_min_theory << "\n";
				print_verbose("\n", verbose * (!write_parities && verbose < 2), 1);
			}

			if (write_parities)
			{
				int min_mag_minima = std::round(*thrust::min_element(thrust::device, pixels_minima, pixels_minima + num_pixels_y.re * num_pixels_y.im) * factor);
				int max_mag_minima = std::round(*thrust::max_element(thrust::device, pixels_minima, pixels_minima + num_pixels_y.re * num_pixels_y.im) * factor);

				mu_min_actual = 1.0 * min_mag_minima / factor;

				if (mu_ave > 1 && mu_min_actual < mu_min_theory)
				{
					std::cerr << "Warning. Minimum positive parity magnification after shooting cells is less than the theoretical minimum.\n";
					std::cerr << "   mu_min_actual = " << mu_min_actual << "\n";
					std::cerr << "   mu_min_theory = 1 / (1 - (kappa_tot - kappa_star))^2\n";
					std::cerr << "                 = 1 / (1 - (" << kappa_tot << " - " << kappa_star << "))^2 = " << mu_min_theory << "\n";
					print_verbose("\n", verbose * (verbose < 2), 1);
				}

				int min_mag_saddles = std::round(*thrust::min_element(thrust::device, pixels_saddles, pixels_saddles + num_pixels_y.re * num_pixels_y.im) * factor);
				int max_mag_saddles = std::round(*thrust::max_element(thrust::device, pixels_saddles, pixels_saddles + num_pixels_y.re * num_pixels_y.im) * factor);

				min_mag = std::min({min_mag, min_mag_minima, min_mag_saddles});
				max_mag = std::max({max_mag, max_mag_minima, max_mag_saddles});
			}

			histogram_length = max_mag - min_mag + 1;

			cudaMallocManaged(&histogram, histogram_length * sizeof(int));
			if (cuda_error("cudaMallocManaged(*histogram)", false, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				cudaMallocManaged(&histogram_minima, histogram_length * sizeof(int));
				if (cuda_error("cudaMallocManaged(*histogram_minima)", false, __FILE__, __LINE__)) return false;
				cudaMallocManaged(&histogram_saddles, histogram_length * sizeof(int));
				if (cuda_error("cudaMallocManaged(*histogram_saddles)", false, __FILE__, __LINE__)) return false;
			}


			set_threads(threads, 512);
			set_blocks(threads, blocks, histogram_length);

			initialize_array_kernel<int> <<<blocks, threads>>> (histogram, 1, histogram_length);
			if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				initialize_array_kernel<int> <<<blocks, threads>>> (histogram_minima, 1, histogram_length);
				if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
				initialize_array_kernel<int> <<<blocks, threads>>> (histogram_saddles, 1, histogram_length);
				if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
			}


			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels_y.re, num_pixels_y.im);

			histogram_kernel<T> <<<blocks, threads>>> (pixels, num_pixels_y, min_mag, histogram, factor);
			if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				histogram_kernel<T> <<<blocks, threads>>> (pixels_minima, num_pixels_y, min_mag, histogram_minima, factor);
				if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return false;
				histogram_kernel<T> <<<blocks, threads>>> (pixels_saddles, num_pixels_y, min_mag, histogram_saddles, factor);
				if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return false;
			}
			

			/******************************************************************************
			histogram of log_10(mu)
			******************************************************************************/

			min_log_mag = std::round(std::log10(*thrust::min_element(thrust::device, pixels, pixels + num_pixels_y.re * num_pixels_y.im)) * factor);
			max_log_mag = std::round(std::log10(*thrust::max_element(thrust::device, pixels, pixels + num_pixels_y.re * num_pixels_y.im)) * factor);

			if (write_parities)
			{
				int min_log_mag_minima = std::round(std::log10(*thrust::min_element(thrust::device, pixels_minima, pixels_minima + num_pixels_y.re * num_pixels_y.im)) * factor);
				int max_log_mag_minima = std::round(std::log10(*thrust::max_element(thrust::device, pixels_minima, pixels_minima + num_pixels_y.re * num_pixels_y.im)) * factor);

				int min_log_mag_saddles = std::round(std::log10(*thrust::min_element(thrust::device, pixels_saddles, pixels_saddles + num_pixels_y.re * num_pixels_y.im)) * factor);
				int max_log_mag_saddles = std::round(std::log10(*thrust::max_element(thrust::device, pixels_saddles, pixels_saddles + num_pixels_y.re * num_pixels_y.im)) * factor);

				min_log_mag = std::min({min_log_mag, min_log_mag_minima, min_log_mag_saddles});
				max_log_mag = std::max({max_log_mag, max_log_mag_minima, max_log_mag_saddles});
			}

			log_histogram_length = max_log_mag - min_log_mag + 1;

			cudaMallocManaged(&log_histogram, log_histogram_length * sizeof(int));
			if (cuda_error("cudaMallocManaged(*log_histogram)", false, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				cudaMallocManaged(&log_histogram_minima, log_histogram_length * sizeof(int));
				if (cuda_error("cudaMallocManaged(*log_histogram_minima)", false, __FILE__, __LINE__)) return false;
				cudaMallocManaged(&log_histogram_saddles, log_histogram_length * sizeof(int));
				if (cuda_error("cudaMallocManaged(*log_histogram_saddles)", false, __FILE__, __LINE__)) return false;
			}


			set_threads(threads, 512);
			set_blocks(threads, blocks, log_histogram_length);

			initialize_array_kernel<int> <<<blocks, threads>>> (log_histogram, 1, log_histogram_length);
			if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				initialize_array_kernel<int> <<<blocks, threads>>> (log_histogram_minima, 1, log_histogram_length);
				if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
				initialize_array_kernel<int> <<<blocks, threads>>> (log_histogram_saddles, 1, log_histogram_length);
				if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
			}


			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels_y.re, num_pixels_y.im);

			log_histogram_kernel<T> <<<blocks, threads>>> (pixels, num_pixels_y, min_log_mag, log_histogram, factor);
			if (cuda_error("log_histogram_kernel", true, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				log_histogram_kernel<T> <<<blocks, threads>>> (pixels_minima, num_pixels_y, min_log_mag, log_histogram_minima, factor);
				if (cuda_error("log_histogram_kernel", true, __FILE__, __LINE__)) return false;
				log_histogram_kernel<T> <<<blocks, threads>>> (pixels_saddles, num_pixels_y, min_log_mag, log_histogram_saddles, factor);
				if (cuda_error("log_histogram_kernel", true, __FILE__, __LINE__)) return false;
			}


			t_elapsed = stopwatch.stop();
			print_verbose("Done creating histograms. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 2);
		}

		/******************************************************************************
		done creating histograms of pixel values
		******************************************************************************/

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
		fname = outfile_prefix + "ipm_parameter_info.txt";
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
		outfile << "center_y1 " << center_y.re << "\n";
		outfile << "center_y2 " << center_y.im << "\n";
		outfile << "half_length_y1 " << half_length_y.re << "\n";
		outfile << "half_length_y2 " << half_length_y.im << "\n";
		outfile << "num_pixels_y1 " << num_pixels_y.re << "\n";
		outfile << "num_pixels_y2 " << num_pixels_y.im << "\n";
		outfile << "center_x1 " << center_x.re << "\n";
		outfile << "center_x2 " << center_x.im << "\n";
		outfile << "half_length_x1 " << half_length_x.re << "\n";
		outfile << "half_length_x2 " << half_length_x.im << "\n";
		outfile << "num_rays_y " << num_rays_y << "\n";
		outfile << "num_rays_x " << num_rays_x << "\n";
		outfile << "ray_half_sep_1 " << ray_half_sep.re << "\n";
		outfile << "ray_half_sep_2 " << ray_half_sep.im << "\n";
		outfile << "alpha_error " << alpha_error << "\n";
		outfile << "expansion_order " << expansion_order << "\n";
		outfile << "root_half_length " << root_half_length << "\n";
		outfile << "num_ray_threads_1 " << num_ray_threads.re << "\n";
		outfile << "num_ray_threads_2 " << num_ray_threads.im << "\n";
		outfile << "tree_levels " << tree_levels << "\n";
		outfile << "t_shoot_cells " << t_shoot_cells << "\n";
		outfile.close();
		print_verbose("Done writing parameter info to file " << fname << "\n", verbose, 1);
		print_verbose("\n", verbose * (write_stars || write_histograms || write_maps), 2);


		if (write_stars)
		{
			print_verbose("Writing star info...\n", verbose, 2);
			fname = outfile_prefix + "ipm_stars" + outfile_type;
			if (!write_star_file<T>(num_stars, rectangular, corner, theta_star, stars, fname))
			{
				std::cerr << "Error. Unable to write star info to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing star info to file " << fname << "\n", verbose, 1);
			print_verbose("\n", verbose * (write_histograms || write_maps), 2);
		}


		/******************************************************************************
		histograms of magnification maps
		******************************************************************************/
		if (write_histograms)
		{
			print_verbose("Writing magnification histograms...\n", verbose, 2);

			fname = outfile_prefix + "ipm_mags_numpixels.txt";
			if (!write_histogram<int>(histogram, histogram_length, min_mag, fname))
			{
				std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing magnification histogram to file " << fname << "\n", verbose, 1);

			fname = outfile_prefix + "ipm_log_mags_numpixels.txt";
			if (!write_histogram<int>(log_histogram, log_histogram_length, min_log_mag, fname))
			{
				std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing magnification histogram to file " << fname << "\n", verbose, 1);

			if (write_parities)
			{
				fname = outfile_prefix + "ipm_mags_numpixels_minima.txt";
				if (!write_histogram<int>(histogram_minima, histogram_length, min_mag, fname))
				{
					std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing magnification histogram to file " << fname << "\n", verbose, 1);

				fname = outfile_prefix + "ipm_log_mags_numpixels_minima.txt";
				if (!write_histogram<int>(log_histogram_minima, log_histogram_length, min_log_mag, fname))
				{
					std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing magnification histogram to file " << fname << "\n", verbose, 1);

				fname = outfile_prefix + "ipm_mags_numpixels_saddles.txt";
				if (!write_histogram<int>(histogram_saddles, histogram_length, min_mag, fname))
				{
					std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing magnification histogram to file " << fname << "\n", verbose, 1);

				fname = outfile_prefix + "ipm_log_mags_numpixels_saddles.txt";
				if (!write_histogram<int>(log_histogram_saddles, log_histogram_length, min_log_mag, fname))
				{
					std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing magnification histogram to file " << fname << "\n", verbose, 1);
			}
			print_verbose("\n", verbose * write_maps, 2);
		}


		/******************************************************************************
		write magnifications for minima, saddle, and combined maps
		******************************************************************************/
		if (write_maps)
		{
			print_verbose("Writing magnifications...\n", verbose, 2);

			fname = outfile_prefix + "ipm_magnifications" + outfile_type;
			if (!write_array<T>(pixels, num_pixels_y.im, num_pixels_y.re, fname))
			{
				std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing magnifications to file " << fname << "\n", verbose, 1);
			if (write_parities)
			{
				fname = outfile_prefix + "ipm_magnifications_minima" + outfile_type;
				if (!write_array<T>(pixels_minima, num_pixels_y.im, num_pixels_y.re, fname))
				{
					std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing magnifications to file " << fname << "\n", verbose, 1);

				fname = outfile_prefix + "ipm_magnifications_saddles" + outfile_type;
				if (!write_array<T>(pixels_saddles, num_pixels_y.im, num_pixels_y.re, fname))
				{
					std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing magnifications to file " << fname << "\n", verbose, 1);
			}
		}

		return true;
	}


public:

	bool run(int verbose)
	{
		if (!clear_memory(verbose)) return false;
		if (!set_cuda_devices(verbose)) return false;
		if (!check_input_params(verbose)) return false;
		if (!calculate_derived_params(verbose)) return false;
		if (!allocate_initialize_memory(verbose)) return false;
		if (!populate_star_array(verbose)) return false;
		if (!create_tree(verbose)) return false;
		if (!shoot_cells(verbose)) return false;
		if (!create_histograms(verbose)) return false;

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
	T* get_pixels()				{return pixels;}
	T* get_pixels_minima()		{return pixels_minima;}
	T* get_pixels_saddles()		{return pixels_saddles;}
	double get_t_shoot_cells()	{return t_shoot_cells;}

};

