#pragma once

#include "array_functions.cuh"
#include "complex.cuh"
#include "ipm_functions.cuh"
#include "microlensing.cuh"
#include "star.cuh"
#include "stopwatch.hpp"
#include "util/util.cuh"

#include <thrust/execution_policy.h> //for thrust::device
#include <thrust/extrema.h> //for thrust::min_element, thrust::max_element
#include <thrust/fill.h> //for thrust::fill
#include <thrust/functional.h> //for thrust::plus
#include <thrust/transform.h> //for thrust::transform

#include <algorithm> //for std::min and std::max
#include <cmath>
#include <fstream> //for writing files
#include <iostream>
#include <limits> //for std::numeric_limits
#include <numbers>
#include <string>


template <typename T>
class IPM : public Microlensing<T>
{

public:
	/******************************************************************************
	default input variables
	******************************************************************************/
	//must include microlensing variables to put them into scope since class is a template
	using Microlensing<T>::kappa_tot;
	using Microlensing<T>::shear;
	using Microlensing<T>::kappa_star;
	using Microlensing<T>::theta_star;
	using Microlensing<T>::mass_function_str;
	using Microlensing<T>::m_solar;
	using Microlensing<T>::m_lower;
	using Microlensing<T>::m_upper;
	using Microlensing<T>::rectangular;
	using Microlensing<T>::approx;
	using Microlensing<T>::safety_scale;
	using Microlensing<T>::starfile;
	using Microlensing<T>::random_seed;
	using Microlensing<T>::write_stars;
	using Microlensing<T>::outfile_prefix;
	
	T light_loss = static_cast<T>(0.001); //average fraction of light lost due to scatter by the microlenses in the large deflection angle limit
	Complex<T> center_y = Complex<T>();
	Complex<T> half_length_y = Complex<T>(5, 5);
	Complex<int> num_pixels_y = Complex<int>(1000, 1000);
	int num_rays_y = 1; //number density of rays per pixel in the source plane
	int write_maps = 1;
	int write_parities = 0;
	int write_histograms = 1;


private:
	//must include microlensing variables to put them into scope since class is a template
	using Microlensing<T>::num_stars;
	using Microlensing<T>::outfile_type;
	using Microlensing<T>::MAX_TAYLOR_SMOOTH;
	using Microlensing<T>::threads;
	using Microlensing<T>::blocks;
	using Microlensing<T>::mean_mass;
	using Microlensing<T>::mean_mass2;
	using Microlensing<T>::mean_mass2_ln_mass;
	using Microlensing<T>::kappa_star_actual;
	using Microlensing<T>::m_lower_actual;
	using Microlensing<T>::m_upper_actual;
	using Microlensing<T>::mean_mass_actual;
	using Microlensing<T>::mean_mass2_actual;
	using Microlensing<T>::mean_mass2_ln_mass_actual;
	using Microlensing<T>::mu_ave;
	using Microlensing<T>::corner;
	using Microlensing<T>::taylor_smooth;
	using Microlensing<T>::alpha_error;
	using Microlensing<T>::expansion_order;
	using Microlensing<T>::root_half_length;
	using Microlensing<T>::tree_levels;
	using Microlensing<T>::tree;
	using Microlensing<T>::stars;

	/******************************************************************************
	stopwatch for timing purposes
	******************************************************************************/
	Stopwatch stopwatch;
	double t_elapsed;
	//to store how long creating the magnification map took
	double t_shoot_cells;

	/******************************************************************************
	derived variables
	******************************************************************************/
	T num_rays_x; //number density of rays per unit area in the image plane
	Complex<T> ray_half_sep; //distance from center of cell to corner
	Complex<T> center_x;
	Complex<T> half_length_x;
	Complex<int> num_ray_threads; //number of threads in x1 and x2 directions

	/******************************************************************************
	dynamic memory
	******************************************************************************/
	T* pixels = nullptr;
	T* pixels_minima = nullptr;
	T* pixels_saddles = nullptr;

	int min_mag;
	int max_mag;
	int histogram_length;
	int* histogram = nullptr;
	int* histogram_minima = nullptr;
	int* histogram_saddles = nullptr;

	int min_log_mag;
	int max_log_mag;
	int log_histogram_length;
	int* log_histogram = nullptr;
	int* log_histogram_minima = nullptr;
	int* log_histogram_saddles = nullptr;



	//optional return or not, so memory can be cleared in destructor without error checking
	bool clear_memory(int verbose, bool return_on_error = true)
	{
		print_verbose("Clearing IPM<T> memory...\n", verbose, 3);
		
		/******************************************************************************
		free memory and set variables to nullptr
		******************************************************************************/

		if (!Microlensing<T>::clear_memory(verbose, return_on_error)) return false;

		cudaFree(pixels);
		if (return_on_error && cuda_error("cudaFree(*pixels)", false, __FILE__, __LINE__)) return false;
		pixels = nullptr;
		
		cudaFree(pixels_minima);
		if (return_on_error && cuda_error("cudaFree(*pixels_minima)", false, __FILE__, __LINE__)) return false;
		pixels_minima = nullptr;
		
		cudaFree(pixels_saddles);
		if (return_on_error && cuda_error("cudaFree(*pixels_saddles)", false, __FILE__, __LINE__)) return false;
		pixels_saddles = nullptr;
		
		cudaFree(histogram);
		if (return_on_error && cuda_error("cudaFree(*histogram)", false, __FILE__, __LINE__)) return false;
		histogram = nullptr;
		
		cudaFree(histogram_minima);
		if (return_on_error && cuda_error("cudaFree(*histogram_minima)", false, __FILE__, __LINE__)) return false;
		histogram_minima = nullptr;
		
		cudaFree(histogram_saddles);
		if (return_on_error && cuda_error("cudaFree(*histogram_saddles)", false, __FILE__, __LINE__)) return false;
		histogram_saddles = nullptr;
		
		cudaFree(log_histogram);
		if (return_on_error && cuda_error("cudaFree(*log_histogram)", false, __FILE__, __LINE__)) return false;
		log_histogram = nullptr;
		
		cudaFree(log_histogram_minima);
		if (return_on_error && cuda_error("cudaFree(*log_histogram_minima)", false, __FILE__, __LINE__)) return false;
		log_histogram_minima = nullptr;
		
		cudaFree(log_histogram_saddles);
		if (return_on_error && cuda_error("cudaFree(*log_histogram_saddles)", false, __FILE__, __LINE__)) return false;
		log_histogram_saddles = nullptr;

		print_verbose("Done clearing IPM<T> memory.\n", verbose, 3);
		return true;
	}

	bool check_input_params(int verbose)
	{
		print_verbose("Checking IPM<T> input parameters...\n", verbose, 3);

		if (!Microlensing<T>::check_input_params(verbose)) return false;

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

		print_verbose("Done checking IPM<T> input parameters.\n", verbose, 3);
		return true;
	}

	bool calculate_derived_params(int verbose)
	{
		print_verbose("Calculating IPM<T> derived parameters...\n", verbose, 3);
		stopwatch.start();

		if (!Microlensing<T>::calculate_derived_params(verbose)) return false;

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
		set_param("alpha_error", alpha_error, alpha_error, verbose);

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
		set_param("taylor_smooth", taylor_smooth, taylor_smooth, verbose);
		if (rectangular && taylor_smooth > MAX_TAYLOR_SMOOTH)
		{
			std::cerr << "Error. taylor_smooth must be <= " << MAX_TAYLOR_SMOOTH << "\n";
			return false;
		}

		t_elapsed = stopwatch.stop();
		print_verbose("Done calculating IPM<T> derived parameters. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 3);

		return true;
	}

	bool allocate_initialize_memory(int verbose)
	{
		print_verbose("Allocating IPM<T> memory...\n", verbose, 3);
		stopwatch.start();
		
		if (!Microlensing<T>::allocate_initialize_memory(verbose)) return false;

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
		print_verbose("Done allocating IPM<T> memory. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 3);


		/******************************************************************************
		initialize pixel values
		******************************************************************************/
		print_verbose("Initializing array values...\n", verbose, 3);
		stopwatch.start();

		thrust::fill(thrust::device, pixels, pixels + num_pixels_y.re * num_pixels_y.im, 0);
		if (write_parities)
		{
			thrust::fill(thrust::device, pixels_minima, pixels_minima + num_pixels_y.re * num_pixels_y.im, 0);
			thrust::fill(thrust::device, pixels_saddles, pixels_saddles + num_pixels_y.re * num_pixels_y.im, 0);
		}

		t_elapsed = stopwatch.stop();
		print_verbose("Done initializing array values. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 3);

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
		print_verbose("\nDone shooting cells. Elapsed time: " << t_shoot_cells << " seconds.\n", verbose, 1);


		cudaFree(percentage);
		if (cuda_error("cudaFree(*percentage)", false, __FILE__, __LINE__)) return false;
		percentage = nullptr;


		if (write_parities)
		{
			print_verbose("Adding arrays...\n", verbose, 2);
			thrust::transform(thrust::device, 
							  pixels_minima, pixels_minima + num_pixels_y.re * num_pixels_y.im, 
							  pixels_saddles, 
							  pixels, 
							  thrust::plus<T>());
			print_verbose("Done adding arrays.\n", verbose, 2);
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


			thrust::fill(thrust::device, histogram, histogram + histogram_length, 0);
			if (write_parities)
			{
				thrust::fill(thrust::device, histogram_minima, histogram_minima + histogram_length, 0);
				thrust::fill(thrust::device, histogram_saddles, histogram_saddles + histogram_length, 0);
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


			thrust::fill(thrust::device, log_histogram, log_histogram + log_histogram_length, 0);
			if (write_parities)
			{
				thrust::fill(thrust::device, log_histogram_minima, log_histogram_minima + log_histogram_length, 0);
				thrust::fill(thrust::device, log_histogram_saddles, log_histogram_saddles + log_histogram_length, 0);
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
			print_verbose("Done creating histograms. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 2);
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

	/******************************************************************************
	class initializer is empty
	******************************************************************************/
	IPM()
	{

	}

	/******************************************************************************
	class destructor clears memory with no output or error checking
	******************************************************************************/
	~IPM()
	{
		clear_memory(0, false);
	}

	/******************************************************************************
	copy constructor sets this object's dynamic memory pointers to null
	******************************************************************************/
	IPM(const IPM& other) : Microlensing<T>(other)
	{
		pixels = nullptr;
		pixels_minima = nullptr;
		pixels_saddles = nullptr;

		histogram = nullptr;
		histogram_minima = nullptr;
		histogram_saddles = nullptr;

		log_histogram = nullptr;
		log_histogram_minima = nullptr;
		log_histogram_saddles = nullptr;
	}

	/******************************************************************************
	copy assignment sets this object's dynamic memory pointers to null
	******************************************************************************/
	IPM& operator=(const IPM& other)
	{
        if (this == &other) return *this;

		Microlensing<T>::operator=(other);

		pixels = nullptr;
		pixels_minima = nullptr;
		pixels_saddles = nullptr;

		histogram = nullptr;
		histogram_minima = nullptr;
		histogram_saddles = nullptr;

		log_histogram = nullptr;
		log_histogram_minima = nullptr;
		log_histogram_saddles = nullptr;

		return *this;
	}

	bool run(int verbose)
	{
		if (!Microlensing<T>::set_cuda_devices(verbose)) return false;
		if (!clear_memory(verbose)) return false;
		if (!check_input_params(verbose)) return false;
		if (!calculate_derived_params(verbose)) return false;
		if (!allocate_initialize_memory(verbose)) return false;
		if (!Microlensing<T>::populate_star_array(verbose)) return false;
		if (!Microlensing<T>::create_tree(verbose)) return false;
		if (!shoot_cells(verbose)) return false;
		if (!create_histograms(verbose)) return false;

		return true;
	}

	bool save(int verbose)
	{
		if (!write_files(verbose)) return false;

		return true;
	}

	T* get_pixels()				{return pixels;}
	T* get_pixels_minima()		{return pixels_minima;}
	T* get_pixels_saddles()		{return pixels_saddles;}
	double get_t_shoot_cells()	{return t_shoot_cells;}

};

