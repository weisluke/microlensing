#pragma once

#include "array_functions.cuh"
#include "complex.cuh"
#include "device_array_io.cuh"
#include "device_buffer.cuh"
#include "ipm_functions_fast.cuh"
#include "ipm_leaf.cuh"
#include "microlensing.cuh"
#include "star.cuh"
#include "stopwatch.hpp"
#include "util/util.cuh"

#include <thrust/device_ptr.h> //for thrust::device_ptr
#include <thrust/execution_policy.h> //for thrust::device
#include <thrust/extrema.h> //for thrust::minmax_element
#include <thrust/fill.h> //for thrust::fill
#include <thrust/functional.h> //for thrust::plus
#include <thrust/pair.h> //for thrust::pair
#include <thrust/transform.h> //for thrust::transform

#include <algorithm> //for std::min and std::max
#include <cmath>
#include <exception> //for std::exception
#include <fstream> //for writing files
#include <iostream>
#include <limits> //for std::numeric_limits
#include <new> //for std::nothrow
#include <numbers>
#include <string>
#include <vector>


template <typename T>
class IPM : public Microlensing<T>
{

public:
	using Microlensing<T>::device;
	using Microlensing<T>::reuse_memory;
	using Microlensing<T>::use_device;
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

	/******************************************************************************
	test-only hook: when set, every triangle takes the direct deposit, as when the
	step buffers of the perimeter deposit do not fit. not reachable from the
	command line or the Python wrapper; tests/gpu/ipm_force_direct.cu sets it
	******************************************************************************/
	static inline bool testing_force_direct_deposit = false;


private:
	//must include microlensing variables to put them into scope since class is a template
	using Microlensing<T>::num_stars;
	using Microlensing<T>::outfile_type;
	using Microlensing<T>::MAX_TAYLOR_SMOOTH;
	using Microlensing<T>::threads;
	using Microlensing<T>::blocks;
	using Microlensing<T>::cuda_device_prop;
	using Microlensing<T>::stream;
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
	using Microlensing<T>::num_nodes;
	using Microlensing<T>::stars;
	using Microlensing<T>::stars_cap;

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
	precision of the pixel-allocation arithmetic: the map precision. single
	precision would suffice for the origin-shifted coordinates in the float build,
	but in the double build it changes the maps beyond their rounding floor
	******************************************************************************/
	using pixel_arith_t = T;

	/******************************************************************************
	dynamic memory
	the maps live in plain device memory (cudaMalloc) for the life of the object;
	run() does not copy them to the host. save() streams the maps to their files
	through one pair of small pinned buffers (a DeviceArrayWriter), and copy_pixels*()
	copy a map straight into a caller's buffer (the Python wrapper passes its
	numpy arrays). get_pixels*() fill a host copy (new T[]) on their first call
	and return it afterwards; run() frees the copies first (clear_memory() frees
	them too), so a copy never outlives the map it was made from.
	this replaces cudaMallocManaged, whose fault-driven migrations cost more than
	the shooting for large maps, and the pinned host copy of every map that
	replaced it in turn.
	the device buffers are grow-only (see reserve_buffer); capacities are in
	elements. a reused object (reuse_memory) keeps them between runs, while other
	objects free them at the start of every run (clear_memory) and after the
	step that used them, as before
	******************************************************************************/
	T* d_pixels = nullptr;        //device
	size_t d_pixels_cap = 0;
	T* d_pixels_minima = nullptr; //device
	size_t d_pixels_minima_cap = 0;
	T* d_pixels_saddles = nullptr; //device
	size_t d_pixels_saddles_cap = 0;
	T* h_pixels = nullptr;         //host copy for get_pixels(), made on its first call
	T* h_pixels_minima = nullptr;  //host copy for get_pixels_minima(), made on its first call
	T* h_pixels_saddles = nullptr; //host copy for get_pixels_saddles(), made on its first call

	/******************************************************************************
	whether the parity maps and the histograms are those of the last run. a
	reused object keeps their buffers after a run that made them, so a later run
	without them leaves them behind: the kernels and accessors use them only when
	the last run made them (the parity histograms only when it made both)
	******************************************************************************/
	bool have_parity_maps = false;
	bool have_histograms = false;

	/******************************************************************************
	a snapshot holds host copies of the histograms (and of the stars, see
	Microlensing<T>::host_stars) instead of device memory, and saves with save_host
	******************************************************************************/
	bool is_snapshot = false;
	std::vector<int> h_histogram, h_histogram_minima, h_histogram_saddles;
	std::vector<int> h_log_histogram, h_log_histogram_minima, h_log_histogram_saddles;

	//whether the last shoot_cells used the step buffer of the perimeter deposit
	bool perimeter_deposit_used = false;

	/******************************************************************************
	step buffer of the perimeter deposit and the clamp statistics of its row sums
	(device; allocated within shoot_cells, and freed there unless reuse_memory)
	******************************************************************************/
	T* d_steps = nullptr;
	size_t d_steps_cap = 0;
	unsigned long long* d_clamp_count = nullptr;
	size_t d_clamp_count_cap = 0;
	double* d_clamp_total = nullptr;
	size_t d_clamp_total_cap = 0;

	/******************************************************************************
	progress counter of the shooting kernel (device; allocated within
	shoot_cells, and freed there unless reuse_memory)
	******************************************************************************/
	unsigned long long* d_percentage = nullptr;
	size_t d_percentage_cap = 0;

	/******************************************************************************
	compact tree for the shooting kernel (device)
	******************************************************************************/
	ipm_leaf::LeafBuffers<T> leaf_buffers;
	int num_leaves = 0;
	int num_direct = 0;

	/******************************************************************************
	histograms (device; copied to the host by write_files)
	******************************************************************************/
	int min_mag;
	int max_mag;
	int histogram_length;
	int* histogram = nullptr;
	size_t histogram_cap = 0;
	int* histogram_minima = nullptr;
	size_t histogram_minima_cap = 0;
	int* histogram_saddles = nullptr;
	size_t histogram_saddles_cap = 0;

	int min_log_mag;
	int max_log_mag;
	int log_histogram_length;
	int* log_histogram = nullptr;
	size_t log_histogram_cap = 0;
	int* log_histogram_minima = nullptr;
	size_t log_histogram_minima_cap = 0;
	int* log_histogram_saddles = nullptr;
	size_t log_histogram_saddles_cap = 0;

	/******************************************************************************
	temporary storage of the min/max pass of the histograms of a reused object
	(device). other objects use Thrust's own allocator, as before
	******************************************************************************/
	ThrustScratch scratch;



	/******************************************************************************
	the parity maps of the last run, nullptr if it made none
	******************************************************************************/
	T* pixels_minima() const		{return have_parity_maps ? d_pixels_minima : nullptr;}
	T* pixels_saddles() const		{return have_parity_maps ? d_pixels_saddles : nullptr;}

	bool free_step_buffers(bool return_on_error = true)
	{
		if (!release_buffer(d_steps, d_steps_cap, return_on_error, "cudaFree(*d_steps)")) return false;
		if (!release_buffer(d_clamp_count, d_clamp_count_cap, return_on_error, "cudaFree(*d_clamp_count)")) return false;
		if (!release_buffer(d_clamp_total, d_clamp_total_cap, return_on_error, "cudaFree(*d_clamp_total)")) return false;
		return true;
	}

	void free_host_copies()
	{
		delete[] h_pixels;
		h_pixels = nullptr;
		delete[] h_pixels_minima;
		h_pixels_minima = nullptr;
		delete[] h_pixels_saddles;
		h_pixels_saddles = nullptr;
	}

	//optional return or not, so memory can be cleared in destructor without error checking
	bool clear_memory(int verbose, bool return_on_error = true)
	{
		print_verbose("Clearing IPM<T> memory...\n", verbose, 3);
		
		/******************************************************************************
		free memory and set variables to nullptr and capacities to 0
		the host copies go first: freeing them cannot fail, so even a run() whose
		clear_memory() fails leaves no stale copy for get_pixels*() to return
		******************************************************************************/

		free_host_copies();
		have_parity_maps = false;
		have_histograms = false;

		if (!Microlensing<T>::use_device_to_free(d_pixels || d_pixels_minima || d_pixels_saddles || d_steps || d_clamp_count || d_clamp_total
			|| d_percentage || leaf_buffers.leaves || leaf_buffers.coef_re || leaf_buffers.coef_im || leaf_buffers.direct || leaf_buffers.counters
			|| histogram || histogram_minima || histogram_saddles || log_histogram || log_histogram_minima || log_histogram_saddles
			|| scratch.ptr || !scratch.extra.empty(), return_on_error)) return false;

		if (!Microlensing<T>::clear_memory(verbose, return_on_error)) return false;

		if (!release_buffer(d_pixels, d_pixels_cap, return_on_error, "cudaFree(*d_pixels)")) return false;
		if (!release_buffer(d_pixels_minima, d_pixels_minima_cap, return_on_error, "cudaFree(*d_pixels_minima)")) return false;
		if (!release_buffer(d_pixels_saddles, d_pixels_saddles_cap, return_on_error, "cudaFree(*d_pixels_saddles)")) return false;

		if (!free_step_buffers(return_on_error)) return false;
		if (!release_buffer(d_percentage, d_percentage_cap, return_on_error, "cudaFree(*percentage)")) return false;

		if (!leaf_buffers.release(return_on_error)) return false;

		if (!release_buffer(histogram, histogram_cap, return_on_error, "cudaFree(*histogram)")) return false;
		if (!release_buffer(histogram_minima, histogram_minima_cap, return_on_error, "cudaFree(*histogram_minima)")) return false;
		if (!release_buffer(histogram_saddles, histogram_saddles_cap, return_on_error, "cudaFree(*histogram_saddles)")) return false;
		if (!release_buffer(log_histogram, log_histogram_cap, return_on_error, "cudaFree(*log_histogram)")) return false;
		if (!release_buffer(log_histogram_minima, log_histogram_minima_cap, return_on_error, "cudaFree(*log_histogram_minima)")) return false;
		if (!release_buffer(log_histogram_saddles, log_histogram_saddles_cap, return_on_error, "cudaFree(*log_histogram_saddles)")) return false;

		if (!scratch.release() && return_on_error) return false;

		print_verbose("Done clearing IPM<T> memory.\n", verbose, 3);
		return true;
	}

	/******************************************************************************
	start a run of a reused object (reuse_memory): the device buffers stay for the
	new map, so like the stream of create_stream they tie the object to its
	device. the host copies of the previous maps are freed. read_star_file fills
	an existing star array without checking its size and allocates one of exactly
	the file's size when stars is nullptr, so with a star file the stars are
	released first
	******************************************************************************/
	bool start_reused_run(int verbose)
	{
		print_verbose("Keeping IPM<T> memory for this run...\n", verbose, 3);
		free_host_copies();
		have_parity_maps = false;
		have_histograms = false;
		if (starfile != "" && !release_buffer(stars, stars_cap, true, "cudaFree(*stars)")) return false;
		return true;
	}

	/******************************************************************************
	copy the input parameters of another object (see Microlensing<T>::copy_inputs_from)
	******************************************************************************/
	void copy_inputs_from(const IPM<T>& o)
	{
		Microlensing<T>::copy_inputs_from(o);
		light_loss = o.light_loss;
		center_y = o.center_y;
		half_length_y = o.half_length_y;
		num_pixels_y = o.num_pixels_y;
		num_rays_y = o.num_rays_y;
		write_maps = o.write_maps;
		write_parities = o.write_parities;
		write_histograms = o.write_histograms;
	}

	/******************************************************************************
	copy the values derived from the input parameters of another object's run, and
	whether its parity maps and histograms are those of the run
	******************************************************************************/
	void copy_derived_from(const IPM<T>& o)
	{
		Microlensing<T>::copy_derived_from(o);
		t_shoot_cells = o.t_shoot_cells;
		num_rays_x = o.num_rays_x;
		ray_half_sep = o.ray_half_sep;
		center_x = o.center_x;
		half_length_x = o.half_length_x;
		num_ray_threads = o.num_ray_threads;
		num_leaves = o.num_leaves;
		num_direct = o.num_direct;
		min_mag = o.min_mag;
		max_mag = o.max_mag;
		histogram_length = o.histogram_length;
		min_log_mag = o.min_log_mag;
		max_log_mag = o.max_log_mag;
		log_histogram_length = o.log_histogram_length;
		perimeter_deposit_used = o.perimeter_deposit_used;
		have_parity_maps = o.have_parity_maps;
		have_histograms = o.have_histograms;
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
			1.0 * num_rays_y * (1ULL * num_pixels_y.re * num_pixels_y.im) / (2 * half_length_y.re * 2 * half_length_y.im),
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
		if (rectangular && approx && taylor_smooth > MAX_TAYLOR_SMOOTH)
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
		allocate memory for pixels (device only, see dynamic memory above)
		******************************************************************************/
		const size_t npix = 1ULL * num_pixels_y.re * num_pixels_y.im;
		const size_t pixel_bytes = npix * sizeof(T);
		if (!reserve_buffer(d_pixels, d_pixels_cap, npix, BufferKind::device, 1.0, "cudaMalloc(*d_pixels)")) return false;
		if (write_parities)
		{
			if (!reserve_buffer(d_pixels_minima, d_pixels_minima_cap, npix, BufferKind::device, 1.0, "cudaMalloc(*d_pixels_minima)")) return false;
			if (!reserve_buffer(d_pixels_saddles, d_pixels_saddles_cap, npix, BufferKind::device, 1.0, "cudaMalloc(*d_pixels_saddles)")) return false;
		}
		have_parity_maps = write_parities;

		t_elapsed = stopwatch.stop();
		print_verbose("Done allocating IPM<T> memory. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 3);


		/******************************************************************************
		initialize pixel values
		******************************************************************************/
		print_verbose("Initializing array values...\n", verbose, 3);
		stopwatch.start();

		cudaMemsetAsync(d_pixels, 0, pixel_bytes, stream);
		if (cuda_error("cudaMemsetAsync(*d_pixels)", false, __FILE__, __LINE__)) return false;
		if (write_parities)
		{
			cudaMemsetAsync(d_pixels_minima, 0, pixel_bytes, stream);
			if (cuda_error("cudaMemsetAsync(*d_pixels_minima)", false, __FILE__, __LINE__)) return false;
			cudaMemsetAsync(d_pixels_saddles, 0, pixel_bytes, stream);
			if (cuda_error("cudaMemsetAsync(*d_pixels_saddles)", false, __FILE__, __LINE__)) return false;
		}

		t_elapsed = stopwatch.stop();
		print_verbose("Done initializing array values. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 3);

		return true;
	}

	bool build_compact_tree(int verbose)
	{
		print_verbose("Building compact leaf records...\n", verbose, 2);
		stopwatch.start();
		//the sizes depend on the star field: a reused object leaves room for more
		const double headroom = reuse_memory ? 1.25 : 1.0;
		if (!ipm_leaf::build_leaf_records<T>(tree, num_nodes, stars, expansion_order,
			leaf_buffers, num_leaves, num_direct, headroom, stream, verbose)) return false;
		//a reused object keeps the counters for its next run
		if (!reuse_memory && !release_buffer(leaf_buffers.counters, leaf_buffers.counters_cap, true, "cudaFree(*counters)")) return false;
		t_elapsed = stopwatch.stop();
		print_verbose("Done building compact leaf records. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 2);
		return true;
	}

	bool shoot_cells(int verbose)
	{
		perimeter_deposit_used = false;
		/******************************************************************************
		one block per TILE x TILE tile of cells; launch as many blocks as the device
		can keep resident (occupancy API), not one per SM as get_blocks() does, so
		the latency of the tree walk, star loads and atomics can be hidden
		******************************************************************************/
		const int tiles_x = (num_ray_threads.re + ipm_fast::TILE - 1) / ipm_fast::TILE;
		const int tiles_y = (num_ray_threads.im + ipm_fast::TILE - 1) / ipm_fast::TILE;
		const long long num_tiles = 1LL * tiles_x * tiles_y;

		int blocks_per_sm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, ipm_fast::shoot_cells_tiled_kernel<T, pixel_arith_t>, ipm_fast::TILE * ipm_fast::TILE, 0);
		if (cuda_error("cudaOccupancyMaxActiveBlocksPerMultiprocessor", false, __FILE__, __LINE__)) return false;
		if (blocks_per_sm < 1) blocks_per_sm = 1;
		long long grid = 1LL * blocks_per_sm * cuda_device_prop.multiProcessorCount;
		if (grid > num_tiles) grid = num_tiles;
		print_verbose("Shooting " << num_tiles << " tiles of " << ipm_fast::TILE << "x" << ipm_fast::TILE << " cells with " << grid
			<< " blocks (" << blocks_per_sm << " resident blocks per SM)\n", verbose, 2);

		const unsigned long long int percentage_start = 1;
		if (!reserve_buffer(d_percentage, d_percentage_cap, 1, BufferKind::device, 1.0, "cudaMalloc(*percentage)")) return false;
		cudaMemcpyAsync(d_percentage, &percentage_start, sizeof(unsigned long long int), cudaMemcpyHostToDevice, stream);
		if (cuda_error("cudaMemcpyAsync(*percentage)", false, __FILE__, __LINE__)) return false;
		//the source is a local that goes out of scope
		cudaStreamSynchronize(stream);
		if (cuda_error("cudaStreamSynchronize", false, __FILE__, __LINE__)) return false;

		ipm_fast::SmoothConst<T> smooth_const = ipm_fast::make_smooth_const<T>(corner);

		/******************************************************************************
		step buffers of the perimeter deposit (see tri_cov::allocate_triangle):
		triangles whose clamped bounding box has more than EDGE_MIN_BBOX pixels
		deposit values along their edges into the map and steps into a step buffer
		of the map's size; after the kernel the row-wise prefix sum of the steps is
		added to the map. the steps follow the triangle's parity. with parities the
		total map d_pixels is not written before minima + saddles are summed below,
		and that sum overwrites it, so d_pixels holds the minima's steps and one
		extra buffer suffices with or without parities.

		if the buffer does not fit into free device memory with a margin, every
		triangle takes the direct deposit. both deposits are exact, so the map is
		the same up to rounding either way. a reused object keeps its step buffer,
		so the check is needed only when the buffer has to grow
		******************************************************************************/
		constexpr size_t MiB = 1ULL << 20;
		const size_t npix = 1ULL * num_pixels_y.re * num_pixels_y.im;
		const size_t pixel_bytes = npix * sizeof(T);
		const int num_step_buffers = 1; //with parities, d_pixels is the second
		long long edge_min_bbox = LLONG_MAX;
		T* steps_min = nullptr;
		T* steps_sad = nullptr;
		T* steps = nullptr;
		bool perimeter = false; //whether the step buffer is used
		if (testing_force_direct_deposit)
		{
			print_verbose("Perimeter deposit disabled for testing.\n", verbose, 2);
		}
		else
		{
			bool fits = true;
			size_t free_bytes = 0;
			size_t total_bytes = 0;
			const size_t need = num_step_buffers * pixel_bytes + 64 * MiB;
			if (d_steps == nullptr || d_steps_cap < npix)
			{
				cudaMemGetInfo(&free_bytes, &total_bytes);
				if (cuda_error("cudaMemGetInfo", false, __FILE__, __LINE__)) return false;
				//growing frees the smaller buffer first
				fits = need <= free_bytes + d_steps_cap * sizeof(T);
				if (fits)
				{
					//another process can take the memory after the check: a failed allocation also falls back
					if (!reserve_buffer(d_steps, d_steps_cap, npix, BufferKind::device, 1.0, "cudaMalloc(*d_steps)"))
					{
						if (last_cuda_error_code != cudaErrorMemoryAllocation) return false;
						last_cuda_error_code = 0; //the fallback is not an error of the run
						fits = false;
					}
				}
			}
			if (!fits)
			{
				print_verbose("Perimeter deposit disabled: step buffers need " << (need + MiB - 1) / MiB << " MiB, "
					<< free_bytes / MiB << " MiB free.\n", verbose, 2);
			}
			else
			{
				if (!reserve_buffer(d_clamp_count, d_clamp_count_cap, 1, BufferKind::device, 1.0, "cudaMalloc(*d_clamp_count)")) return false;
				if (!reserve_buffer(d_clamp_total, d_clamp_total_cap, 1, BufferKind::device, 1.0, "cudaMalloc(*d_clamp_total)")) return false;
				cudaMemsetAsync(d_clamp_count, 0, sizeof(unsigned long long), stream);
				if (cuda_error("cudaMemsetAsync(*d_clamp_count)", false, __FILE__, __LINE__)) return false;
				cudaMemsetAsync(d_clamp_total, 0, sizeof(double), stream);
				if (cuda_error("cudaMemsetAsync(*d_clamp_total)", false, __FILE__, __LINE__)) return false;
				cudaMemsetAsync(d_steps, 0, pixel_bytes, stream);
				if (cuda_error("cudaMemsetAsync(*d_steps)", false, __FILE__, __LINE__)) return false;
				if (write_parities)
				{
					cudaMemsetAsync(d_pixels, 0, pixel_bytes, stream);
					if (cuda_error("cudaMemsetAsync(*d_pixels)", false, __FILE__, __LINE__)) return false;
					steps_min = d_pixels;
					steps_sad = d_steps;
				}
				else
				{
					steps = d_steps;
				}
				//the memsets must not run into the timed region
				if (cuda_error("cudaStreamSynchronize", true, __FILE__, __LINE__, stream)) return false;
				edge_min_bbox = ipm_fast::EDGE_MIN_BBOX;
				perimeter = true;
				print_verbose("Perimeter deposit for bounding boxes above " << ipm_fast::EDGE_MIN_BBOX << " pixels.\n", verbose, 2);
			}
		}

		perimeter_deposit_used = perimeter;

		/******************************************************************************
		shoot rays and calculate time taken in seconds (including the row sums of
		the perimeter deposit)
		******************************************************************************/
		print_verbose("Shooting cells...\n", verbose, 1);
		stopwatch.start();
		ipm_fast::shoot_cells_tiled_kernel<T, pixel_arith_t> <<<static_cast<unsigned int>(grid), dim3(ipm_fast::TILE, ipm_fast::TILE), 0, stream>>> (
			kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
			leaf_buffers.leaves, leaf_buffers.coef_re, leaf_buffers.coef_im, expansion_order + 1, leaf_buffers.direct,
			rectangular, corner, approx, taylor_smooth, smooth_const,
			ray_half_sep, num_ray_threads, center_x, half_length_x,
			center_y, half_length_y, pixels_minima(), pixels_saddles(), d_pixels, num_pixels_y,
			steps_min, steps_sad, steps, edge_min_bbox, d_percentage, verbose);
		if (cuda_error("shoot_cells_tiled_kernel", true, __FILE__, __LINE__, stream)) return false;
		if (perimeter)
		{
			const unsigned int row_blocks = static_cast<unsigned int>(
				(1LL * num_pixels_y.im * 32 + ipm_fast::ROW_PREFIX_BLOCK - 1) / ipm_fast::ROW_PREFIX_BLOCK); //one warp per row
			if (write_parities)
			{
				ipm_fast::add_row_prefix_kernel<T> <<<row_blocks, ipm_fast::ROW_PREFIX_BLOCK, 0, stream>>> (d_pixels_minima, steps_min, num_pixels_y, d_clamp_count, d_clamp_total);
				if (cuda_error("add_row_prefix_kernel", false, __FILE__, __LINE__)) return false;
				ipm_fast::add_row_prefix_kernel<T> <<<row_blocks, ipm_fast::ROW_PREFIX_BLOCK, 0, stream>>> (d_pixels_saddles, steps_sad, num_pixels_y, d_clamp_count, d_clamp_total);
			}
			else
			{
				ipm_fast::add_row_prefix_kernel<T> <<<row_blocks, ipm_fast::ROW_PREFIX_BLOCK, 0, stream>>> (d_pixels, steps, num_pixels_y, d_clamp_count, d_clamp_total);
			}
			if (cuda_error("add_row_prefix_kernel", true, __FILE__, __LINE__, stream)) return false;
		}
		t_shoot_cells = stopwatch.stop();
		print_verbose("\nDone shooting cells. Elapsed time: " << t_shoot_cells << " seconds.\n", verbose, 1);

		if (perimeter)
		{
			unsigned long long n_clamped = 0;
			double clamped_total = 0;
			cudaMemcpyAsync(&n_clamped, d_clamp_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
			if (cuda_error("cudaMemcpyAsync(*d_clamp_count)", false, __FILE__, __LINE__)) return false;
			cudaMemcpyAsync(&clamped_total, d_clamp_total, sizeof(double), cudaMemcpyDeviceToHost, stream);
			if (cuda_error("cudaMemcpyAsync(*d_clamp_total)", false, __FILE__, __LINE__)) return false;
			cudaStreamSynchronize(stream);
			if (cuda_error("cudaStreamSynchronize", false, __FILE__, __LINE__)) return false;
			//a reused object keeps the step buffers for its next run
			if (!reuse_memory && !free_step_buffers()) return false;
			print_verbose("Row sums clamped " << n_clamped << " pixels, total " << clamped_total << ".\n", verbose, 2);
		}

		//a reused object keeps the progress counter for its next run
		if (!reuse_memory && !release_buffer(d_percentage, d_percentage_cap, true, "cudaFree(*percentage)")) return false;

		if (write_parities)
		{
			print_verbose("Adding arrays...\n", verbose, 2);
			thrust::transform(thrust::cuda::par_nosync.on(stream),
							  thrust::device_pointer_cast(d_pixels_minima), thrust::device_pointer_cast(d_pixels_minima) + (1ULL * num_pixels_y.re * num_pixels_y.im),
							  thrust::device_pointer_cast(d_pixels_saddles),
							  thrust::device_pointer_cast(d_pixels),
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

		have_histograms = false;
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
			one fused min/max pass per array (device memory, so go through
			thrust::device_ptr; the two values are copied to the host on the stream).
			a reused object takes the temporary storage of minmax_element from
			scratch, reset before each pass, as Thrust's own allocator would
			cudaMalloc and cudaFree it in every pass
			******************************************************************************/
			const size_t npix_total = 1ULL * num_pixels_y.re * num_pixels_y.im;
			auto minmax = [&](T* d_arr, T& lo, T& hi)
			{
				thrust::device_ptr<T> dp = thrust::device_pointer_cast(d_arr);
				thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T>> mm;
				try
				{
					if (reuse_memory)
					{
						scratch.reset();
						mm = thrust::minmax_element(thrust::cuda::par_nosync(scratch).on(stream), dp, dp + npix_total);
					}
					else
					{
						mm = thrust::minmax_element(thrust::cuda::par_nosync.on(stream), dp, dp + npix_total);
					}
				}
				catch (const std::exception& e)
				{
					std::cerr << "Error. thrust::minmax_element failed: " << e.what() << "\n";
					return false;
				}
				if (cuda_error("thrust::minmax_element", false, __FILE__, __LINE__)) return false;
				cudaMemcpyAsync(&lo, d_arr + (mm.first - dp), sizeof(T), cudaMemcpyDeviceToHost, stream);
				if (cuda_error("cudaMemcpyAsync(*lo)", false, __FILE__, __LINE__)) return false;
				cudaMemcpyAsync(&hi, d_arr + (mm.second - dp), sizeof(T), cudaMemcpyDeviceToHost, stream);
				if (cuda_error("cudaMemcpyAsync(*hi)", false, __FILE__, __LINE__)) return false;
				cudaStreamSynchronize(stream);
				return !cuda_error("cudaStreamSynchronize", false, __FILE__, __LINE__);
			};
			T lo_all, hi_all;
			if (!minmax(d_pixels, lo_all, hi_all)) return false;
			min_mag = std::round(lo_all * factor);
			max_mag = std::round(hi_all * factor);
			min_log_mag = std::round(std::log10(lo_all) * factor);
			max_log_mag = std::round(std::log10(hi_all) * factor);

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
				T lo_min, hi_min, lo_sad, hi_sad;
				if (!minmax(d_pixels_minima, lo_min, hi_min)) return false;
				if (!minmax(d_pixels_saddles, lo_sad, hi_sad)) return false;
				int min_mag_minima = std::round(lo_min * factor);
				int max_mag_minima = std::round(hi_min * factor);

				mu_min_actual = 1.0 * min_mag_minima / factor;

				if (mu_ave > 1 && mu_min_actual < mu_min_theory)
				{
					std::cerr << "Warning. Minimum positive parity magnification after shooting cells is less than the theoretical minimum.\n";
					std::cerr << "   mu_min_actual = " << mu_min_actual << "\n";
					std::cerr << "   mu_min_theory = 1 / (1 - (kappa_tot - kappa_star))^2\n";
					std::cerr << "                 = 1 / (1 - (" << kappa_tot << " - " << kappa_star << "))^2 = " << mu_min_theory << "\n";
					print_verbose("\n", verbose * (verbose < 2), 1);
				}

				int min_mag_saddles = std::round(lo_sad * factor);
				int max_mag_saddles = std::round(hi_sad * factor);

				min_mag = std::min({min_mag, min_mag_minima, min_mag_saddles});
				max_mag = std::max({max_mag, max_mag_minima, max_mag_saddles});

				int min_log_mag_minima = std::round(std::log10(lo_min) * factor);
				int max_log_mag_minima = std::round(std::log10(hi_min) * factor);
				int min_log_mag_saddles = std::round(std::log10(lo_sad) * factor);
				int max_log_mag_saddles = std::round(std::log10(hi_sad) * factor);

				min_log_mag = std::min({min_log_mag, min_log_mag_minima, min_log_mag_saddles});
				max_log_mag = std::max({max_log_mag, max_log_mag_minima, max_log_mag_saddles});
			}

			histogram_length = max_mag - min_mag + 1;

			//the lengths depend on the map: a reused object leaves room for more
			const double headroom = reuse_memory ? 1.25 : 1.0;
			if (!reserve_buffer(histogram, histogram_cap, histogram_length, BufferKind::device, headroom, "cudaMalloc(*histogram)")) return false;
			if (write_parities)
			{
				if (!reserve_buffer(histogram_minima, histogram_minima_cap, histogram_length, BufferKind::device, headroom, "cudaMalloc(*histogram_minima)")) return false;
				if (!reserve_buffer(histogram_saddles, histogram_saddles_cap, histogram_length, BufferKind::device, headroom, "cudaMalloc(*histogram_saddles)")) return false;
			}


			cudaMemsetAsync(histogram, 0, histogram_length * sizeof(int), stream);
			if (cuda_error("cudaMemsetAsync(*histogram)", false, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				cudaMemsetAsync(histogram_minima, 0, histogram_length * sizeof(int), stream);
				if (cuda_error("cudaMemsetAsync(*histogram_minima)", false, __FILE__, __LINE__)) return false;
				cudaMemsetAsync(histogram_saddles, 0, histogram_length * sizeof(int), stream);
				if (cuda_error("cudaMemsetAsync(*histogram_saddles)", false, __FILE__, __LINE__)) return false;
			}


			threads = get_threads(histogram_kernel<T>, num_pixels_y.re, num_pixels_y.im);
			blocks = get_blocks(threads, num_pixels_y.re, num_pixels_y.im);
			histogram_kernel<T> <<<blocks, threads, 0, stream>>> (d_pixels, num_pixels_y, min_mag, histogram, factor);
			if (cuda_error("histogram_kernel", true, __FILE__, __LINE__, stream)) return false;
			if (write_parities)
			{
				histogram_kernel<T> <<<blocks, threads, 0, stream>>> (d_pixels_minima, num_pixels_y, min_mag, histogram_minima, factor);
				if (cuda_error("histogram_kernel", true, __FILE__, __LINE__, stream)) return false;
				histogram_kernel<T> <<<blocks, threads, 0, stream>>> (d_pixels_saddles, num_pixels_y, min_mag, histogram_saddles, factor);
				if (cuda_error("histogram_kernel", true, __FILE__, __LINE__, stream)) return false;
			}
			

			/******************************************************************************
			histogram of log_10(mu) (ranges computed above)
			******************************************************************************/

			log_histogram_length = max_log_mag - min_log_mag + 1;

			if (!reserve_buffer(log_histogram, log_histogram_cap, log_histogram_length, BufferKind::device, headroom, "cudaMalloc(*log_histogram)")) return false;
			if (write_parities)
			{
				if (!reserve_buffer(log_histogram_minima, log_histogram_minima_cap, log_histogram_length, BufferKind::device, headroom, "cudaMalloc(*log_histogram_minima)")) return false;
				if (!reserve_buffer(log_histogram_saddles, log_histogram_saddles_cap, log_histogram_length, BufferKind::device, headroom, "cudaMalloc(*log_histogram_saddles)")) return false;
			}


			cudaMemsetAsync(log_histogram, 0, log_histogram_length * sizeof(int), stream);
			if (cuda_error("cudaMemsetAsync(*log_histogram)", false, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				cudaMemsetAsync(log_histogram_minima, 0, log_histogram_length * sizeof(int), stream);
				if (cuda_error("cudaMemsetAsync(*log_histogram_minima)", false, __FILE__, __LINE__)) return false;
				cudaMemsetAsync(log_histogram_saddles, 0, log_histogram_length * sizeof(int), stream);
				if (cuda_error("cudaMemsetAsync(*log_histogram_saddles)", false, __FILE__, __LINE__)) return false;
			}


			threads = get_threads(log_histogram_kernel<T>, num_pixels_y.re, num_pixels_y.im);
			blocks = get_blocks(threads, num_pixels_y.re, num_pixels_y.im);
			log_histogram_kernel<T> <<<blocks, threads, 0, stream>>> (d_pixels, num_pixels_y, min_log_mag, log_histogram, factor);
			if (cuda_error("log_histogram_kernel", true, __FILE__, __LINE__, stream)) return false;
			if (write_parities)
			{
				log_histogram_kernel<T> <<<blocks, threads, 0, stream>>> (d_pixels_minima, num_pixels_y, min_log_mag, log_histogram_minima, factor);
				if (cuda_error("log_histogram_kernel", true, __FILE__, __LINE__, stream)) return false;
				log_histogram_kernel<T> <<<blocks, threads, 0, stream>>> (d_pixels_saddles, num_pixels_y, min_log_mag, log_histogram_saddles, factor);
				if (cuda_error("log_histogram_kernel", true, __FILE__, __LINE__, stream)) return false;
			}


			have_histograms = true;

			t_elapsed = stopwatch.stop();
			print_verbose("Done creating histograms. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 2);
		}

		/******************************************************************************
		done creating histograms of pixel values
		******************************************************************************/

		return true;
	}

	/******************************************************************************
	write the output files. With host_maps (a snapshot: the total, minima and
	saddle maps in host memory), the histograms come from the host copies and the
	maps are written from host memory, with no CUDA call. Otherwise they are
	copied from the device
	******************************************************************************/
	bool write_files(int verbose, const T* const* host_maps = nullptr)
	{
		/******************************************************************************
		stream for writing output files
		set precision to 9 digits
		******************************************************************************/
		std::ofstream outfile;
		outfile.precision(9);
		std::string fname;

		if (host_maps == nullptr && !use_device()) return false;

		if (!Microlensing<T>::write_files(verbose, "ipm")) return false;

		print_verbose("Writing IPM<T> parameter info...\n", verbose, 2);
		fname = outfile_prefix + "ipm_parameter_info.txt";
		outfile.open(fname, std::ios_base::app);
		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		outfile << "light_loss " << light_loss << "\n";
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
		outfile << "num_ray_threads_1 " << num_ray_threads.re << "\n";
		outfile << "num_ray_threads_2 " << num_ray_threads.im << "\n";
		outfile << "t_shoot_cells " << t_shoot_cells << "\n";
		outfile.close();
		print_verbose("Done writing IPM<T> parameter info to file " << fname << "\n", verbose, 1);


		/******************************************************************************
		histograms of magnification maps
		******************************************************************************/
		if (write_histograms)
		{
			print_verbose("Writing magnification histograms...\n", verbose, 2);

			//the histograms are in device memory: each one is copied into h_hist before it is written.
			//a failed allocation returns false: it must not throw through the C interface (save)
			//a snapshot has host copies of them (h_vec) and copies nothing
			std::vector<int> h_hist;
			auto copy_histogram = [&](const int* d_hist, std::vector<int>& h_vec, int n)
			{
				if (host_maps != nullptr)
				{
					if (!have_histograms || n < 1 || h_vec.size() != static_cast<size_t>(n)) return false;
					h_hist = h_vec;
					return true;
				}
				if (!have_histograms || d_hist == nullptr || n < 1) return false;
				try
				{
					h_hist.resize(n);
				}
				catch (const std::bad_alloc&)
				{
					std::cerr << "Error. Unable to allocate a host copy of a histogram.\n";
					return false;
				}
				cudaMemcpyAsync(h_hist.data(), d_hist, n * sizeof(int), cudaMemcpyDeviceToHost, stream);
				if (cuda_error("cudaMemcpyAsync(*h_hist)", false, __FILE__, __LINE__)) return false;
				cudaStreamSynchronize(stream);
				return !cuda_error("cudaStreamSynchronize", false, __FILE__, __LINE__);
			};

			fname = outfile_prefix + "ipm_mags_numpixels.txt";
			if (!copy_histogram(histogram, h_histogram, histogram_length) || !write_histogram<int>(h_hist.data(), histogram_length, min_mag, fname))
			{
				std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing magnification histogram to file " << fname << "\n", verbose, 1);

			fname = outfile_prefix + "ipm_log_mags_numpixels.txt";
			if (!copy_histogram(log_histogram, h_log_histogram, log_histogram_length) || !write_histogram<int>(h_hist.data(), log_histogram_length, min_log_mag, fname))
			{
				std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing magnification histogram to file " << fname << "\n", verbose, 1);

			if (write_parities)
			{
				fname = outfile_prefix + "ipm_mags_numpixels_minima.txt";
				if (!copy_histogram(have_parity_maps ? histogram_minima : nullptr, h_histogram_minima, histogram_length) || !write_histogram<int>(h_hist.data(), histogram_length, min_mag, fname))
				{
					std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing magnification histogram to file " << fname << "\n", verbose, 1);

				fname = outfile_prefix + "ipm_log_mags_numpixels_minima.txt";
				if (!copy_histogram(have_parity_maps ? log_histogram_minima : nullptr, h_log_histogram_minima, log_histogram_length) || !write_histogram<int>(h_hist.data(), log_histogram_length, min_log_mag, fname))
				{
					std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing magnification histogram to file " << fname << "\n", verbose, 1);

				fname = outfile_prefix + "ipm_mags_numpixels_saddles.txt";
				if (!copy_histogram(have_parity_maps ? histogram_saddles : nullptr, h_histogram_saddles, histogram_length) || !write_histogram<int>(h_hist.data(), histogram_length, min_mag, fname))
				{
					std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing magnification histogram to file " << fname << "\n", verbose, 1);

				fname = outfile_prefix + "ipm_log_mags_numpixels_saddles.txt";
				if (!copy_histogram(have_parity_maps ? log_histogram_saddles : nullptr, h_log_histogram_saddles, log_histogram_length) || !write_histogram<int>(h_hist.data(), log_histogram_length, min_log_mag, fname))
				{
					std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing magnification histogram to file " << fname << "\n", verbose, 1);
			}
		}


		/******************************************************************************
		write magnifications for minima, saddle, and combined maps, streamed from
		device memory through one writer, so that they share its pinned buffers.
		a snapshot writes them from the host maps
		******************************************************************************/
		if (write_maps)
		{
			print_verbose("Writing magnifications...\n", verbose, 2);
			//a snapshot makes no CUDA call, not even the one of the writer's release
			std::unique_ptr<DeviceArrayWriter<T>> writer;
			if (host_maps == nullptr) writer.reset(new DeviceArrayWriter<T>());
			//the maps of this run: the device maps, or the host maps of a snapshot
			auto write_map = [&](const T* device_map, int which, const std::string& file)
			{
				if (host_maps != nullptr) return host_maps[which] != nullptr && write_array<T>(host_maps[which], num_pixels_y.im, num_pixels_y.re, file);
				return writer->write(device_map, num_pixels_y.im, num_pixels_y.re, file);
			};

			fname = outfile_prefix + "ipm_magnifications" + outfile_type;
			if (!write_map(d_pixels, 0, fname))
			{
				std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing magnifications to file " << fname << "\n", verbose, 1);
			if (write_parities)
			{
				fname = outfile_prefix + "ipm_magnifications_minima" + outfile_type;
				if (!write_map(pixels_minima(), 1, fname))
				{
					std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing magnifications to file " << fname << "\n", verbose, 1);

				fname = outfile_prefix + "ipm_magnifications_saddles" + outfile_type;
				if (!write_map(pixels_saddles(), 2, fname))
				{
					std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing magnifications to file " << fname << "\n", verbose, 1);
			}
			if (writer && !writer->release())
			{
				std::cerr << "Error. CUDA error while releasing the buffers used to write the magnifications.\n";
				return false;
			}
		}

		return true;
	}

	bool copy_map(const T* d_map, T* dst, const char* name) const
	{
		if (d_map == nullptr || dst == nullptr) return false;
		if (!use_device()) return false;
		cudaMemcpyAsync(dst, d_map, (1ULL * num_pixels_y.re * num_pixels_y.im) * sizeof(T), cudaMemcpyDeviceToHost, stream);
		if (cuda_error(name, false, __FILE__, __LINE__)) return false;
		cudaStreamSynchronize(stream);
		return !cuda_error("cudaStreamSynchronize", false, __FILE__, __LINE__);
	}

	T* host_copy(const T* d_map, T*& h_map, const char* name)
	{
		if (h_map == nullptr && d_map != nullptr)
		{
			if (!use_device()) return nullptr;
			T* copy = new (std::nothrow) T[1ULL * num_pixels_y.re * num_pixels_y.im];
			if (copy == nullptr)
			{
				std::cerr << "Error. Unable to allocate a host copy of a magnification map.\n";
				return nullptr;
			}
			if (!copy_map(d_map, copy, name))
			{
				delete[] copy;
				return nullptr;
			}
			h_map = copy;
		}
		return h_map;
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
	copy constructor sets this object's dynamic memory pointers to null and their
	capacities to 0 (the base makes it a default, not reused, object)
	******************************************************************************/
	IPM(const IPM& other) : Microlensing<T>(other)
	{
		d_pixels = nullptr;
		d_pixels_cap = 0;
		d_pixels_minima = nullptr;
		d_pixels_minima_cap = 0;
		d_pixels_saddles = nullptr;
		d_pixels_saddles_cap = 0;
		h_pixels = nullptr;
		h_pixels_minima = nullptr;
		h_pixels_saddles = nullptr;
		have_parity_maps = false;
		have_histograms = false;
		d_steps = nullptr;
		d_steps_cap = 0;
		d_clamp_count = nullptr;
		d_clamp_count_cap = 0;
		d_clamp_total = nullptr;
		d_clamp_total_cap = 0;
		d_percentage = nullptr;
		d_percentage_cap = 0;
		leaf_buffers = ipm_leaf::LeafBuffers<T>();

		histogram = nullptr;
		histogram_cap = 0;
		histogram_minima = nullptr;
		histogram_minima_cap = 0;
		histogram_saddles = nullptr;
		histogram_saddles_cap = 0;

		log_histogram = nullptr;
		log_histogram_cap = 0;
		log_histogram_minima = nullptr;
		log_histogram_minima_cap = 0;
		log_histogram_saddles = nullptr;
		log_histogram_saddles_cap = 0;

		scratch = ThrustScratch();
	}

	/******************************************************************************
	copy assignment sets this object's dynamic memory pointers to null and their
	capacities to 0 (the base makes it a default, not reused, object)
	******************************************************************************/
	IPM& operator=(const IPM& other)
	{
        if (this == &other) return *this;

		Microlensing<T>::operator=(other);

		d_pixels = nullptr;
		d_pixels_cap = 0;
		d_pixels_minima = nullptr;
		d_pixels_minima_cap = 0;
		d_pixels_saddles = nullptr;
		d_pixels_saddles_cap = 0;
		h_pixels = nullptr;
		h_pixels_minima = nullptr;
		h_pixels_saddles = nullptr;
		have_parity_maps = false;
		have_histograms = false;
		d_steps = nullptr;
		d_steps_cap = 0;
		d_clamp_count = nullptr;
		d_clamp_count_cap = 0;
		d_clamp_total = nullptr;
		d_clamp_total_cap = 0;
		d_percentage = nullptr;
		d_percentage_cap = 0;
		leaf_buffers = ipm_leaf::LeafBuffers<T>();

		histogram = nullptr;
		histogram_cap = 0;
		histogram_minima = nullptr;
		histogram_minima_cap = 0;
		histogram_saddles = nullptr;
		histogram_saddles_cap = 0;

		log_histogram = nullptr;
		log_histogram_cap = 0;
		log_histogram_minima = nullptr;
		log_histogram_minima_cap = 0;
		log_histogram_saddles = nullptr;
		log_histogram_saddles_cap = 0;

		scratch = ThrustScratch();

		return *this;
	}

	bool run(int verbose)
	{
		if (!Microlensing<T>::set_cuda_devices(verbose)) return false;
		if (reuse_memory)
		{
			if (!start_reused_run(verbose)) return false;
		}
		else if (!clear_memory(verbose)) return false;
		if (!check_input_params(verbose)) return false;
		if (!calculate_derived_params(verbose)) return false;
		if (!allocate_initialize_memory(verbose)) return false;
		if (!Microlensing<T>::populate_star_array(verbose)) return false;
		if (!Microlensing<T>::create_tree(verbose)) return false;
		if (!build_compact_tree(verbose)) return false;
		if (!shoot_cells(verbose)) return false;
		if (!create_histograms(verbose)) return false;

		//everything has finished on the stream before run() returns
		cudaStreamSynchronize(stream);
		if (cuda_error("cudaStreamSynchronize", false, __FILE__, __LINE__)) return false;

		return true;
	}

	bool save(int verbose)
	{
		if (!use_device()) return false;
		if (!write_files(verbose)) return false;

		return true;
	}

	/******************************************************************************
	set every input parameter to its default, as in a new object, to start the
	next map of a reused object. device, reuse_memory, the stream and every buffer
	stay as they are
	******************************************************************************/
	void reset_inputs()
	{
		const IPM<T> d;
		copy_inputs_from(d);
	}

	/******************************************************************************
	a new object that holds this object's last map: its input parameters, derived
	values and device, a host copy of the stars and of the histograms, and no
	device memory. save_host writes the files of save() from it, with the maps
	the caller copied (copy_pixels*). the caller owns the object (delete)

	\return IPM<T>* -- the snapshot, or nullptr if this object has not run or a
	                   copy or allocation fails
	******************************************************************************/
	IPM<T>* snapshot() const
	{
		if (stars == nullptr || num_stars < 1 || is_snapshot) return nullptr;
		IPM<T>* s = new (std::nothrow) IPM<T>();
		if (s == nullptr) return nullptr;
		s->copy_inputs_from(*this);
		s->copy_derived_from(*this);
		s->device = device;
		s->is_snapshot = true;

		bool ok = use_device();
		try
		{
			s->host_stars.resize(num_stars);
			if (write_histograms && have_histograms)
			{
				auto copy_histogram = [&](const int* d_hist, std::vector<int>& h_vec, int n)
				{
					if (d_hist == nullptr || n < 1) return false;
					h_vec.resize(n);
					cudaMemcpyAsync(h_vec.data(), d_hist, n * sizeof(int), cudaMemcpyDeviceToHost, stream);
					return !cuda_error("cudaMemcpyAsync(*h_hist)", false, __FILE__, __LINE__);
				};
				ok = ok && copy_histogram(histogram, s->h_histogram, histogram_length)
					&& copy_histogram(log_histogram, s->h_log_histogram, log_histogram_length);
				if (write_parities && have_parity_maps)
				{
					ok = ok && copy_histogram(histogram_minima, s->h_histogram_minima, histogram_length)
						&& copy_histogram(histogram_saddles, s->h_histogram_saddles, histogram_length)
						&& copy_histogram(log_histogram_minima, s->h_log_histogram_minima, log_histogram_length)
						&& copy_histogram(log_histogram_saddles, s->h_log_histogram_saddles, log_histogram_length);
				}
			}
		}
		catch (const std::bad_alloc&)
		{
			std::cerr << "Error. Unable to allocate the host copies of a snapshot.\n";
			ok = false;
		}
		if (ok)
		{
			cudaMemcpyAsync(s->host_stars.data(), stars, 1ULL * num_stars * sizeof(star<T>), cudaMemcpyDeviceToHost, stream);
			ok = !cuda_error("cudaMemcpyAsync(*host_stars)", false, __FILE__, __LINE__);
		}
		if (ok)
		{
			//the copies are pageable and the stream is not the legacy stream: wait for them
			cudaStreamSynchronize(stream);
			ok = !cuda_error("cudaStreamSynchronize", false, __FILE__, __LINE__);
		}
		if (!ok)
		{
			delete s;
			return nullptr;
		}
		return s;
	}

	/******************************************************************************
	write the files of save() from host data, with no CUDA call. only for a
	snapshot

	\param pixels, minima, saddles -- host copies of the total, minima and saddle
	                                  maps (num_pixels_y.re * num_pixels_y.im values
	                                  each). the parity maps are needed with write_parities
	\param verbose -- verbosity level

	\return bool -- false (with an error message) for an object that is not a
	                snapshot or a needed map that is null, or a file that cannot
	                be written
	******************************************************************************/
	bool save_host(const T* pixels, const T* minima, const T* saddles, int verbose)
	{
		if (!is_snapshot)
		{
			std::cerr << "Error. save_host is only available for a snapshot.\n";
			return false;
		}
		if (write_maps && (pixels == nullptr || (write_parities && (minima == nullptr || saddles == nullptr))))
		{
			std::cerr << "Error. save_host needs every map that is to be written.\n";
			return false;
		}
		const T* const host_maps[3] = {pixels, minima, saddles};
		return write_files(verbose, host_maps);
	}

	/******************************************************************************
	whether the last run's shoot_cells used the step buffer of the perimeter
	deposit (false when every triangle took the direct deposit)
	******************************************************************************/
	bool get_perimeter_deposit_used() const	{return perimeter_deposit_used;}

	/******************************************************************************
	copy a map (num_pixels_y.re * num_pixels_y.im values) from the device into
	dst. false if the map does not exist (before run(), or the parity maps without
	write_parities) or the copy fails
	******************************************************************************/
	bool copy_pixels(T* dst) const			{return copy_map(d_pixels, dst, "cudaMemcpy(*d_pixels)");}
	bool copy_pixels_minima(T* dst) const	{return copy_map(pixels_minima(), dst, "cudaMemcpy(*d_pixels_minima)");}
	bool copy_pixels_saddles(T* dst) const	{return copy_map(pixels_saddles(), dst, "cudaMemcpy(*d_pixels_saddles)");}

	/******************************************************************************
	a host copy of a map, made on the first call after run() and valid until the
	next run() or the object's destruction. nullptr where copy_pixels*() would
	return false, or if the host copy cannot be allocated
	******************************************************************************/
	T* get_pixels()				{return host_copy(d_pixels, h_pixels, "cudaMemcpy(*d_pixels)");}
	T* get_pixels_minima()		{return host_copy(pixels_minima(), h_pixels_minima, "cudaMemcpy(*d_pixels_minima)");}
	T* get_pixels_saddles()		{return host_copy(pixels_saddles(), h_pixels_saddles, "cudaMemcpy(*d_pixels_saddles)");}
	double get_t_shoot_cells()	{return t_shoot_cells;}

	/******************************************************************************
	device and managed memory this object holds, in bytes: the maps, the step
	buffer and the histograms (pixel_bytes); the curand states, the two star
	arrays and the binomial coefficients (star_bytes); and the tree pool and its
	counters, the compact tree, the clamp counters, the progress counter and the
	scratch of the min/max pass (tree_bytes)
	******************************************************************************/
	void allocated_bytes(unsigned long long& pixel_bytes, unsigned long long& star_bytes, unsigned long long& tree_bytes) const
	{
		Microlensing<T>::allocated_bytes(star_bytes, tree_bytes);
		pixel_bytes = 1ULL * (d_pixels_cap + d_pixels_minima_cap + d_pixels_saddles_cap + d_steps_cap) * sizeof(T)
			+ 1ULL * (histogram_cap + histogram_minima_cap + histogram_saddles_cap
				+ log_histogram_cap + log_histogram_minima_cap + log_histogram_saddles_cap) * sizeof(int);
		tree_bytes += 1ULL * leaf_buffers.leaves_cap * sizeof(ipm_leaf::LeafRecord<T>)
			+ 1ULL * (leaf_buffers.coef_re_cap + leaf_buffers.coef_im_cap) * sizeof(T)
			+ 1ULL * leaf_buffers.direct_cap * sizeof(ipm_leaf::StarXYM<T>)
			+ 1ULL * leaf_buffers.counters_cap * sizeof(int)
			+ 1ULL * d_clamp_count_cap * sizeof(unsigned long long) + 1ULL * d_clamp_total_cap * sizeof(double)
			+ 1ULL * d_percentage_cap * sizeof(unsigned long long) + scratch.cap;
	}

	/******************************************************************************
	bytes per star of the tree pool, its counters, the compact tree and the leaf
	buffers: the largest allocation per star measured by
	tests/batch/measure_tree_bytes.py (nb 1000^2 and 5000^2, hm100_f, stars1e6_f),
	times 1.1, rounded up to a multiple of 64. (A margin of 1.5 puts the estimate
	of stars1e6_f above 1.5 times its allocation, the bound the estimate is
	tested against, because the bytes per star differ by a factor of 1.7 between
	the configurations.) Per-star values measured on an A100:
	float:  1416.10, 1432.86, 1724.50, 1323.65 -> 1920
	double: 2313.98, 2354.55, 2798.78, 2175.06 -> 3136
	******************************************************************************/
	static constexpr unsigned long long tree_bytes_per_star = sizeof(T) == 8 ? 3136 : 1920;

	/******************************************************************************
	estimate the device memory a run of this object's inputs needs, in bytes, and
	the host memory its saved results need. This object is not changed: the
	estimate is made on a copy of the inputs. For a map without a star file no
	CUDA call is made.

	\param pixel_bytes  -- maps, step buffer and histograms (16 MiB per map kind: the
	                       histogram length depends on the magnification range, and a
	                       reused object grows its histograms with 1.25 headroom)
	\param star_bytes   -- curand states, two star arrays and binomial coefficients
	\param tree_bytes   -- tree pool and compact tree, from tree_bytes_per_star
	\param result_bytes -- the maps and the stars of a finished map held on the host

	\return bool -- true if the inputs are valid, false if not
	******************************************************************************/
	bool estimate_memory(unsigned long long& pixel_bytes, unsigned long long& star_bytes,
		unsigned long long& tree_bytes, unsigned long long& result_bytes) const
	{
		IPM<T> probe;
		probe.copy_inputs_from(*this);
		if (!probe.check_input_params(0)) return false;
		if (!probe.calculate_derived_params(0)) return false;

		unsigned long long npix = 1ULL * probe.num_pixels_y.re * probe.num_pixels_y.im;
		unsigned long long n_maps = probe.write_parities ? 3 : 1;
		unsigned long long n_stars = probe.num_stars;

		pixel_bytes = npix * sizeof(T) * (n_maps + 1) + (probe.write_histograms ? n_maps * (16ULL << 20) : 0);
		star_bytes = n_stars * (sizeof(curandState) + 2 * sizeof(star<T>))
			+ 1ULL * (2 * treenode::MAX_EXPANSION_ORDER * (2 * treenode::MAX_EXPANSION_ORDER + 3) / 2 + 1) * sizeof(long long);
		tree_bytes = n_stars * tree_bytes_per_star + (1ULL << 20);
		result_bytes = npix * sizeof(T) * n_maps + n_stars * 3 * sizeof(T);

		return true;
	}

	/******************************************************************************
	free every buffer of this object (a reused object keeps them otherwise). the
	object can run again afterwards
	******************************************************************************/
	bool release_memory()			{return clear_memory(0);}

};

