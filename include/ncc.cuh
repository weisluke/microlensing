#pragma once

#include "array_functions.cuh"
#include "complex.cuh"
#include "ncc_functions.cuh"
#include "stopwatch.hpp"
#include "util/util.cuh"

#include <thrust/execution_policy.h> //for thrust::device
#include <thrust/extrema.h> // for thrust::min_element, thrust::max_element
#include <thrust/fill.h> //for thrust::fill

#include <chrono> //for setting random seed with clock
#include <fstream> //for writing files
#include <iostream>
#include <limits> //for std::numeric_limits
#include <string>


template <typename T>
class NCC
{

public:
	/******************************************************************************
	default input variables
	******************************************************************************/
	std::string infile_prefix = "./";
	Complex<T> center_y = Complex<T>();
	Complex<T> half_length_y = Complex<T>(5, 5);
	Complex<int> num_pixels_y = Complex<int>(1000, 1000);
	int over_sample = 2;
	int write_maps = 1;
	int write_histograms = 1;
	std::string outfile_prefix = "./";


private:
	/******************************************************************************
	constant variables
	******************************************************************************/
	const std::string caustics_file = "ccf_caustics.bin";
	const std::string outfile_type = ".bin";

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
	double t_ncc;
	double t_reduce;

	/******************************************************************************
	derived variables
	******************************************************************************/
	int num_rows;
	int num_cols;

	/******************************************************************************
	dynamic memory
	******************************************************************************/
	Complex<T>* caustics = nullptr;
	int* num_crossings = nullptr;

	int min_num;
	int max_num;
	int histogram_length;
	int* histogram = nullptr;

	

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
			std::cout << "Available CUDA capable devices:\n";

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
			print_verbose("More than one CUDA capable device detected. Defaulting to first device.\n", verbose, 2);
		}
		cudaSetDevice(0);
		if (cuda_error("cudaSetDevice", false, __FILE__, __LINE__)) return false;
		cudaGetDeviceProperties(&cuda_device_prop, 0);
		if (cuda_error("cudaGetDeviceProperties", false, __FILE__, __LINE__)) return false;

		print_verbose("Done setting device.\n", verbose, 3);
		return true;
	}

	//optional return or not, so memory can be cleared in destructor without error checking
	bool clear_memory(int verbose, bool return_on_error = true)
	{
		print_verbose("Clearing memory...\n", verbose, 3);
		
		/******************************************************************************
		free memory and set variables to nullptr
		******************************************************************************/

		cudaFree(caustics);
		if (return_on_error && cuda_error("cudaFree(*caustics)", false, __FILE__, __LINE__)) return false;
		caustics = nullptr;
		
		cudaFree(num_crossings);
		if (return_on_error && cuda_error("cudaFree(*num_crossings)", false, __FILE__, __LINE__)) return false;
		num_crossings = nullptr;
		
		cudaFree(histogram);
		if (return_on_error && cuda_error("cudaFree(*histogram)", false, __FILE__, __LINE__)) return false;
		histogram = nullptr;

		print_verbose("Done clearing memory.\n", verbose, 3);
		return true;
	}

	bool check_input_params(int verbose)
	{
		print_verbose("Checking input parameters...\n", verbose, 3);


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

		if (over_sample < 0)
		{
			std::cerr << "Error. over_sample must be an integer >= 0\n";
			return false;
		}

		if (write_maps != 0 && write_maps != 1)
		{
			std::cerr << "Error. write_maps must be 1 (true) or 0 (false).\n";
			return false;
		}

		if (write_histograms != 0 && write_histograms != 1)
		{
			std::cerr << "Error. write_histograms must be 1 (true) or 0 (false).\n";
			return false;
		}


		print_verbose("Done checking input parameters.\n", verbose, 3);
		
		return true;
	}
	
	bool read_caustics(int verbose)
	{
		print_verbose("Reading in caustics...\n", verbose, 1);
		stopwatch.start();

		std::string fname = infile_prefix + caustics_file;

		num_rows = 0;
		num_cols = 0;

		if (!read_complex_array<T>(caustics, num_rows, num_cols, fname))
		{
			std::cerr << "Error. Unable to read caustics from file " << fname << "\n";
			return false;
		}

		set_param("num_rows", num_rows, num_rows, verbose);
		set_param("num_cols", num_cols, num_cols, verbose);
		
		if (num_rows < 1 || num_cols < 2)
		{
			std::cerr << "Error. File " << fname << " does not contain valid values for num_rows and num_cols.\n";
			return false;
		}

		t_elapsed = stopwatch.stop();
		print_verbose("Done reading in caustics. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 1);

		return true;
	}

	bool allocate_initialize_memory(int verbose)
	{
		print_verbose("Allocating memory...\n", verbose, 3);
		stopwatch.start();

		/******************************************************************************
		increase the number of pixels by 2^over_sample for initial sampling
		******************************************************************************/
		num_pixels_y.re <<= over_sample;
		num_pixels_y.im <<= over_sample;
		set_param("num_pixels_y1", num_pixels_y.re, num_pixels_y.re, verbose);
		set_param("num_pixels_y2", num_pixels_y.im, num_pixels_y.im, verbose);

		/******************************************************************************
		allocate memory for pixels
		******************************************************************************/
		cudaMallocManaged(&num_crossings, num_pixels_y.re * num_pixels_y.im * sizeof(int));
		if (cuda_error("cudaMallocManaged(*num_crossings)", false, __FILE__, __LINE__)) return false;

		t_elapsed = stopwatch.stop();
		print_verbose("Done allocating memory. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 3);


		/******************************************************************************
		initialize pixel values
		******************************************************************************/
		print_verbose("Initializing array values...\n", verbose, 3);
		stopwatch.start();

		thrust::fill(thrust::device, num_crossings, num_crossings + num_pixels_y.re * num_pixels_y.im, 0);

		t_elapsed = stopwatch.stop();
		print_verbose("Done initializing array values. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 3);

		return true;
	}

	bool calculate_num_caustic_crossings(int verbose)
	{
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_rows, num_cols - 1);

		unsigned long long int* percentage = nullptr;
		cudaMallocManaged(&percentage, sizeof(unsigned long long int));
		if (cuda_error("cudaMallocManaged(*percentage)", false, __FILE__, __LINE__)) return false;

		*percentage = 1;

		/******************************************************************************
		calculate number of caustic crossings and calculate time taken in seconds
		******************************************************************************/
		print_verbose("Calculating number of caustic crossings...\n", verbose, 1);
		stopwatch.start();
		find_num_caustic_crossings_kernel<T> <<<blocks, threads>>> (caustics, num_rows, num_cols, center_y, half_length_y, num_crossings, num_pixels_y, percentage, verbose);
		if (cuda_error("find_num_caustic_crossings_kernel", true, __FILE__, __LINE__)) return false;
		t_ncc = stopwatch.stop();
		print_verbose("\nDone calculating number of caustic crossings. Elapsed time: " << t_ncc << " seconds.\n", verbose, 1);


		cudaFree(percentage);
		if (cuda_error("cudaFree(*percentage)", false, __FILE__, __LINE__)) return false;
		percentage = nullptr;


		/******************************************************************************
		downsample number of caustic crossings and calculate time taken in seconds
		******************************************************************************/
		print_verbose("Downsampling number of caustic crossings...\n", verbose, 1);
		stopwatch.start();

		for (int i = 0; i < over_sample; i++)
		{
			print_verbose("Loop " << i + 1 << " / " << over_sample << "\n", verbose, 2);
			num_pixels_y.re >>= 1;
			num_pixels_y.im >>= 1;

			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels_y.re, num_pixels_y.im);

			reduce_pix_array_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels_y);
			if (cuda_error("reduce_pix_array_kernel", true, __FILE__, __LINE__)) return false;

			set_threads(threads, 512);
			set_blocks(threads, blocks, num_pixels_y.im);
			for (int j = 1; j < num_pixels_y.re; j++)
			{
				shift_pix_column_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels_y, j);
			}
			set_threads(threads, 512);
			set_blocks(threads, blocks, num_pixels_y.re);
			for (int j = 1; j < num_pixels_y.im; j++)
			{
				shift_pix_row_kernel<T> <<<blocks, threads>>> (num_crossings, num_pixels_y, j);
			}
			if (cuda_error("shift_pix_kernel", true, __FILE__, __LINE__)) return false;
		}
		double t_reduce = stopwatch.stop();
		print_verbose("Done downsampling number of caustic crossings. Elapsed time: " << t_reduce << " seconds.\n", verbose, 1);
		
		min_num = *thrust::min_element(thrust::device, num_crossings, num_crossings + num_pixels_y.re * num_pixels_y.im);
		if (min_num < 0)
		{
			std::cerr << "Error. Number of caustic crossings should be >= 0\n";
			return false;
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

			min_num = *thrust::min_element(thrust::device, num_crossings, num_crossings + num_pixels_y.re * num_pixels_y.im);
			max_num = *thrust::max_element(thrust::device, num_crossings, num_crossings + num_pixels_y.re * num_pixels_y.im);

			histogram_length = max_num - min_num + 1;

			cudaMallocManaged(&histogram, histogram_length * sizeof(int));
			if (cuda_error("cudaMallocManaged(*histogram)", false, __FILE__, __LINE__)) return false;

			thrust::fill(thrust::device, histogram, histogram + histogram_length, 0);

			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels_y.re, num_pixels_y.im);

			histogram_kernel<int> <<<blocks, threads>>> (num_crossings, num_pixels_y, min_num, histogram);
			if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return false;

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

		print_verbose("Writing NCC<T> parameter info...\n", verbose, 2);
		fname = outfile_prefix + "ncc_parameter_info.txt";
		outfile.open(fname);
		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		outfile << "center_y1 " << center_y.re << "\n";
		outfile << "center_y2 " << center_y.im << "\n";
		outfile << "half_length_y1 " << half_length_y.re << "\n";
		outfile << "half_length_y2 " << half_length_y.im << "\n";
		outfile << "num_pixels_y1 " << num_pixels_y.re << "\n";
		outfile << "num_pixels_y2 " << num_pixels_y.im << "\n";
		outfile << "over_sample " << over_sample << "\n";
		outfile << "t_ncc " << t_ncc << "\n";
		outfile << "t_reduce " << t_reduce << "\n";
		outfile.close();
		print_verbose("Done writing NCC<T> parameter info to file " << fname << "\n", verbose, 1);


		/******************************************************************************
		histogram of number of caustic crossings map
		******************************************************************************/
		if (write_histograms)
		{
			print_verbose("Writing number of caustic crossings histogram...\n", verbose, 2);
			fname = outfile_prefix + "ncc_ncc_numpixels.txt";
			if (!write_histogram<int>(histogram, histogram_length, min_num, fname))
			{
				std::cerr << "Error. Unable to write caustic crossings histogram to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing number of caustic crossings histogram to file " << fname << "\n", verbose, 1);
		}


		/******************************************************************************
		write number of caustic crossings
		******************************************************************************/
		if (write_maps)
		{
			print_verbose("Writing number of caustic crossings...\n", verbose, 2);
			fname = outfile_prefix + "ncc_ncc" + outfile_type;
			if (!write_array<int>(num_crossings, num_pixels_y.im, num_pixels_y.re, fname))
			{
				std::cerr << "Error. Unable to write number of caustic crossings to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing number of caustic crossings to file " << fname << "\n", verbose, 1);
		}

		return true;
	}


public:

	/******************************************************************************
	class initializer is empty
	******************************************************************************/
	NCC()
	{

	}

	/******************************************************************************
	class destructor clears memory with no output or error checking
	******************************************************************************/
	~NCC()
	{
		clear_memory(0, false);
	}

	/******************************************************************************
	copy constructor sets this object's dynamic memory pointers to null
	******************************************************************************/
	NCC(const NCC& other)
	{
		caustics = nullptr;
		num_crossings = nullptr;

		histogram = nullptr;
	}

	/******************************************************************************
	copy assignment sets this object's dynamic memory pointers to null
	******************************************************************************/
	NCC& operator=(const NCC& other)
	{
        if (this == &other) return *this;

		caustics = nullptr;
		num_crossings = nullptr;

		histogram = nullptr;

		return *this;
	}

	bool run(int verbose)
	{
		if (!set_cuda_devices(verbose)) return false;
		if (!clear_memory(verbose)) return false;
		if (!check_input_params(verbose)) return false;
		if (!read_caustics(verbose)) return false;
		if (!allocate_initialize_memory(verbose)) return false;
		if (!calculate_num_caustic_crossings(verbose)) return false;
		if (!create_histograms(verbose)) return false;

		return true;
	}

	bool save(int verbose)
	{
		if (!write_files(verbose)) return false;

		return true;
	}

	int* get_num_crossings()	{return num_crossings;}

};

