#pragma once

#include "complex.cuh"
#include "util/util.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric> //for std::reduce
#include <string>
#include <vector>


/******************************************************************************
transpose array

\param z1 -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param z2 -- pointer to transposed array of values
******************************************************************************/
template <typename T>
__global__ void transpose_array_kernel(T* z1, int nrows, int ncols, T* z2)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int a = x_index; a < nrows * ncols; a += x_stride)
	{
		int col = a % ncols;
		int row = (a - col) / ncols;

		z2[col * nrows + row] = z1[a];
	}
}

/******************************************************************************
complex point in the source plane converted to pixel position

\param w -- complex source plane position
\param hly -- half length of the source plane receiving region
\param npixels -- number of pixels per side for the source plane receiving
				  region

\return (w + hly) * npixels / (2 * hly)
******************************************************************************/
template <typename T, typename U>
__device__ Complex<T> point_to_pixel(Complex<U> w, Complex<U> hly, Complex<int> npixels)
{
	Complex<T> result((w + hly).re * npixels.re / (2 * hly.re), (w + hly).im * npixels.im / (2 * hly.im));
	return result;
}

/******************************************************************************
calculate the histogram of values for the pixel array

\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
\param hist_min -- minimum value in the histogram
\param histogram -- pointer to histogram
\param factor -- factor by which to multiply the pixel values before casting
                 to integers for the histogram
******************************************************************************/
template <typename T>
__global__ void histogram_kernel(T* pixels, Complex<int> npixels, int hist_min, int* histogram, int factor = 1)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < npixels.re; i += x_stride)
	{
		for (int j = y_index; j < npixels.im; j += y_stride)
		{
			int index = std::round(1.0 * pixels[j * npixels.re + i] * factor - hist_min);
			atomicAdd(&histogram[index], 1);
		}
	}
}

/******************************************************************************
calculate the histogram of log_10(values) for the pixel array

\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
\param hist_min -- minimum value in the histogram
\param histogram -- pointer to histogram
\param factor -- factor by which to multiply the pixel values before casting
                 to integers for the histogram
******************************************************************************/
template <typename T>
__global__ void log_histogram_kernel(T* pixels, Complex<int> npixels, int hist_min, int* histogram, int factor = 1)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < npixels.re; i += x_stride)
	{
		for (int j = y_index; j < npixels.im; j += y_stride)
		{
			int index = std::round(std::log10(pixels[j * npixels.re + i]) * factor - hist_min);
			atomicAdd(&histogram[index], 1);
		}
	}
}

/******************************************************************************
read array of complex values from disk into array

\param vals -- pointer to array of values
\param nrows -- pointer to number of rows in array
\param ncols -- pointer to number of columns in array
\param fname -- location of the .bin file to read from

\return bool -- true if file is successfully read, false if not
******************************************************************************/
template <typename T>
bool read_complex_array(Complex<T>*& vals, int& nrows, int& ncols, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	if (fpath.extension() != ".bin")
	{
		std::cerr << "Error. File " << fname << " is not a .bin file.\n";
		return false;
	}

	std::ifstream infile;

	std::error_code err;
	std::uintmax_t fsize = std::filesystem::file_size(fname, err);

	if (err)
	{
		std::cerr << "Error determining size of input file " << fname << "\n";
		return false;
	}

	infile.open(fname, std::ios_base::binary);

	if (!infile.is_open())
	{
		std::cerr << "Error. Failed to open file " << fname << "\n";
		return false;
	}

	infile.read((char*)(&nrows), sizeof(int));
	infile.read((char*)(&ncols), sizeof(int));
	if (nrows < 1 || ncols < 1)
	{
		std::cerr << "Error. File " << fname << " does not contain valid values for num_rows and num_cols.\n";
		return false;
	}

	/******************************************************************************
	allocate memory for array if needed
	******************************************************************************/
	if (vals == nullptr)
	{
		cudaMallocManaged(&vals, nrows * ncols * sizeof(Complex<T>));
		if (cuda_error("cudaMallocManaged(*vals)", false, __FILE__, __LINE__)) return false;
	}


	/******************************************************************************
	objects in the file are nrows + ncols + complex values
	******************************************************************************/
	if (fsize == 2 * sizeof(int) + nrows * ncols * sizeof(Complex<T>))
	{
		infile.read((char*)vals, nrows * ncols * sizeof(Complex<T>));
	}
	else if (fsize == 2 * sizeof(int) + nrows * ncols * sizeof(Complex<float>))
	{
		Complex<float>* temp_vals = new (std::nothrow) Complex<float>[nrows * ncols];
		if (!temp_vals)
		{
			std::cerr << "Error. Memory allocation for *temp_vals failed.\n";
			return false;
		}
		infile.read((char*)temp_vals, nrows * ncols * sizeof(Complex<float>));
		for (int i = 0; i < nrows * ncols; i++)
		{
			vals[i] = Complex<T>(temp_vals[i]);
		}
		delete[] temp_vals;
		temp_vals = nullptr;
	}
	else if (fsize == 2 * sizeof(int) + nrows * ncols * sizeof(Complex<double>))
	{
		Complex<double>* temp_vals = new (std::nothrow) Complex<double>[nrows * ncols];
		if (!temp_vals)
		{
			std::cerr << "Error. Memory allocation for *temp_vals failed.\n";
			return false;
		}
		infile.read((char*)temp_vals, nrows * ncols * sizeof(Complex<double>));
		for (int i = 0; i < nrows * ncols; i++)
		{
			vals[i] = Complex<T>(temp_vals[i]);
		}
		delete[] temp_vals;
		temp_vals = nullptr;
	}
	else
	{
		std::cerr << "Error. Binary file does not contain valid single or double precision numbers.\n";
		return false;
	}

	infile.close();

	return true;
}

/******************************************************************************
write array of values to disk

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the file to write to

\return bool -- true if file is successfully written, false if not
******************************************************************************/
template <typename T>
bool write_array(T* vals, int nrows, int ncols, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	if (fpath.extension() != ".bin")
	{
		std::cerr << "Error. File " << fname << " is not a .bin file.\n";
		return false;
	}

	std::ofstream outfile;

	outfile.open(fname, std::ios_base::binary);

	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << fname << "\n";
		return false;
	}
	outfile.write((char*)(&nrows), sizeof(int));
	outfile.write((char*)(&ncols), sizeof(int));
	outfile.write((char*)vals, nrows * ncols * sizeof(T));
	outfile.close();

	return true;
}

/******************************************************************************
write ragged array of values to disk

\param vals -- pointer to array of values
\param n_cols_per_row -- vector of the number of columns for each row
\param fname -- location of the file to write to

\return bool -- true if file is successfully written, false if not
******************************************************************************/
template <typename T>
bool write_ragged_array(T* vals, std::vector<int> n_cols_per_row, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	if (fpath.extension() != ".bin")
	{
		std::cerr << "Error. File " << fname << " is not a .bin file.\n";
		return false;
	}

	std::ofstream outfile;

	outfile.open(fname, std::ios_base::binary);

	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << fname << "\n";
		return false;
	}
	int n_rows = n_cols_per_row.size();
	outfile.write((char*)(&n_rows), sizeof(int));
	for (int i = 0; i < n_rows; i++)
	{
		int start = std::reduce(&n_cols_per_row[0], &n_cols_per_row[i], 0);
		outfile.write((char*)(&n_cols_per_row[i]), sizeof(int));
		outfile.write((char*)(&vals[start]), n_cols_per_row[i] * sizeof(T));
	}
	outfile.close();

	return true;
}

/******************************************************************************
write histogram

\param histogram -- pointer to histogram
\param n -- length of histogram
\param minrays -- minimum number of rays
\param fname -- location of the file to write to

\return bool -- true if file is successfully written, false if not
******************************************************************************/
template <typename T>
bool write_histogram(T* histogram, int n, int minrays, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	if (fpath.extension() != ".txt")
	{
		std::cerr << "Error. File " << fname << " is not a .txt file.\n";
		return false;
	}

	std::ofstream outfile;

	outfile.precision(9);
	outfile.open(fname);

	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << fname << "\n";
		return false;
	}
	for (int i = 0; i < n; i++)
	{
		if (histogram[i] != 0)
		{
			outfile << i + minrays << " " << histogram[i] << "\n";
		}
	}
	outfile.close();

	return true;
}

