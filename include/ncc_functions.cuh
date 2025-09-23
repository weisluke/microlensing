#pragma once

#include "array_functions.cuh"
#include "complex.cuh"
#include "util/complex_math_util.cuh"
#include "util/math_util.cuh"
#include "util/util.cuh"

#include <cstdint> //for std::uintmax_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <string>
#include <system_error> //for std::error_code


/******************************************************************************
determine whether a point lies within a rectangular region centered on the
origin. our particular consideration is a rectangular region such that
-hl < y < hl, and -hl < x
this is so that we can keep track of how many caustics are crossed in a line
extending from a particular point (e.g., a pixel center) to positive infinity
in the x-direction

\param p0 -- point
\param hl -- half length of the square region

\return true if point in region, false if not
******************************************************************************/
template <typename T>
__device__ bool point_in_region(Complex<T> p0, Complex<T> hl)
{
	if (p0.re > -hl.re && fabs(p0.im) < hl.im)
	{
		return true;
	}
	return false;
}

/******************************************************************************
given two points, corrects the first point so that it lies within the desired
region of the point_in_region function
this function implicitly assumes that the line connecting the two points
intersects, or lies within, the desired region

\param p1 -- first point
\param p2 -- second point
\param hl -- half length of the square region
\param npixels -- number of pixels per side for the square region

\return position of corrected point
******************************************************************************/
template <typename T>
__device__ Complex<T> corrected_point(Complex<T> p1, Complex<T> p2, Complex<T> hl, Complex<int> npixels)
{
	T x = p1.re;
	T y = p1.im;
	if (x <= -hl.re)
	{
		/******************************************************************************
		if the x position is outside of our region, calculate where the point would be
		with an x position 1/100 of a pixel inside the desired region
		1/100 used due to later code only considering line crossings at the half pixel
		mark
		******************************************************************************/
		x = -hl.re + static_cast<T>(0.01) * 2 * hl.re / npixels.re;
		y = get_y_intersection(x, p1, p2);
	}
	if (fabs(y) >= hl.im)
	{
		/******************************************************************************
		if the y position is outside of our region, calculate where the point would be
		with a y position 1/100 of a pixel inside the desired region
		1/100 used due to later code only considering line crossings at the half pixel
		mark
		******************************************************************************/
		y = sgn(y) * (hl.im - static_cast<T>(0.01) * 2 * hl.im / npixels.im);
		x = get_x_intersection(y, p1, p2);
	}

	return Complex<T>(x, y);
}

/******************************************************************************
calculate the number of caustic crossings

\param caustics -- array of caustic positions
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param center_y -- center of the source plane receiving region
\param hly -- half length of the source plane receiving region
\param num -- array of number of caustic crossings
\param npixels -- number of pixels per side for the square region
\param percentage -- pointer to percentage complete
\param verbose -- verbose level
******************************************************************************/
template <typename T>
__global__ void find_num_caustic_crossings_kernel(Complex<T>* caustics, int nrows, int ncols, Complex<T> center_y, Complex<T> hly, 
	int* num, Complex<int> npixels, unsigned long long int* percentage, int verbose)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < nrows; i += x_stride)
	{
		/******************************************************************************
		only able to calculate a caustic crossing if a caustic point has a succeeding
		point to form a line segment with (i.e., we are not at the end of the 2*pi
		phase chain that was traced out), hence ncols - 1
		******************************************************************************/
		for (int j = y_index; j < ncols - 1; j += y_stride)
		{
			/******************************************************************************
			initial and final point of the line segment
			we will be calculating what pixels this line segment crosses
			******************************************************************************/
			Complex<T> pt0 = caustics[i * ncols + j] - center_y;
			Complex<T> pt1 = caustics[i * ncols + j + 1] - center_y;

			/******************************************************************************
			if one of the endpoints lies within the region, correct the points so they both
			lie within the region
			possibly redundant if both lie within the region
			******************************************************************************/
			if (point_in_region(pt0, hly) || point_in_region(pt1, hly))
			{
				pt0 = corrected_point(pt0, pt1, hly, npixels);
				pt1 = corrected_point(pt1, pt0, hly, npixels);
			}
			/******************************************************************************
			else if both points are outside the region, but intersect the region boundaries
			(i.e., really long caustic segments), correct the points so they both lie
			within the region
			******************************************************************************/
			else if (get_x_intersection(hly.im, pt0, pt1) >= -hly.re
				|| get_x_intersection(-hly.im, pt0, pt1) >= -hly.re
				|| fabs(get_y_intersection(-hly.re, pt0, pt1)) <= hly.im)
			{
				pt0 = corrected_point(pt0, pt1, hly, npixels);
				pt1 = corrected_point(pt1, pt0, hly, npixels);
			}
			/******************************************************************************
			else continue on to the next caustic segment
			******************************************************************************/
			else
			{
				if (threadIdx.x == 0 && threadIdx.y == 0)
				{
					unsigned long long int p = atomicAdd(percentage, 1);
					unsigned long long int imax = ((nrows - 1) / blockDim.x + 1);
					imax *= (((ncols - 1) - 1) / blockDim.y + 1);
					if (p * 100 / imax > (p - 1) * 100 / imax)
					{
						print_progress(verbose, p, imax);
					}
				}
				continue;
			}

			/******************************************************************************
			make complex pixel start and end positions
			******************************************************************************/
			pt0 = point_to_pixel<T, T>(pt0, hly, npixels);
			pt1 = point_to_pixel<T, T>(pt1, hly, npixels);

			Complex<int> ypix;

			/******************************************************************************
			our caustics are traced in a clockwise manner
			if p1.y > p0.y, then as a line from a pixel center goes to infinity towards the
			right, we are entering a caustic region if we cross this segment
			conversely, this means a line from infinity to a pixel center leaves a caustic
			region at this segment, and so we subtract one from all pixels to the left of
			the line segment
			******************************************************************************/
			if (pt0.im < pt1.im)
			{
				/******************************************************************************
				for all y-pixel values from the start of the line segment to the end
				******************************************************************************/
				for (int k = static_cast<int>(pt0.im + 0.5); k < static_cast<int>(pt1.im + 0.5); k++)
				{
					/******************************************************************************
					find the x position of the caustic segment at the middle of the pixel
					find the integer pixel value for this x-position and the integer y pixel value
					   we would naively subtract 0.5 from x, as the caustic segment needs to be to
					   the right of the center. however, segments to the left of the center of the
					   0 pixel would still get cast to 0. we therefore add 0.5, cast, and subtract 
					   1 instead
					if the x-pixel value is greater than the number of pixels, we perform the
					subtraction on the whole pixel row
					******************************************************************************/
					T x1 = get_x_intersection(k + 0.5, pt0, pt1);
					ypix = Complex<int>(x1 + 0.5, k) - Complex<int>(1, 0);

					ypix.im = npixels.im - 1 - ypix.im;
					if (ypix.re > npixels.re - 1)
					{
						ypix.re = npixels.re - 1;
					}
					for (int l = 0; l <= ypix.re; l++)
					{
						if (ypix.im * npixels.re + l < 0 || ypix.im * npixels.re + l > npixels.re * npixels.im - 1)
						{
							printf("Error. Caustic crossing takes place outside the desired region.\n");
							continue;
						}
						atomicSub(&num[ypix.im * npixels.re + l], 1);
					}
				}
			}
			/******************************************************************************
			our caustics are traced in a clockwise manner
			if p1.y < p0.y, then as a line from a pixel center goes to infinity towards the
			right, we are leaving a caustic region if we cross this segment
			conversely, this means a line from infinity to a pixel center enters a caustic
			region at this segment, and so we add one to all pixels to the left of the line
			segment
			******************************************************************************/
			else if (pt0.im > pt1.im)
			{
				/******************************************************************************
				for all y-pixel values from the start of the line segment to the end
				******************************************************************************/
				for (int k = static_cast<int>(pt0.im + 0.5); k > static_cast<int>(pt1.im + 0.5); k--)
				{
					/******************************************************************************
					find the x position of the caustic segment at the middle of the pixel
					   since we are heading in the negative y direction, subtract 0.5
					find the integer pixel value for this x-position and the integer y pixel value
					   we would naively subtract 0.5 from x, as the caustic segment needs to be to
					   the right of the center. however, segments to the left of the center of the
					   0 pixel would still get cast to 0. we therefore add 0.5, cast, and subtract 
					   1 instead. we subtract 1 in the y direction as well since we are moving in
					   the negative direction
					if the x-pixel value is greater than the number of pixels, we perform the
					subtraction on the whole pixel row
					******************************************************************************/
					T x1 = get_x_intersection(k - 0.5, pt0, pt1);
					ypix = Complex<int>(x1 + 0.5, k) - Complex<int>(1, 1);

					ypix.im = npixels.im - 1 - ypix.im;
					if (ypix.re > npixels.re - 1)
					{
						ypix.re = npixels.re - 1;
					}
					for (int l = 0; l <= ypix.re; l++)
					{
						if (ypix.im * npixels.re + l < 0 || ypix.im * npixels.re + l > npixels.re * npixels.im - 1)
						{
							printf("Error. Caustic crossing takes place outside the desired region.\n");
							continue;
						}
						atomicAdd(&num[ypix.im * npixels.re + l], 1);
					}
				}
			}
			
			if (threadIdx.x == 0 && threadIdx.y == 0)
			{
				unsigned long long int p = atomicAdd(percentage, 1);
				unsigned long long int imax = ((nrows - 1) / blockDim.x + 1);
				imax *= (((ncols - 1) - 1) / blockDim.y + 1);
				if (p * 100 / imax > (p - 1) * 100 / imax)
				{
					print_progress(verbose, p, imax);
				}
			}
		}
	}
}

/******************************************************************************
reduce the pixel array

\param num -- array of number of caustic crossings
\param npixels -- number of pixels per side for the square region
******************************************************************************/
template <typename T>
__global__ void reduce_pix_array_kernel(int* num, Complex<int> npixels)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < npixels.re; i += x_stride)
	{
		for (int j = y_index; j < npixels.im; j += y_stride)
		{
			int n1 = num[2 * j * 2 * npixels.re + 2 * i];
			int n2 = num[2 * j * 2 * npixels.re + 2 * i + 1];
			int n3 = num[(2 * j + 1) * 2 * npixels.re + 2 * i];
			int n4 = num[(2 * j + 1) * 2 * npixels.re + 2 * i + 1];

			num[2 * j * 2 * npixels.re + 2 * i] = max(max(n1, n2), max(n3, n4));
		}
	}
}

/******************************************************************************
shift the provided pixel column from 2*i to i

\param num -- array of number of caustic crossings
\param npixels -- number of pixels per side for the square region
\param column -- the column to shift to
******************************************************************************/
template <typename T>
__global__ void shift_pix_column_kernel(int* num, Complex<int> npixels, int column)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int i = x_index; i < npixels.im; i += x_stride)
	{
		num[2 * i * 2 * npixels.re + column] = num[2 * i * 2 * npixels.re + 2 * column];
	}
}

/******************************************************************************
shift the provided pixel row from 2*i to i

\param num -- array of number of caustic crossings
\param npixels -- number of pixels per side for the square region
\param row -- the row to shift to
******************************************************************************/
template <typename T>
__global__ void shift_pix_row_kernel(int* num, Complex<int> npixels, int row)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int i = x_index; i < npixels.re; i += x_stride)
	{
		num[row * npixels.re + i] = num[2 * row * 2 * npixels.re + i];
	}
}

