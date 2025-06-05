#pragma once

#include "array_functions.cuh"
#include "complex.cuh"
#include "lens_equations.cuh"
#include "polygon.cuh"
#include "star.cuh"
#include "tree_node.cuh"
#include "util.cuh"


/******************************************************************************
shoot rays from image plane to source plane

\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param root -- pointer to root node
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if
						approximate
\param ray_half_sep -- half separation between central rays of shooting squares
\param num_ray_threads -- number of threads of rays for the image plane shooting region
\param center_x -- center of the image plane shooting region
\param hlx -- half length of the image plane shooting region
\param center_y -- center of the source plane receiving region
\param hly -- half length of the source plane receiving region
\param pixmin -- pointer to array of positive parity pixels
\param pixsad -- pointer to array of negative parity pixels
\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
\param percentage -- pointer to percentage complete
\param verbose -- verbose level
******************************************************************************/
template <typename T>
__global__ void shoot_cells_kernel(T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth,
	Complex<T> ray_half_sep, Complex<int> num_ray_threads, Complex<T> center_x, Complex<T> hlx,
	Complex<T> center_y, Complex<T> hly, T* pixmin, T* pixsad, T* pixels, Complex<int> npixels, 
	unsigned long long int* percentage, int verbose)
{
	for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < num_ray_threads.im; j += blockDim.y * gridDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_ray_threads.re; i += blockDim.x * gridDim.x)
		{
			Complex<T> x[4];

			Complex<T> z = center_x - hlx + ray_half_sep + 2 * Complex<T>(ray_half_sep.re * i, ray_half_sep.im * j);

			x[0] = z + ray_half_sep;
			x[1] = z - ray_half_sep.conj();
			x[2] = z - ray_half_sep;
			x[3] = z + ray_half_sep.conj();

			Complex<T> y[4];
#pragma unroll
			for (int a = 0; a < 4; a++)
			{
				TreeNode<T>* node = treenode::get_nearest_node(x[a], root);
				y[a] = microlensing::w<T>(x[a], kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
				/******************************************************************************
				if the ray location is the same as a star position, we will have a nan returned
				******************************************************************************/
				if (isnan(y[a].re) || isnan(y[a].im))
				{
					if (threadIdx.x == 0 && threadIdx.y == 0)
					{
						unsigned long long int p = atomicAdd(percentage, 1);
						unsigned long long int imax = ((num_ray_threads.re - 1) / blockDim.x + 1);
						imax *= ((num_ray_threads.im - 1) / blockDim.y + 1);
						if (p * 100 / imax > (p - 1) * 100 / imax)
						{
							device_print_progress(verbose, p, imax);
						}
					}
					break;
					continue;
				}
				/******************************************************************************
				shift ray position relative to center
				******************************************************************************/
				y[a] -= center_y;
				
				y[a] = point_to_pixel<T, T>(y[a], hly, npixels);
				/******************************************************************************
				reverse y coordinate so array forms image in correct orientation
				******************************************************************************/
				y[a].im = npixels.im - y[a].im;
			}

			Polygon<T> y_poly;

			T image_plane_area = 2 * ray_half_sep.re * ray_half_sep.im * npixels.re * npixels.im / (2 * hly.re * 2 * hly.im);

			y_poly.points[0] = y[0];
			y_poly.points[1] = y[1];
			y_poly.points[2] = y[2];
			y_poly.numsides = 3;
			if (fabs(y_poly.area()) < 10000 * image_plane_area)
			{
				if (pixmin && pixsad)
				{
					if (y_poly.area() < 0)
					{
						y_poly.allocate_area_among_pixels(image_plane_area, pixmin, npixels);
					}
					else
					{
						y_poly.allocate_area_among_pixels(image_plane_area, pixsad, npixels);
					}
				}
				else
				{
					y_poly.allocate_area_among_pixels(image_plane_area, pixels, npixels);
				}
			}

			y_poly.points[0] = y[2];
			y_poly.points[1] = y[3];
			y_poly.points[2] = y[0];
			y_poly.numsides = 3;
			if (fabs(y_poly.area()) < 10000 * image_plane_area)
			{
				if (pixmin && pixsad)
				{
					if (y_poly.area() < 0)
					{
						y_poly.allocate_area_among_pixels(image_plane_area, pixmin, npixels);
					}
					else
					{
						y_poly.allocate_area_among_pixels(image_plane_area, pixsad, npixels);
					}
				}
				else
				{
					y_poly.allocate_area_among_pixels(image_plane_area, pixels, npixels);
				}
			}
			
			if (threadIdx.x == 0 && threadIdx.y == 0)
			{
				unsigned long long int p = atomicAdd(percentage, 1);
				unsigned long long int imax = ((num_ray_threads.re - 1) / blockDim.x + 1);
				imax *= ((num_ray_threads.im - 1) / blockDim.y + 1);
				if (p * 100 / imax > (p - 1) * 100 / imax)
				{
					device_print_progress(verbose, p, imax);
				}
			}
		}
	}
}

