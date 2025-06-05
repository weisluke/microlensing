#pragma once

#include "deflection_angles/alpha_local.cuh"
#include "deflection_angles/alpha_macro.cuh"
#include "deflection_angles/alpha_smooth.cuh"
#include "deflection_angles/alpha_star.cuh"
#include "complex.cuh"
#include "star.cuh"
#include "tree_node.cuh"


namespace microlensing
{

/******************************************************************************
lens equation from image plane to source plane

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param node -- node within which to calculate the deflection angle
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if
                        approximate

\return w = z - a_macro - alpha_star - alpha_local - alpha_smooth
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> w(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
    Complex<T> a_macro = alpha_macro<T>(z, kappa, gamma);
	Complex<T> a_star = alpha_star<T>(z, theta, stars, node);
	Complex<T> a_local = alpha_local<T>(z, theta, node);
	Complex<T> a_smooth = alpha_smooth<T>(z, kappastar, rectangular, corner, approx, taylor_smooth);

	/******************************************************************************
	z - a_macro - alpha_star - alpha_local - alpha_smooth
	******************************************************************************/
	return z - a_macro - a_star - a_local - a_smooth;
}

/******************************************************************************
derivative of the lens equation with respect to z

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not

\return d_w_d_z
******************************************************************************/
template <typename T>
__host__ __device__ T d_w_d_z(Complex<T> z, T kappa, T gamma, T kappastar, int rectangular, Complex<T> corner, int approx)
{
    T d_a_macro_d_z = d_alpha_macro_d_z<T>(z, kappa, gamma);
	T d_a_smooth_d_z = d_alpha_smooth_d_z<T>(z, kappastar, rectangular, corner, approx);

    return 1 - d_a_macro_d_z - d_a_smooth_d_z;
}

/******************************************************************************
derivative of the lens equation with respect to zbar

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param node -- node within which to calculate the deflection angle
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if
                        approximate

\return w = z - a_macro - alpha_star - alpha_local - alpha_smooth
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> d_w_d_zbar(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
    T d_a_macro_d_zbar = d_alpha_macro_d_zbar<T>(z, kappa, gamma);
	Complex<T> d_a_star_d_zbar = d_alpha_star_d_zbar<T>(z, theta, stars, node);
	Complex<T> d_a_local_d_zbar = d_alpha_local_d_zbar<T>(z, theta, node);
	Complex<T> d_a_smooth_d_zbar = d_alpha_smooth_d_zbar<T>(z, kappastar, rectangular, corner, approx, taylor_smooth);

	return -d_a_macro_d_zbar - d_a_star_d_zbar - d_a_local_d_zbar - d_a_smooth_d_zbar;
}

/******************************************************************************
second derivative of the lens equation with respect to zbar^2

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param node -- node within which to calculate the deflection angle
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if
                        approximate

\return w = z - a_macro - alpha_star - alpha_local - alpha_smooth
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> d2_w_d_zbar2(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	Complex<T> d2_a_star_d_zbar2 = d2_alpha_star_d_zbar2<T>(z, theta, stars, node);
	Complex<T> d2_a_local_d_zbar2 = d2_alpha_local_d_zbar2<T>(z, theta, node);
	Complex<T> d2_a_smooth_d_zbar2 = d2_alpha_smooth_d_zbar2<T>(z, kappastar, rectangular, corner, approx, taylor_smooth);

	return -d2_a_star_d_zbar2 - d2_a_local_d_zbar2 - d2_a_smooth_d_zbar2;
}

/******************************************************************************
magnification at a point in the image plane

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param node -- node within which to calculate the deflection angle
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if
						approximate

\return mu = ( (dw / dz)^2 - dw/dz * (dw/dz)bar ) ^ -1
******************************************************************************/
template <typename T>
__device__ T magnification(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
    T a = d_w_d_z<T>(z, kappa, gamma, kappastar, rectangular, corner, approx);
    Complex<T> b = d_w_d_zbar<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);

	T mu_inv = a * a - b.abs() * b.abs();

	return 1 / mu_inv;
}

}