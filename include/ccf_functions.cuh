#pragma once

#include "alpha_local.cuh"
#include "alpha_smooth.cuh"
#include "alpha_star.cuh"
#include "complex.cuh"
#include "star.cuh"
#include "tree_node.cuh"

#include <numbers>


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

\return w = (1 - kappa) * z + gamma * z_bar 
            - alpha_star - alpha_local - alpha_smooth
******************************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	Complex<T> a_star = alpha_star<T>(z, theta, stars, node);
	Complex<T> a_local = alpha_local<T>(z, theta, node);
	Complex<T> a_smooth = alpha_smooth<T>(z, kappastar, rectangular, corner, approx, taylor_smooth);

	/******************************************************************************
	(1 - kappa) * z + gamma * z_bar - alpha_star - alpha_local - alpha_smooth
	******************************************************************************/
	return (1 - kappa) * z + gamma * z.conj() - a_star - a_local - a_smooth;
}

/******************************************************************************
parametric critical curve equation for a star field
we seek the values of z that make this equation equal to 0 for a given phi

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
\param phi -- value of the variable parametrizing z

\return gamma - (d_alpha_star / d_zbar)_bar - (d_alpha_smooth / d_zbar)_bar
		- (1 - kappa - d_alpha_smooth / dz) * e^(-i * phi)
******************************************************************************/
template <typename T>
__device__ Complex<T> parametric_critical_curve(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth, T phi)
{
	Complex<T> d_a_star_d_zbar = d_alpha_star_d_zbar(z, theta, stars, node);
	Complex<T> d_a_local_d_zbar = d_alpha_local_d_zbar(z, theta, node);
	T d_a_smooth_d_z = d_alpha_smooth_d_z(z, kappastar, rectangular, corner, approx);
	Complex<T> d_a_smooth_d_zbar = d_alpha_smooth_d_zbar(z, kappastar, rectangular, corner, approx, taylor_smooth);

	/******************************************************************************
	gamma - (d_alpha_star / d_zbar)_bar - (d_alpha_local / d_zbar)_bar
	- (d_alpha_smooth / d_zbar)_bar
	- (1 - kappa - d_alpha_smooth / d_z) * e^(-i * phi)
	******************************************************************************/
	return gamma - d_a_star_d_zbar.conj() - d_a_local_d_zbar.conj()
		- d_a_smooth_d_zbar.conj()
		- (1 - kappa - d_a_smooth_d_z) * Complex<T>(cos(phi), -sin(phi));
}

/******************************************************************************
derivative of the parametric critical curve equation with respect to z

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

\return -2 * theta^2 * sum(m_i / (z - z_i)^3)
		- (d^2alpha_smooth / dz_bar^2)_bar
******************************************************************************/
template <typename T>
__device__ Complex<T> d_parametric_critical_curve_dz(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	Complex<T> d2_a_star_d_zbar2 = d2_alpha_star_d_zbar2(z, theta, stars, node);
	Complex<T> d2_a_local_d_zbar2 = d2_alpha_local_d_zbar2(z, theta, node);
	Complex<T> d2_a_smooth_d_zbar2 = d2_alpha_smooth_d_zbar2(z, kappastar, rectangular, corner, approx, taylor_smooth);

	/******************************************************************************
	-(d2_alpha_star / d_zbar2)_bar - (d2_alpha_local / d_zbar2)_bar
	- (d2_alpha_smooth / d_zbar2)_bar
	******************************************************************************/
	return -d2_a_star_d_zbar2.conj() - d2_a_local_d_zbar2.conj() - d2_a_smooth_d_zbar2.conj();
}

/******************************************************************************
derivative of the parametric critical curve equation with respect to phi

\param z -- complex image plane position
\param kappa -- total convergence
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param phi -- value of the variable parametrizing z

\return (1 - kappa - d_alpha_smooth / d_z) * i * e^(-i * phi)
******************************************************************************/
template <typename T>
__device__ Complex<T> d_parametric_critical_curve_dphi(Complex<T> z, T kappa, T kappastar,
	int rectangular, Complex<T> corner, int approx, T phi)
{
	T d_a_smooth_d_z = d_alpha_smooth_d_z(z, kappastar, rectangular, corner, approx);

	/******************************************************************************
	(1 - kappa - d_alpha_smooth / d_z) * i * e^(-i * phi)
	******************************************************************************/
	return Complex<T>(0, 1 - kappa - d_a_smooth_d_z) * Complex<T>(0, -phi).exp();
}

/******************************************************************************
magnification length scale for a given location on the critical curve

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

\return the magnification length scale d_0 in the approximation
        mu = sqrt(d_0 / d) for the magnification of an individual microimage
        some distance d perpendicular to a caustic
******************************************************************************/
template <typename T>
__device__ T mu_length_scale(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	Complex<T> d_a_star_d_zbar = d_alpha_star_d_zbar(z, theta, stars, node);
	Complex<T> d_a_local_d_zbar = d_alpha_local_d_zbar(z, theta, node);
	T d_a_smooth_d_z = d_alpha_smooth_d_z(z, kappastar, rectangular, corner, approx);
	Complex<T> d_a_smooth_d_zbar = d_alpha_smooth_d_zbar(z, kappastar, rectangular, corner, approx, taylor_smooth);

	T d_w_d_z = 1 - kappa - d_a_smooth_d_z;
	Complex<T> d_w_d_zbar = gamma - d_a_star_d_zbar - d_a_local_d_zbar - d_a_smooth_d_zbar;

	Complex<T> d2_a_star_d_zbar2 = d2_alpha_star_d_zbar2(z, theta, stars, node);
	Complex<T> d2_a_local_d_zbar2 = d2_alpha_local_d_zbar2(z, theta, node);
	Complex<T> d2_a_smooth_d_zbar2 = d2_alpha_smooth_d_zbar2(z, kappastar, rectangular, corner, approx, taylor_smooth);

	Complex<T> d2_w_d_zbar2 = -d2_a_star_d_zbar2 - d2_a_local_d_zbar2 - d2_a_smooth_d_zbar2;

	Complex<T> critical_curve_tangent = Complex<T>(0, -2) * d_w_d_zbar.conj() * d2_w_d_zbar2;
	Complex<T> caustic_tangent = d_w_d_z * critical_curve_tangent + d_w_d_zbar * critical_curve_tangent.conj();

	return 1 / (2 * caustic_tangent.abs());
}

/******************************************************************************
find an updated approximation for a particular critical curve
root given the current approximation z and all other roots

\param k -- index of z within the roots array
			0 <= k < nroots
\param z -- complex image plane position
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
\param phi -- value of the variable parametrizing z
\param roots -- pointer to array of roots
\param nroots -- number of roots in array

\return z_new -- updated value of the root z
******************************************************************************/
template <typename T>
__device__ Complex<T> find_critical_curve_root(int k, Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
	int rectangular, Complex<T> corner, int approx, int taylor_smooth, T phi, Complex<T>* roots, int nroots)
{
	TreeNode<T>* node = treenode::get_nearest_node(z, root);

	Complex<T> f0 = parametric_critical_curve(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, phi);
	T d_a_smooth_d_z = d_alpha_smooth_d_z(z, kappastar, rectangular, corner, approx);

	/******************************************************************************
	if 1/mu < 10^-9, return same position
	the value of 1/mu depends on the value of f0
	this check ensures that the maximum possible value of 1/mu is less than desired
	******************************************************************************/
	if (fabs(f0.abs() * (f0.abs() + 2 * (1 - kappa - d_a_smooth_d_z))) < static_cast<T>(0.000000001) &&
		fabs(f0.abs() * (f0.abs() - 2 * (1 - kappa - d_a_smooth_d_z))) < static_cast<T>(0.000000001))
	{
		return z;
	}

	Complex<T> f1 = d_parametric_critical_curve_dz(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);

	/******************************************************************************
	contribution due to distance between root and poles
	******************************************************************************/
	Complex<T> p_sum;
	for (int i = 0; i < root->numstars; i++)
	{
		p_sum += 1 / (z - stars[root->stars + i].position);
	}
	if (!rectangular && !approx)
	{
		p_sum += 1 / z;
	}
	p_sum *= 2;

	/******************************************************************************
	contribution due to distance between root and other roots
	******************************************************************************/
	Complex<T> root_sum;
	for (int i = 0; i < nroots; i++)
	{
		if (i != k)
		{
			root_sum += 1 / (z - roots[i]);
		}
	}

	Complex<T> result = f1 + f0 * (p_sum - root_sum);
	return z - f0 / result;
}

/******************************************************************************
take the list of all roots z, the given value of j out of nphi steps, and the
number of branches, and set the initial roots for step j equal to the final
roots of step j-1
reset values for whether roots have all been found to sufficient accuracy to
false

\param z -- pointer to array of root positions
\param nroots -- number of roots
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi]
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param fin -- pointer to array of boolean values for whether roots have been
			  found to sufficient accuracy
			  array is of size nbranches * 2 * nroots
******************************************************************************/
template <typename T>
__global__ void prepare_roots_kernel(Complex<T>* z, int nroots, int j, int nphi, int nbranches, bool* fin)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	int z_index = blockIdx.z * blockDim.z + threadIdx.z;
	int z_stride = blockDim.z * gridDim.z;

	for (int c = z_index; c < nbranches; c += z_stride)
	{
		for (int b = y_index; b < 2; b += y_stride)
		{
			for (int a = x_index; a < nroots; a += x_stride)
			{
				int center = (nphi / (2 * nbranches) + c * nphi / nbranches + c) * nroots;

				if (b == 0)
				{

					z[center + j * nroots + a] = z[center + (j - 1) * nroots + a];
					fin[c * 2 * nroots + a] = false;
				}
				else
				{
					z[center - j * nroots + a] = z[center - (j - 1) * nroots + a];
					fin[c * 2 * nroots + a + nroots] = false;
				}
			}
		}
	}
}

/******************************************************************************
find new critical curve roots for a star field

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
\param roots -- pointer to array of roots
\param nroots -- number of roots in array
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi]
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param fin -- pointer to array of boolean values for whether roots have been
			  found to sufficient accuracy
			  array is of size nbranches * 2 * nroots
******************************************************************************/
template <typename T>
__global__ void find_critical_curve_roots_kernel(T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth, Complex<T>* roots, int nroots, int j, int nphi, int nbranches, bool* fin)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	int z_index = blockIdx.z * blockDim.z + threadIdx.z;
	int z_stride = blockDim.z * gridDim.z;

	Complex<T> result;
	T norm;
	int sgn;

	T dphi = 2 * std::numbers::pi_v<T> / nphi * j;

	for (int c = z_index; c < nbranches; c += z_stride)
	{
		for (int b = y_index; b < 2; b += y_stride)
		{
			T phi0 = std::numbers::pi_v<T> / nbranches + c * 2 * std::numbers::pi_v<T> / nbranches;

			for (int a = x_index; a < nroots; a += x_stride)
			{
				/******************************************************************************
				we use the following variable to determine whether we are on the positive or
				negative side of phi0, as we are simultaneously growing 2 sets of roots after
				having stepped away from the middle by j out of nphi steps
				******************************************************************************/
				sgn = (b == 0 ? -1 : 1);

				/******************************************************************************
				if root has not already been calculated to desired precision
				we are calculating nbranches * 2 * nroots roots in parallel, so
				" c * 2 * nroots " indicates what branch we are in,
				" b * nroots " indicates whether we are on the positive or negative side, and
				" a " indicates the particular root position
				******************************************************************************/
				if (!fin[c * 2 * nroots + b * nroots + a])
				{
					/******************************************************************************
					calculate new root
					center of the roots array (ie the index of phi0) for all branches is
					( nphi / (2 * nbranches) + c  * nphi / nbranches + c) * nroots
					for the particular value of phi here (i.e., phi0 +/- dphi), roots start
					at +/- j*nroots of that center
					a is then added to get the final index of this particular root
					******************************************************************************/

					int center = (nphi / (2 * nbranches) + c * nphi / nbranches + c) * nroots;
					result = find_critical_curve_root(a, roots[center + sgn * j * nroots + a], kappa, gamma, theta, stars, kappastar, root, rectangular, corner, approx, taylor_smooth, phi0 + sgn * dphi, &(roots[center + sgn * j * nroots]), nroots);

					/******************************************************************************
					distance between old root and new root in units of theta_star
					******************************************************************************/
					norm = (result - roots[center + sgn * j * nroots + a]).abs() / theta;

					/******************************************************************************
					compare position to previous value, if less than desired precision of 10^-9,
					set fin[root] to true
					******************************************************************************/
					if (norm < static_cast<T>(0.000000001))
					{
						fin[c * 2 * nroots + b * nroots + a] = true;
					}
					roots[center + sgn * j * nroots + a] = result;
				}
			}
		}
	}
}

/******************************************************************************
find maximum error in critical curve roots for a star field

\param z -- pointer to array of roots
\param nroots -- number of roots in array
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
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param errs -- pointer to array of errors
			   array is of size nbranches * 2 * nroots
******************************************************************************/
template <typename T>
__global__ void find_errors_kernel(Complex<T>* z, int nroots, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
	int rectangular, Complex<T> corner, int approx, int taylor_smooth, int j, int nphi, int nbranches, T* errs)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	int z_index = blockIdx.z * blockDim.z + threadIdx.z;
	int z_stride = blockDim.z * gridDim.z;

	int sgn;

	T dphi = 2 * std::numbers::pi_v<T> / nphi * j;

	for (int c = z_index; c < nbranches; c += z_stride)
	{
		for (int b = y_index; b < 2; b += y_stride)
		{
			T phi0 = std::numbers::pi_v<T> / nbranches + c * 2 * std::numbers::pi_v<T> / nbranches;

			for (int a = x_index; a < nroots; a += x_stride)
			{
				sgn = (b == 0 ? -1 : 1);

				int center = (nphi / (2 * nbranches) + c * nphi / nbranches + c) * nroots;

				TreeNode<T>* node = treenode::get_nearest_node(z[center + sgn * j * nroots + a], root);

				/******************************************************************************
				the value of 1/mu depends on the value of f0
				this calculation ensures that the maximum possible value of 1/mu is given
				******************************************************************************/
				Complex<T> f0 = parametric_critical_curve(z[center + sgn * j * nroots + a], kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, phi0 + sgn * dphi);
				T d_a_smooth_d_z = d_alpha_smooth_d_z(z[center + sgn * j * nroots + a], kappastar, rectangular, corner, approx);

				T e1 = fabs(f0.abs() * (f0.abs() + 2 * (1 - kappa - d_a_smooth_d_z)));
				T e2 = fabs(f0.abs() * (f0.abs() - 2 * (1 - kappa - d_a_smooth_d_z)));

				/******************************************************************************
				return maximum possible error in 1/mu at root position
				******************************************************************************/
				errs[center + sgn * j * nroots + a] = fmax(e1, e2);
			}
		}
	}
}

/******************************************************************************
determine whether errors have nan values

\param errs -- pointer to array of errors
\param nerrs -- number of errors in array
\param hasnan -- pointer to int (bool) of whether array has nan values or not
******************************************************************************/
template <typename T>
__global__ void has_nan_err_kernel(T* errs, int nerrs, int* hasnan)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int a = x_index; a < nerrs; a += x_stride)
	{
		if (!isfinite(errs[a]))
		{
			atomicExch(hasnan, 1);
		}
	}
}

/******************************************************************************
find caustics from critical curves for a star field

\param z -- pointer to array of roots
\param nroots -- number of roots in array
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
\param w -- pointer to array of caustic positions
******************************************************************************/
template <typename T>
__global__ void find_caustics_kernel(Complex<T>* z, int nroots, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
	int rectangular, Complex<T> corner, int approx, int taylor_smooth, Complex<T>* w)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int a = x_index; a < nroots; a += x_stride)
	{
		TreeNode<T>* node = treenode::get_nearest_node(z[a], root);

		/******************************************************************************
		map image plane positions to source plane positions
		******************************************************************************/
		w[a] = complex_image_to_source(z[a], kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
	}
}

/******************************************************************************
find magnification length scales from critical curves for a star field

\param z -- pointer to array of roots
\param nroots -- number of roots in array
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
\param d -- pointer to array of mu length scales
******************************************************************************/
template <typename T>
__global__ void find_mu_length_scales_kernel(Complex<T>* z, int nroots, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
	int rectangular, Complex<T> corner, int approx, int taylor_smooth, T* d)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int a = x_index; a < nroots; a += x_stride)
	{
		TreeNode<T>* node = treenode::get_nearest_node(z[a], root);

		/******************************************************************************
		calculate caustic strengths
		******************************************************************************/
		d[a] = mu_length_scale(z[a], kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
	}
}

