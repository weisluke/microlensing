#pragma once

#include "complex.cuh"
#include "lens_equations.cuh"
#include "star.cuh"
#include "tree_node.cuh"
#include "util/math_util.cuh"
#include "util/util.cuh"


/******************************************************************************
determine whether a position is close to a star

\param z -- complex image plane position
\param stars -- pointer to array of point mass lenses
\param root -- pointer to root node
\param dz -- distance to check whether a star is within
******************************************************************************/
template <typename T>
__host__ __device__ bool is_near_star(Complex<T> z, star<T>* stars, TreeNode<T>* root, Complex<T> dz)
{
    TreeNode<T>* node = treenode::get_nearest_node(z, root);

    for (int i = 0; i < node->numstars; i++)
    {
        if ((z - stars[node->stars + i].position).abs() < dz.abs())
        {
            return true;
        }
    }
    for (int i = 0; i < node->num_neighbors; i++)
    {
        TreeNode<T>* neighbor = node->neighbors[i];
        for (int j = 0; j < neighbor->numstars; j++)
        {
            if ((z - stars[neighbor->stars + j].position).abs() < dz.abs())
            {
                return true;
            }
        }
    }
    return false;
}

/******************************************************************************
get the index of the nearest star to a given position

\param z -- complex image plane position
\param stars -- pointer to array of point mass lenses
\param root -- pointer to root node
******************************************************************************/
template <typename T>
__host__ __device__ int get_nearest_star(Complex<T> z, star<T>* stars, TreeNode<T>* root)
{
    int res = -1;
    Complex<T> min_dz = Complex<T>(root->half_length, root->half_length);
    Complex<T> dz;

    TreeNode<T>* node = treenode::get_nearest_node(z, root);

    for (int i = 0; i < node->numstars; i++)
    {
        dz = (z - stars[node->stars + i].position);
        if (dz.abs() < min_dz.abs())
        {
            min_dz = dz;
            res = node->stars + i;
        }
    }
    for (int i = 0; i < node->num_neighbors; i++)
    {
        TreeNode<T>* neighbor = node->neighbors[i];
        for (int j = 0; j < neighbor->numstars; j++)
        {
            dz = (z - stars[neighbor->stars + j].position);
            if (dz.abs() < min_dz.abs())
            {
                min_dz = dz;
                res = neighbor->stars + j;
            }
        }
    }
    return res;
}

/******************************************************************************
parametric image line equation

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
\param w0 -- complex source plane position
\param v -- complex source velocity
******************************************************************************/
template <typename T>
__host__ __device__ T parametric_image_line(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
    int rectangular, Complex<T> corner, int approx, int taylor_smooth, Complex<T> w0, Complex<T> v)
{
	//this function return value is equivalent to
	//Complex<T>(0, 1) / 2 * (v * (w - w0).conj() - (w - w0) * v.conj())

    Complex<T> w = microlensing::w<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
    
	v *= Complex<T>(0, 1); //rotate velocity vector
	Complex<T> res = (w - w0) * v.conj(); //dot product of the complex vectors
	return re.re; 
}

/******************************************************************************
derivative of the parametric image line equation with respect to z

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
\param w0 -- complex source plane position
\param v -- complex source velocity
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> d_parametric_image_line_d_z(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
    int rectangular, Complex<T> corner, int approx, int taylor_smooth, Complex<T> w0, Complex<T> v)
{
    T d_w_d_z = microlensing::d_w_d_z<T>(z, kappa, gamma, kappastar, rectangular, corner, approx);
    Complex<T> d_w_d_zbar = microlensing::d_w_d_zbar<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
    
    return Complex<T>(0, 1) / 2 * (v * d_w_d_zbar.conj() - v.conj() * d_w_d_z);
}

/******************************************************************************
derivative of the parametric image line equation with respect to zbar

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
\param w0 -- complex source plane position
\param v -- complex source velocity
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> d_parametric_image_line_d_zbar(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
    int rectangular, Complex<T> corner, int approx, int taylor_smooth, Complex<T> w0, Complex<T> v)
{
    T d_w_d_z = microlensing::d_w_d_z<T>(z, kappa, gamma, kappastar, rectangular, corner, approx);
    Complex<T> d_w_d_zbar = microlensing::d_w_d_zbar<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
    
    return Complex<T>(0, 1) / 2 * (v * d_w_d_z - v.conj() * d_w_d_zbar);
}

/******************************************************************************
generate a first guess for an updated image position

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
\param w0 -- complex source plane position
\param v -- complex source velocity
\param dt -- time step size
\param min_dz -- minimum allowed dz
\param max_dz -- maximum allowed dz

\return z_new -- updated value of the root z
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> step_tangent(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
    int rectangular, Complex<T> corner, int approx, int taylor_smooth, Complex<T> w0, Complex<T> v, T& dt, T min_dz, T max_dz)
{
    TreeNode<T>* node = treenode::get_nearest_node(z, root);

    Complex<T> dfdzbar = d_parametric_image_line_d_zbar<T>(z, kappa, gamma, theta, stars, kappastar, node, 
		rectangular, corner, approx, taylor_smooth, w0, v);
    Complex<T> dzdt = Complex<T>(0, -2) * dfdzbar;
    Complex<T> dz = dzdt * dt;

    if (dz.abs() > max_dz)
    {
        dt *= max_dz / dz.abs();
        dz = dzdt * dt;
    }
    else if (dz.abs() < min_dz)
    {
        dt *= min_dz / dz.abs();
        dz = dzdt * dt;
    }

    //macro-magnification changes dzdt
    return z + dz * sgn(microlensing::macro_mu(z, kappa, gamma));
}

/******************************************************************************
find an updated image position for a new source position and previous image
position

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
\param w0 -- complex source plane position
\param v -- complex source velocity

\return z_new -- updated value of the root z
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> find_root(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
    int rectangular, Complex<T> corner, int approx, int taylor_smooth, Complex<T> w0, Complex<T> v)
{
    TreeNode<T>* node;
    T f;
    Complex<T> dfdz;
    Complex<T> dz;

    int MAX_NUM_ITERS = 100;
    for (int i = 0; i < MAX_NUM_ITERS; i++)
    {
        node = treenode::get_nearest_node(z, root);

        f = parametric_image_line<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, w0, v);

        /******************************************************************************
        if mapping is within 10^-9 * theta_star of desired source track, return
        ******************************************************************************/
        if (f.abs() / theta < static_cast<T>(0.000000001))
        {
            return z;
        }

        dfdz = d_parametric_image_line_d_z<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, w0, v);
        dz = f / (2 * dfdz);
        
        //don't iterate needlessly if z is changing by a tiny amount
        if (dz.abs() / theta < static_cast<T>(0.000000001))
        {
            return z;
        }

        z -= dz;
    }
    return z;
}

/******************************************************************************
macro parametric image line equation

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param w0 -- complex source plane position
\param v -- complex source velocity
******************************************************************************/
template <typename T>
__host__ __device__ T macro_parametric_image_line(Complex<T> z, T kappa, T gamma, Complex<T> w0, Complex<T> v)
{
    Complex<T> w = microlensing::macro_w<T>(z, kappa, gamma);
	v *= Complex<T>(0, 1);

	Complex<T> res = (w - w0) * v.conj();
	return res.re;
}

/******************************************************************************
determine which stars are close enough to the macroimage line to use

\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses
\param kappa -- total convergence
\param gamma -- external shear
\param w0 -- complex source plane position
\param v -- complex source velocity
\param max_r -- maximum allowed distance from macro-line
\param use_star -- pointer to whether or not stars should be used
******************************************************************************/
template <typename T>
__global__ void use_star_kernel(star<T>* stars, int nstars, T kappa, T gamma, Complex<T> w0, Complex<T> v, T max_r, int* use_star)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x_stride = blockDim.x * gridDim.x;

    for (int i = x_index; i < nstars; i += x_stride)
    {       
        T f = macro_parametric_image_line(stars[i].position, kappa, gamma, w0, v);

        if (fabs(f) > max_r)
        {
            use_star[i] = 0;
        }
    }
}

/******************************************************************************
map images to source plane

\param z -- pointer to array of image positions
\param nimages -- number of images in array
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
\param w -- pointer to array of source positions
******************************************************************************/
template <typename T>
__global__ void image_to_source_kernel(Complex<T>* z, int nimages, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
    int rectangular, Complex<T> corner, int approx, int taylor_smooth, Complex<T>* w)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x_stride = blockDim.x * gridDim.x;

    for (int i = x_index; i < nimages; i += x_stride)
    {
        TreeNode<T>* node = treenode::get_nearest_node(z[i], root);

        /******************************************************************************
        map image plane positions to source plane positions
        ******************************************************************************/
        w[i] = microlensing::w<T>(z[i], kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
    }
}

/******************************************************************************
calculate magnifications

\param z -- pointer to array of image positions
\param nimages -- number of images in array
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
\param mu -- pointer to array of magnifications
******************************************************************************/
template <typename T>
__global__ void magnifications_kernel(Complex<T>* z, int nimages, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
    int rectangular, Complex<T> corner, int approx, int taylor_smooth, T* mu)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x_stride = blockDim.x * gridDim.x;

    for (int i = x_index; i < nimages; i += x_stride)
    {
        TreeNode<T>* node = treenode::get_nearest_node(z[i], root);

        /******************************************************************************
        map image plane positions to source plane positions
        ******************************************************************************/
        mu[i] = microlensing::mu<T>(z[i], kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
    }
}

/******************************************************************************
find the image position for a given source position

\param z -- complex image plane position initial guess
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
\param w0 -- complex source plane position
\param v -- complex source velocity (arbitrary)

\return z_new -- updated value of the image position
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> find_point_image(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
    int rectangular, Complex<T> corner, int approx, int taylor_smooth, Complex<T> w0, Complex<T> v)
{
    TreeNode<T>* node;

    T f1, f2;
	Complex<T> df1dz, df1dzbar, df2dz, df2dzbar, dz;
    Complex<T> v1 = v;
    Complex<T> v2 = v * Complex<T>(0, 1);

    int MAX_NUM_ITERS = 100;
    for (int i = 0; i < MAX_NUM_ITERS; i++)
    {
        node = treenode::get_nearest_node(z, root);

        f1 = parametric_image_line<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, w0, v1);
        f2 = parametric_image_line<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, w0, v2);

        /******************************************************************************
        if mapping is within 10^-9 * theta_star of desired source track, return
        ******************************************************************************/
        if (f1.abs() / theta < static_cast<T>(0.000000001) && f2.abs() / theta < static_cast<T>(0.000000001))
        {
            return z;
        }

        df1dz = d_parametric_image_line_d_z<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, w0, v1);
        df1dzbar = d_parametric_image_line_d_zbar<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, w0, v1);
        df2dz = d_parametric_image_line_d_z<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, w0, v2);
        df2dzbar = d_parametric_image_line_d_zbar<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, w0, v2);

        dz = (df2dzbar * f1 - df1dzbar * f2) / (df1dz * df2dzbar - df1dzbar * df2dz);
        
        //don't iterate needlessly if z is changing by a tiny amount
        if (dz.abs() / theta < static_cast<T>(0.000000001))
        {
            return z;
        }

        z -= dz;
    }
    return z;
}

/******************************************************************************
find an image position for the given source track that also lies on a critical
curve

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
\param w0 -- complex source plane position
\param v -- complex source velocity

\return z_new -- updated value of the image position
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> find_critical_curve_image(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
    int rectangular, Complex<T> corner, int approx, int taylor_smooth, Complex<T> w0, Complex<T> v)
{
    TreeNode<T>* node;

    T f, inv_mu;
	Complex<T> dfdz, dfdzbar, dinvmudz, dinvmudzbar, dz;

    int MAX_NUM_ITERS = 100;
    for (int i = 0; i < MAX_NUM_ITERS; i++)
    {
        node = treenode::get_nearest_node(z, root);

        f = parametric_image_line<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, w0, v);
        mu = microlensing::inv_mu<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);

        /******************************************************************************
        if mapping is within 10^-9 * theta_star of desired source track, return
        ******************************************************************************/
        if (f.abs() / theta < static_cast<T>(0.000000001) && std::abs(mu) < static_cast<T>(0.000000001))
        {
            return z;
        }

        dfdz = d_parametric_image_line_d_z<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, w0, v);
        dfdzbar = d_parametric_image_line_d_zbar<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, w0, v);

        dinvmudz = microlensing::d_inv_mu_d_z<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
        dinvmudzbar = microlensing::d_inv_mu_d_zbar<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);

        dz = (dinvmudzbar * f - dfdzbar * invmu) / (dfdz * dinvmudzbar - dfdzbar * dinvmudz);
        
        //don't iterate needlessly if z is changing by a tiny amount
        if (dz.abs() / theta < static_cast<T>(0.000000001))
        {
            return z;
        }

        z -= dz;
    }
    return z;
}

