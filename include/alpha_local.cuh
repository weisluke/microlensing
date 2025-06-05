#pragma once

#include "complex.cuh"
#include "tree_node.cuh"


/******************************************************************************
calculate the deflection angle within a node due to far away stars

\param z -- complex image plane position
\param theta -- size of the Einstein radius of a unit mass point lens
\param node -- pointer to node

\return alpha_local = theta^2 * sum(i * a_i * (z - z_0) ^ (i - 1))_bar
           where a_i are coefficients of the lensing potential in units of the
           node size
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> alpha_local(Complex<T> z, T theta, TreeNode<T>* node)
{
	Complex<T> a_local_bar;
	Complex<T> dz = (z - node->center) / node->half_length;

	for (int i = node->expansion_order; i >= 1; i--)
	{
		a_local_bar *= dz;
		a_local_bar += node->local_coeffs[i] * i;
	}
	a_local_bar *= theta * theta;
	/******************************************************************************
	account for node size 
	******************************************************************************/
	a_local_bar /= node->half_length;

	return a_local_bar.conj();
}

/******************************************************************************
calculate the derivative of the deflection angle with respect to zbar within a
node due to far away stars

\param z -- complex image plane position
\param theta -- size of the Einstein radius of a unit mass point lens
\param node -- pointer to node

\return d_alpha_local_d_zbar = theta^2 
 		   * sum(i * (i-1) * a_i * (z - z_0) ^ (i - 2))_bar
		   where a_i are coefficients of the lensing potential in units of the
		   node size
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> d_alpha_local_d_zbar(Complex<T> z, T theta, TreeNode<T>* node)
{
	Complex<T> d_a_local_bar_dz;
	Complex<T> dz = (z - node->center) / node->half_length;

	for (int i = node->expansion_order; i >= 2; i--)
	{
		d_a_local_bar_dz *= dz;
		d_a_local_bar_dz += node->local_coeffs[i] * i * (i - 1);
	}
	d_a_local_bar_dz *= theta * theta;
	/******************************************************************************
	account for node size
	******************************************************************************/
	d_a_local_bar_dz /= node->half_length * node->half_length;

	return d_a_local_bar_dz.conj();
}

/******************************************************************************
calculate the second derivative of the deflection angle with respect to zbar^2
within a node due to far away stars

\param z -- complex image plane position
\param theta -- size of the Einstein radius of a unit mass point lens
\param node -- pointer to node

\return d2_alpha_local_d_zbar2 = theta^2 
 		   * sum(i * (i-1) * (i - 2) * a_i * (z - z_0) ^ (i - 3))_bar
		   where a_i are coefficients of the lensing potential in units of the
		   node size
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> d2_alpha_local_d_zbar2(Complex<T> z, T theta, TreeNode<T>* node)
{
	Complex<T> d2_a_local_bar_dz2;
	Complex<T> dz = (z - node->center) / node->half_length;

	for (int i = node->expansion_order; i >= 3; i--)
	{
		d2_a_local_bar_dz2 *= dz;
		d2_a_local_bar_dz2 += node->local_coeffs[i] * i * (i - 1) * (i - 2);
	}
	d2_a_local_bar_dz2 *= theta * theta;
	/******************************************************************************
	account for node size
	******************************************************************************/
	d2_a_local_bar_dz2 /= node->half_length * node->half_length * node->half_length;

	return d2_a_local_bar_dz2.conj();
}

