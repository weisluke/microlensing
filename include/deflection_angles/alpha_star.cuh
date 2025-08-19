#pragma once

#include "complex.cuh"
#include "star.cuh"
#include "tree_node.cuh"


namespace microlensing
{

/******************************************************************************
calculate the deflection angle within a node due to nearby stars

\param z -- complex image plane position
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param node -- pointer to node

\return alpha_star = theta^2 * sum(m_i / (z - z_i))_bar
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> alpha_star(Complex<T> z, T theta, star<T>* stars, TreeNode<T>* node)
{
	Complex<T> a_star_bar;

	/******************************************************************************
	theta^2 * sum(m_i / (z - z_i))
	******************************************************************************/
	for (int j = 0; j < node->num_neighbors; j++)
	{
		TreeNode<T>* neighbor = node->neighbors[j];
		for (int i = 0; i < neighbor->numstars; i++)
		{
			a_star_bar += stars[neighbor->stars + i].mass / (z - stars[neighbor->stars + i].position);
		}
	}
	for (int i = 0; i < node->numstars; i++)
	{
		a_star_bar += stars[node->stars + i].mass / (z - stars[node->stars + i].position);
	}
	a_star_bar *= theta * theta;

	return a_star_bar.conj();
}

/******************************************************************************
calculate the derivative of the deflection angle with respect to zbar within a
node due to nearby stars

\param z -- complex image plane position
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param node -- pointer to node

\return d_alpha_star_d_zbar = -theta^2 * sum(m_i / (z - z_i)^2)_bar
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> d_alpha_star_d_zbar(Complex<T> z, T theta, star<T>* stars, TreeNode<T>* node)
{
	Complex<T> d_a_star_bar_d_z;

	/******************************************************************************
	-theta^2 * sum(m_i / (z - z_i)^2)
	******************************************************************************/
	for (int j = 0; j < node->num_neighbors; j++)
	{
		TreeNode<T>* neighbor = node->neighbors[j];
		for (int i = 0; i < neighbor->numstars; i++)
		{
			d_a_star_bar_d_z += stars[neighbor->stars + i].mass / ((z - stars[neighbor->stars + i].position) * (z - stars[neighbor->stars + i].position));
		}
	}
	for (int i = 0; i < node->numstars; i++)
	{
		d_a_star_bar_d_z += stars[node->stars + i].mass / ((z - stars[node->stars + i].position) * (z - stars[node->stars + i].position));
	}
	d_a_star_bar_d_z *= -theta * theta;

	return d_a_star_bar_d_z.conj();
}

/******************************************************************************
calculate the second derivative of the deflection angle with respect to zbar^2
within a node due to nearby stars

\param z -- complex image plane position
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param node -- pointer to node

\return d2_alpha_star / d_zbar2 = 2 * theta^2 * sum(m_i / (z - z_i)^3)_bar
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> d2_alpha_star_d_zbar2(Complex<T> z, T theta, star<T>* stars, TreeNode<T>* node)
{
	Complex<T> d2_a_star_bar_d_z2;

	/******************************************************************************
	2 * theta^2 * sum(m_i / (z - z_i)^3)
	******************************************************************************/
	for (int j = 0; j < node->num_neighbors; j++)
	{
		TreeNode<T>* neighbor = node->neighbors[j];
		for (int i = 0; i < neighbor->numstars; i++)
		{
			d2_a_star_bar_d_z2 += stars[neighbor->stars + i].mass / 
				((z - stars[neighbor->stars + i].position) * (z - stars[neighbor->stars + i].position) * (z - stars[neighbor->stars + i].position));
		}
	}
	for (int i = 0; i < node->numstars; i++)
	{
		d2_a_star_bar_d_z2 += stars[node->stars + i].mass / 
			((z - stars[node->stars + i].position) * (z - stars[node->stars + i].position) * (z - stars[node->stars + i].position));
	}
	d2_a_star_bar_d_z2 *= 2 * theta * theta;

	return d2_a_star_bar_d_z2.conj();
}

}

