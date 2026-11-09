#pragma once

#include "binomial_coefficients.cuh"
#include "complex.cuh"
#include "star.cuh"
#include "tree_node.cuh"


namespace fmm
{

    /******************************************************************************
    calculate the multipole coefficient for a node for a given power and a given
    star field

    \param node -- pointer to node
    \param power -- what order to find the multipole coefficient of
    \param stars -- pointer to array of point mass lenses
    \param coeff -- pointer to where to store multipole coefficient
    ******************************************************************************/
    template <typename T>
    __device__ void calculate_multipole_coeff(TreeNode<T>* node, int power, star<T>* stars, Complex<T>* coeff)
    {
        Complex<T> result;

        if (power == 0)
        {
            /******************************************************************************
            a_0 = sum(m_i)
            ******************************************************************************/
            for (int i = 0; i < node->numstars; i++)
            {
                result += stars[node->stars + i].mass;
            }
        }
        else
        {
            /******************************************************************************
            a_k = sum( -m_i * ((z_i - node_center) / node_halflength)^k / k )
            ******************************************************************************/
            for (int i = 0; i < node->numstars; i++)
            {
                result -= ((stars[node->stars + i].position - node->center) / node->half_length).pow(power) * stars[node->stars + i].mass;
            }
            result /= power;
        }

        *coeff = result;
    }

    /******************************************************************************
    calculate the multipole coefficients for a given maximum power

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param power -- maximum expansion order to find the multipole coefficients of
    \param stars -- pointer to array of point mass lenses
    ******************************************************************************/
    template <typename T>
    __global__ void calculate_multipole_coeffs_kernel(TreeNode<T>* nodes, int numnodes, int power, star<T>* stars)
    {
        /******************************************************************************
        each thread is a node in the y thread direction
        ******************************************************************************/
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < numnodes; j += blockDim.y * gridDim.y)
        {
            TreeNode<T>* node = &nodes[j];

            if (node->num_children != 0 || node->numstars == 0)
            {
                continue;
            }

            /******************************************************************************
            each thread calculates a multipole coefficient in the x thread direction
            ******************************************************************************/
            for (int i = threadIdx.x; i <= power; i += blockDim.x)
            {
                calculate_multipole_coeff<T>(node, i, stars, &node->multipole_coeffs[i]);
            }
        }
    }

    /******************************************************************************
    calculate the shifted multipole coefficient for a node for a given power
    assumes that the expansion is being shifted to the center of the parent node

    \param node -- pointer to node
    \param power -- what order to find the multipole coefficient of
    \param binomcoeffs -- pointer to array of binomial coefficients
    \param coeff -- pointer to where to store multipole coefficient
    ******************************************************************************/
    template <typename T>
    __device__ void calculate_M2M_coeff(TreeNode<T>* node, int power, long long* binomcoeffs, Complex<T>* coeff)
    {
        Complex<T> result;

        if (power == 0)
        {
            /******************************************************************************
            b_0 = a_0
            ******************************************************************************/
            result = node->multipole_coeffs[0];
        }
        else
        {
            /******************************************************************************
            dz = (node_center - parent_center) / node_halflength
            b_l = ( sum( a_k * binom(l-1, k-1) / dz^k ) - a_0 / l ) * (dz / 2)^l
            ******************************************************************************/
            Complex<T> dz = (node->center - node->parent->center) / node->half_length;
            for (int i = power; i >= 1; i--)
            {
                result += node->multipole_coeffs[i] * get_binomial_coeff(binomcoeffs, power - 1, i - 1);
                result /= dz;
            }
            result -= node->multipole_coeffs[0] / power;
            result *= dz.pow(power);
            result /= 1 << power;
        }

        *coeff = result;
    }

    /******************************************************************************
    calculate the shifted multipole coefficients for a given maximum power
    assumes that the expansions are being shifted to the center of the parent node

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param power -- maximum expansion order to find the multipole coefficients of
    \param binomcoeffs -- pointer to array of binomial coefficients
    ******************************************************************************/
    template <typename T>
    __global__ void calculate_M2M_coeffs_kernel(TreeNode<T>* nodes, int numnodes, int power, long long* binomcoeffs)
    {
        /******************************************************************************
        array to hold multipole coefficients
        ******************************************************************************/
        extern __shared__ Complex<T> coeffs[];

        /******************************************************************************
        each thread is a node in the z direction
        ******************************************************************************/
        for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < numnodes; k += blockDim.z * gridDim.z)
        {
            TreeNode<T>* node = &nodes[k];

            /******************************************************************************
            each thread calculates a shifted multipole coefficient in the x thread
            direction for a child node in the y thread direction
            ******************************************************************************/
            for (int j = threadIdx.y; j < node->num_children; j += blockDim.y)
            {
                TreeNode<T>* child = node->children[j];
                for (int i = threadIdx.x; i <= power; i += blockDim.x)
                {
                    //assumes that coeffs has size blockDim.x * blockDim.y * blockDim.z
                    int index = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
                    calculate_M2M_coeff<T>(child, i, binomcoeffs, &coeffs[index]);
                    atomicAdd(&node->multipole_coeffs[i].re, coeffs[index].re);
                    atomicAdd(&node->multipole_coeffs[i].im, coeffs[index].im);
                }
            }
        }
    }

    /******************************************************************************
    calculate the shifted local coefficient for a node for a given power
    assumes that the expansion is being shifted from the center of the parent node

    \param node -- pointer to node
    \param power -- what order to find the local coefficient of
    \param maxpower -- maximum order to find the local coefficient of
    \param binomcoeffs -- pointer to array of binomial coefficients
    \param coeff -- pointer to where to store local coefficient
    ******************************************************************************/
    template <typename T>
    __device__ void calculate_L2L_coeff(TreeNode<T>* node, int power, int maxpower, long long* binomcoeffs, Complex<T>* coeff)
    {
        Complex<T> result;

        /******************************************************************************
        dz = (node_center - parent_center) / parent_halflength
        d_l = sum( c_k * binom(k, l) * dz^k ) / (2 * dz)^l
        ******************************************************************************/
        Complex<T> dz = (node->center - node->parent->center) / node->parent->half_length;
        for (int i = maxpower; i >= 1; i--)
        {
            result += node->parent->local_coeffs[i] * get_binomial_coeff(binomcoeffs, i, power);
            result *= dz;
        }
        result /= dz.pow(power);
        result /= 1 << power;

        *coeff = result;
    }

    /******************************************************************************
    calculate the shifted local coefficients for a given power
    assumes that the expansions are being shifted from the center of the parent
     node

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param power -- maximum expansion order to find the local coefficients of
    \param binomcoeffs -- pointer to array of binomial coefficients
    ******************************************************************************/
    template <typename T>
    __global__ void calculate_L2L_coeffs_kernel(TreeNode<T>* nodes, int numnodes, int power, long long* binomcoeffs)
    {
        /******************************************************************************
        each thread is a node in the y thread direction
        ******************************************************************************/
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < numnodes; j += blockDim.y * gridDim.y)
        {
            TreeNode<T>* node = &nodes[j];

            /******************************************************************************
            each thread calculates a local coefficient in the x thread direction
            ******************************************************************************/
            for (int i = threadIdx.x; i <= power; i += blockDim.x)
            {
                calculate_L2L_coeff<T>(node, i, power, binomcoeffs, &node->local_coeffs[i]);
            }
        }
    }

    /******************************************************************************
    calculate the local coefficient for a node for a given power from the multipole
    coefficient of a far node

    \param node_from -- pointer to node whose multipole expansion is being used
    \param node_to -- pointer to node whose local expansion is being calculated
    \param power -- what order to find the local coefficient of
    \param maxpower -- maximum order to find the local coefficient of
    \param binomcoeffs -- pointer to array of binomial coefficients
    \param coeff -- pointer to where to store local coefficient
    ******************************************************************************/
    template <typename T>
    __device__ void calculate_M2L_coeff(TreeNode<T>* node_from, TreeNode<T>* node_to, int power, int maxpower, long long* binomcoeffs, Complex<T>* coeff)
    {
        Complex<T> result;

        /******************************************************************************
        dz = (nodefrom_center - nodeto_center) / nodefrom_halflength
        ******************************************************************************/
        Complex<T> dz = (node_from->center - node_to->center) / node_from->half_length;

        if (power == 0)
        {
            /******************************************************************************
            c_0 = sum( b_k / (-dz)^k ) + b_0 * log(-dz) + b_0 * log(nodefrom_halflength)
            ******************************************************************************/
            for (int i = maxpower; i >= 1; i--)
            {
                result += node_from->multipole_coeffs[i];
                result /= -dz;
            }
            result += node_from->multipole_coeffs[0] * (-dz).log();
            /******************************************************************************
            include factor from true node size when calculating M2L in units of node length
            ******************************************************************************/
            result += node_from->multipole_coeffs[0] * log(node_from->half_length);
        }
        else
        {
            /******************************************************************************
            c_l = ( sum( b_k * binom(l + k - 1, k - 1) / (-dz)^k ) - b_0 / l ) / dz^l
            ******************************************************************************/
            for (int i = maxpower; i >= 1; i--)
            {
                result += node_from->multipole_coeffs[i] * get_binomial_coeff(binomcoeffs, power + i - 1, i - 1);
                result /= -dz;
            }
            result -= node_from->multipole_coeffs[0] / power;
            result /= dz.pow(power);
        }

        *coeff = result;
    }

    /******************************************************************************
    calculate the local coefficients for a given power from the multipole
    coefficients of far nodes

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param power -- maximum expansion order to find the local coefficients of
    \param binomcoeffs -- pointer to array of binomial coefficients
    ******************************************************************************/
    template <typename T>
    __global__ void calculate_M2L_coeffs_kernel(TreeNode<T>* nodes, int numnodes, int power, long long* binomcoeffs)
    {
        /******************************************************************************
        array to hold local coefficients
        ******************************************************************************/
        extern __shared__ Complex<T> coeffs[];

        /******************************************************************************
        each thread is a node in the z thread direction
        ******************************************************************************/
        for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < numnodes; k += blockDim.z * gridDim.z)
        {
            TreeNode<T>* node = &nodes[k];

            /******************************************************************************
            each thread calculates a local coefficient in the x thread direction for a
            member of the interaction list in the y thread direction
            ******************************************************************************/
            for (int j = threadIdx.y; j < node->num_same_level_interaction_list; j += blockDim.y)
            {
                for (int i = threadIdx.x; i <= power; i += blockDim.x)
                {
                    //assumes that coeffs has size blockDim.x * blockDim.y * blockDim.z
                    int index = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
                    calculate_M2L_coeff<T>(node->same_level_interaction_list[j], node, i, power, binomcoeffs, &coeffs[index]);
                    atomicAdd(&node->local_coeffs[i].re, coeffs[index].re);
                    atomicAdd(&node->local_coeffs[i].im, coeffs[index].im);
                }
            }
        }
    }

    /******************************************************************************
    calculate the local coefficient for a node for a given power from the stars
    of a far node

    \param node_from -- pointer to node whose stars are being used
    \param node_to -- pointer to node whose local expansion is being calculated
    \param power -- what order to find the local coefficient of
    \param maxpower -- maximum order to find the local coefficient of
    \param binomcoeffs -- pointer to array of binomial coefficients
    \param coeff -- pointer to where to store local coefficient
    ******************************************************************************/
    template <typename T>
    __device__ void calculate_P2L_coeff(TreeNode<T>* node_from, TreeNode<T>* node_to, int power, int maxpower, long long* binomcoeffs, star<T>* stars, Complex<T>* coeff)
    {
        Complex<T> result;

        for (int i = 0; i < node_from->numstars; i++)
        {
            star<T> star = stars[node_from->stars + i];

            /******************************************************************************
            dz = (star_position - nodeto_center) / nodeto_halflength
            ******************************************************************************/
            Complex<T> dz = (star.position - node_to->center) / node_to->half_length;

            if (power == 0)
            {
                result += star.mass * (-dz).log();
                result += star.mass * log(node_to->half_length);
            }
            else
            {
                result -= star.mass / (power * dz.pow(power));
            }
        }

        *coeff = result;
    }

    /******************************************************************************
    calculate the local coefficients for a given power from the stars of far nodes

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param power -- maximum expansion order to find the local coefficients of
    \param binomcoeffs -- pointer to array of binomial coefficients
    ******************************************************************************/
    template <typename T>
    __global__ void calculate_P2L_coeffs_kernel(TreeNode<T>* nodes, int numnodes, int power, long long* binomcoeffs, star<T>* stars)
    {
        /******************************************************************************
        array to hold local coefficients
        ******************************************************************************/
        extern __shared__ Complex<T> coeffs[];

        /******************************************************************************
        each thread is a node in the z thread direction
        ******************************************************************************/
        for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < numnodes; k += blockDim.z * gridDim.z)
        {
            TreeNode<T>* node = &nodes[k];

            /******************************************************************************
            each thread calculates a local coefficient in the x thread direction for a
            member of the interaction list in the y thread direction
            ******************************************************************************/
            for (int j = threadIdx.y; j < node->num_different_level_interaction_list; j += blockDim.y)
            {
                for (int i = threadIdx.x; i <= power; i += blockDim.x)
                {
                    //assumes that coeffs has size blockDim.x * blockDim.y * blockDim.z
                    int index = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
                    calculate_P2L_coeff<T>(node->different_level_interaction_list[j], node, i, power, binomcoeffs, stars, &coeffs[index]);
                    atomicAdd(&node->local_coeffs[i].re, coeffs[index].re);
                    atomicAdd(&node->local_coeffs[i].im, coeffs[index].im);
                }
            }
        }
    }

}

