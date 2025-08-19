#pragma once

#include "complex.cuh"
#include "star.cuh"


namespace treenode
{

    /******************************************************************************
    number of stars to use directly when shooting rays
    this helps determine the size of the tree
    ******************************************************************************/
    const int MAX_NUM_STARS_DIRECT = 32;
    
    /******************************************************************************
    maximum expansion order for the fast multipole method
    uses 31 so that maximum number of coefficients is 32
    ******************************************************************************/
    const int MAX_EXPANSION_ORDER = 31;

    const int MAX_NUM_CHILDREN = 4; //a node has at most 4 children
    const int MAX_NUM_NEIGHBORS = 8; //a node has at most 8 neighbors
    const int MAX_NUM_SAME_LEVEL_INTERACTION_LIST = 27; //a node has at most 27 elements in its interaction list from the same level
    const int MAX_NUM_DIFFERENT_LEVEL_INTERACTION_LIST = 5; //a node has at most 5 elements in its interaction list from different levels

}

/******************************************************************************
template class for handling nodes of a tree
******************************************************************************/
template <typename T>
class TreeNode
{
public:
    Complex<T> center;
    T half_length;

    int level;

    TreeNode* parent;
    TreeNode* children[treenode::MAX_NUM_CHILDREN];
    int num_children;
    TreeNode* neighbors[treenode::MAX_NUM_NEIGHBORS];
    int num_neighbors;
    TreeNode* same_level_interaction_list[treenode::MAX_NUM_SAME_LEVEL_INTERACTION_LIST];
    int num_same_level_interaction_list;
    TreeNode* different_level_interaction_list[treenode::MAX_NUM_DIFFERENT_LEVEL_INTERACTION_LIST];
    int num_different_level_interaction_list;

    int stars; //position of this node's stars in array of stars
    int numstars; //number of stars in this node

    int expansion_order;
    Complex<T> multipole_coeffs[treenode::MAX_EXPANSION_ORDER + 1];
    Complex<T> local_coeffs[treenode::MAX_EXPANSION_ORDER + 1];


    /******************************************************************************
	default constructor
	******************************************************************************/
    __host__ __device__ TreeNode(Complex<T> ctr = Complex<T>(), T hl = 0, int lvl = 0, TreeNode* p = nullptr)
    {
        center = ctr;
        half_length = hl;
        level = lvl;

		parent = p;
		for (int i = 0; i < treenode::MAX_NUM_CHILDREN; i++)
        {
            children[i] = nullptr;
        }
        num_children = 0;
        for (int i = 0; i < treenode::MAX_NUM_NEIGHBORS; i++)
        {
            neighbors[i] = nullptr;
        }
        num_neighbors = 0;
        for (int i = 0; i < treenode::MAX_NUM_SAME_LEVEL_INTERACTION_LIST; i++)
        {
            same_level_interaction_list[i] = nullptr;
        }
        num_same_level_interaction_list = 0;
        for (int i = 0; i < treenode::MAX_NUM_DIFFERENT_LEVEL_INTERACTION_LIST; i++)
        {
            different_level_interaction_list[i] = nullptr;
        }
        num_different_level_interaction_list = 0;

        stars = 0;
        numstars = 0;
        expansion_order = 0;

        for (int i = 0; i <= treenode::MAX_EXPANSION_ORDER; i++)
        {
            multipole_coeffs[i] = Complex<T>();
            local_coeffs[i] = Complex<T>();
        }
	}
    
    /******************************************************************************
	assignment operator
	******************************************************************************/
	template <typename U> __host__ __device__ TreeNode& operator=(const TreeNode<U>& n)
	{
        center = n.center;
        half_length = n.half_length;
        level = n.level;

        parent = n.parent;
		for (int i = 0; i < treenode::MAX_NUM_CHILDREN; i++)
        {
            children[i] = n.children[i];
        }
        num_children = n.num_children;
        for (int i = 0; i < treenode::MAX_NUM_NEIGHBORS; i++)
        {
            neighbors[i] = n.neighbors[i];
        }
        num_neighbors = n.num_neighbors;
        for (int i = 0; i < treenode::MAX_NUM_SAME_LEVEL_INTERACTION_LIST; i++)
        {
            same_level_interaction_list[i] = n.same_level_interaction_list[i];
        }
        num_same_level_interaction_list = n.num_same_level_interaction_list;
        for (int i = 0; i < treenode::MAX_NUM_DIFFERENT_LEVEL_INTERACTION_LIST; i++)
        {
            different_level_interaction_list[i] = n.different_level_interaction_list[i];
        }
        num_different_level_interaction_list = n.num_different_level_interaction_list;

        stars = n.stars;
        numstars = n.numstars;
        expansion_order = n.expansion_order;
        for (int i = 0; i <= treenode::MAX_EXPANSION_ORDER; i++)
        {
            multipole_coeffs[i] = n.multipole_coeffs[i];
            local_coeffs[i] = n.local_coeffs[i];
        }

		return *this;
	}

    /******************************************************************************
    create a child node

    \param nodes -- pointer to array of nodes for the next level
    \param start -- where to begin placing children
    \param idx -- index of the child to be created (0, 1, 2, or 3)
    ******************************************************************************/
    __host__ __device__ void make_child(TreeNode* nodes, int start, int idx)
    {
        T new_half_length = half_length / 2;

        Complex<T> offsets[treenode::MAX_NUM_CHILDREN] = 
        {
            Complex<T>(1, 1),
            Complex<T>(-1, 1),
            Complex<T>(-1, -1),
            Complex<T>(1, -1)
        };

        Complex<T> new_center = center + new_half_length * offsets[idx];

        int new_index = start + idx;
        nodes[new_index] = TreeNode(new_center, new_half_length, level + 1, this);
        children[idx] = &nodes[new_index];
    }

    __host__ __device__ void make_children(TreeNode* nodes, int start)
    {
        for (int i = 0; i < treenode::MAX_NUM_CHILDREN; i++)
        {
            make_child(nodes, start, i);
        }
        num_children = treenode::MAX_NUM_CHILDREN;
    }


    /******************************************************************************
    set neighbors for a node
    ******************************************************************************/
    __device__ void set_neighbors()
    {
        /******************************************************************************
        root node has no neighbors
        ******************************************************************************/
        if (level == 0)
        {
            return;
        }

        /******************************************************************************
        add parent's children as neighbors
        ******************************************************************************/
        for (int i = 0; i < parent->num_children; i++)
        {
            TreeNode* node = parent->children[i];

            /******************************************************************************
            if node is empty, skip it
            ******************************************************************************/
            if (node->numstars == 0)
            {
                continue;
            }

            if (node != this)
            {
                neighbors[num_neighbors++] = node;
            }
        }
        /******************************************************************************
        parent's neighbors' children are on the same level as the node
        add the close ones as neighbors, and the far ones to the interaction list
        ******************************************************************************/
        for (int i = 0; i < parent->num_neighbors; i++)
        {
            TreeNode* neighbor = parent->neighbors[i];

            /******************************************************************************
            if neighbor is empty, skip it
            ******************************************************************************/
            if (neighbor->numstars == 0)
            {
                continue;
            }

            if (neighbor->num_children == 0)
            {
                if (fabs(neighbor->center.re - center.re) < 3.5 * half_length
                    && fabs(neighbor->center.im - center.im) < 3.5 * half_length)
                {
                    neighbors[num_neighbors++] = neighbor;
                }
                else
                {
                    different_level_interaction_list[num_different_level_interaction_list++] = neighbor;
                }
            }
            else
            {
                for (int j = 0; j < neighbor->num_children; j++)
                {
                    TreeNode* node = neighbor->children[j];

                    /******************************************************************************
                    if node is empty, skip it
                    ******************************************************************************/
                    if (node->numstars == 0)
                    {
                        continue;
                    }

                    if (fabs(node->center.re - center.re) < 3 * half_length
                        && fabs(node->center.im - center.im) < 3 * half_length)
                    {
                        neighbors[num_neighbors++] = node;
                    }
                    else
                    {
                        same_level_interaction_list[num_same_level_interaction_list++] = node;
                    }
                }
            }
        }
    }

    __host__ __device__ void set_multipole_coeffs(Complex<T>* coeffs, int power)
    {
        for (int i = 0; i <= power; i++)
        {
            multipole_coeffs[i] = coeffs[i];
        }
        expansion_order = (expansion_order > power ? expansion_order : power);
    }
    
    __host__ __device__ void add_multipole_coeffs(Complex<T>* coeffs, int power)
    {
        for (int i = 0; i <= power; i++)
        {
            multipole_coeffs[i] += coeffs[i];
        }
        expansion_order = (expansion_order > power ? expansion_order : power);
    }

    __host__ __device__ void set_local_coeffs(Complex<T>* coeffs, int power)
    {
        for (int i = 0; i <= power; i++)
        {
            local_coeffs[i] = coeffs[i];
        }
        expansion_order = (expansion_order > power ? expansion_order : power);
    }

    __host__ __device__ void add_local_coeffs(Complex<T>* coeffs, int power)
    {
        for (int i = 0; i <= power; i++)
        {
            local_coeffs[i] += coeffs[i];
        }
        expansion_order = (expansion_order > power ? expansion_order : power);
    }

};


namespace treenode
{

    /******************************************************************************
    get the number of nodes which, when including its neighbors, are nonempty, and
    the minimum and maximum number of stars contained within them

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param num_nonempty_nodes -- pointer to number of non empty nodes
    \param min_n_stars -- pointer to minimum number of stars
    \param max_n_stars -- pointer to maximum number of stars
    ******************************************************************************/
    template <typename T>
    __global__ void get_node_star_info_kernel(TreeNode<T>* nodes, int numnodes, int* num_nonempty_nodes, int* min_n_stars, int* max_n_stars)
    {
        int x_index = blockIdx.x * blockDim.x + threadIdx.x;
        int x_stride = blockDim.x * gridDim.x;

        for (int i = x_index; i < numnodes; i += x_stride)
        {
            TreeNode<T>* node = &nodes[i];
            int nstars = node->numstars;
            
            for (int j = 0; j < node->num_neighbors; j++)
            {
                nstars += node->neighbors[j]->numstars;
            }

            if (nstars > treenode::MAX_NUM_STARS_DIRECT)
            {
                atomicAdd(num_nonempty_nodes, 1);
            }

            atomicMin(min_n_stars, nstars);
            atomicMax(max_n_stars, nstars);
        }
    }

    /******************************************************************************
    create the children for nodes 

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param num_nonempty_nodes -- pointer to number of non empty nodes
    \param children -- pointer to child nodes
    ******************************************************************************/
    template <typename T>
    __global__ void create_children_kernel(TreeNode<T>* nodes, int numnodes, int* num_nonempty_nodes, TreeNode<T>* children)
    {
        int x_index = blockIdx.x * blockDim.x + threadIdx.x;
        int x_stride = blockDim.x * gridDim.x;

        for (int i = x_index; i < numnodes; i += x_stride)
        {
            TreeNode<T>* node = &nodes[i];
            int nstars = node->numstars;

            for (int j = 0; j < node->num_neighbors; j++)
            {
                nstars += node->neighbors[j]->numstars;
            }

            if (nstars > treenode::MAX_NUM_STARS_DIRECT)
            {
                /******************************************************************************
                assumes that array of child nodes has MAX_NUM_CHILDREN * num_nonempty_nodes at
                the start. every MAX_NUM_CHILDREN elements are then the children of this node
                atomicSub ensures that children are placed at unique locations
                ******************************************************************************/
                node->make_children(children, treenode::MAX_NUM_CHILDREN * atomicSub(num_nonempty_nodes, 1));
            }
        }
    }

    /******************************************************************************
    sort the stars into the children of each node

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param stars -- pointer to array of point mass lenses
    \param temp_stars -- pointer to temp array of point mass lenses, used for
                         swapping
    ******************************************************************************/
    template <typename T>
    __global__ void sort_stars_kernel(TreeNode<T>* nodes, int numnodes, star<T>* stars, star<T>* temp_stars)
    {
	    extern __shared__ int shared_memory[];
        int* n_stars_top = &shared_memory[0];
        int* n_stars_bottom = &n_stars_top[blockDim.x];
        int* n_stars_children = &n_stars_bottom[blockDim.x];

        /******************************************************************************
        each thread is a node in the x thread direction
        ******************************************************************************/
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < numnodes; j += blockDim.x * gridDim.x)
        {
            TreeNode<T>* node = &nodes[j];

            /******************************************************************************
            if node is empty or has no children, skip it
            ******************************************************************************/
            if (node->numstars == 0 || node->num_children == 0)
            {
                continue;
            }

            if (threadIdx.y == 0)
            {
                n_stars_top[threadIdx.x] = 0;
                n_stars_bottom[threadIdx.x] = 0;
                n_stars_children[threadIdx.x * treenode::MAX_NUM_CHILDREN + 0] = 0;
                n_stars_children[threadIdx.x * treenode::MAX_NUM_CHILDREN + 1] = 0;
                n_stars_children[threadIdx.x * treenode::MAX_NUM_CHILDREN + 2] = 0;
                n_stars_children[threadIdx.x * treenode::MAX_NUM_CHILDREN + 3] = 0;
            }
            __syncthreads();

            /******************************************************************************
            each thread is a star within the parent node in the y direction
            in the first pass, figure out whether the star is above or below the center
            ******************************************************************************/
            for (int i = threadIdx.y; i < node->numstars; i += blockDim.y)
            {
                if (stars[node->stars + i].position.im >= node->center.im)
                {
                    /******************************************************************************
                    atomic addition returns the old value, so this is guaranteed to copy the star
                    to a unique location in the temp array
                    ******************************************************************************/
                    temp_stars[node->stars + atomicAdd(&n_stars_top[threadIdx.x], 1)] = stars[node->stars + i];
                }
                else
                {
                    temp_stars[node->stars + node->numstars - 1 - atomicAdd(&n_stars_bottom[threadIdx.x], 1)] = stars[node->stars + i];
                }
            }
            __syncthreads();

            /******************************************************************************
            in the second pass, figure out whether the star is left or right of the center
            ******************************************************************************/
            for (int i = threadIdx.y; i < node->numstars; i += blockDim.y)
            {
                /******************************************************************************
                if the star was in the top, then sort left and right
                ******************************************************************************/
                if (i < n_stars_top[threadIdx.x])
                {
                    if (temp_stars[node->stars + i].position.re >= node->center.re)
                    {
                        stars[node->stars + atomicAdd(&n_stars_children[threadIdx.x * treenode::MAX_NUM_CHILDREN + 0], 1)] = temp_stars[node->stars + i];
                    }
                    else
                    {
                        stars[node->stars + n_stars_top[threadIdx.x] - 1 - atomicAdd(&n_stars_children[threadIdx.x * treenode::MAX_NUM_CHILDREN + 1], 1)] = temp_stars[node->stars + i];
                    }
                }
                /******************************************************************************
                if the star was in the bottom, then sort left and right
                ******************************************************************************/
                else
                {
                    if (temp_stars[node->stars + i].position.re < node->center.re)
                    {
                        stars[node->stars + n_stars_top[threadIdx.x] + atomicAdd(&n_stars_children[threadIdx.x * treenode::MAX_NUM_CHILDREN + 2], 1)] = temp_stars[node->stars + i];
                    }
                    else
                    {
                        stars[node->stars + node->numstars - 1 - atomicAdd(&n_stars_children[threadIdx.x * treenode::MAX_NUM_CHILDREN + 3], 1)] = temp_stars[node->stars + i];
                    }
                }
            }
            __syncthreads();

            /******************************************************************************
            once the sorting is done, assign the starting position of stars to the children
            along with the number of stars
            ******************************************************************************/
            if (threadIdx.y == 0)
            {
                node->children[0]->stars = node->stars;
                node->children[0]->numstars = n_stars_children[threadIdx.x * treenode::MAX_NUM_CHILDREN + 0];

#pragma unroll
                for (int i = 1; i < treenode::MAX_NUM_CHILDREN; i++)
                {
                    node->children[i]->stars = node->children[i - 1]->stars + node->children[i - 1]->numstars;
                    node->children[i]->numstars = n_stars_children[threadIdx.x * treenode::MAX_NUM_CHILDREN + i];
                }
            }
            __syncthreads();
        }
    }

    /******************************************************************************
    set the neighbors for nodes

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    ******************************************************************************/
    template <typename T>
    __global__ void set_neighbors_kernel(TreeNode<T>* nodes, int numnodes)
    {
        int x_index = blockIdx.x * blockDim.x + threadIdx.x;
        int x_stride = blockDim.x * gridDim.x;

        for (int i = x_index; i < numnodes; i += x_stride)
        {
            nodes[i].set_neighbors();
        }
    }

    /******************************************************************************
    find the node nearest to a given position

    \param z -- position to find the nearest node to
    \param root -- pointer to root node

    \return pointer to nearest node
    ******************************************************************************/
    template <typename T>
    __host__ __device__ TreeNode<T>* get_nearest_node(Complex<T> z, TreeNode<T>* root)
    {
        TreeNode<T>* node = root;
        while (node->num_children > 0)
        {
            //critical curves can land outside the field of stars
            if (fabs(z.re - node->center.re) > node->half_length ||
                fabs(z.im - node->center.im) > node->half_length)
            {
                return node;
            }

            if (z.re >= node->center.re && z.im >= node->center.im)
            {
                node = node->children[0];
            }
            else if (z.re < node->center.re && z.im >= node->center.im)
            {
                node = node->children[1];
            }
            else if (z.re < node->center.re && z.im < node->center.im)
            {
                node = node->children[2];
            }
            else
            {
                node = node->children[3];
            }
        }
        
        return node;
    }

}

