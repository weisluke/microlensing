#pragma once

#include "complex.cuh"
#include "device_buffer.cuh"
#include "star.cuh"
#include "tree_node.cuh"
#include "util/util.cuh"

#include <vector>


/******************************************************************************
Compact description of the tree for the shooting kernel.

TreeNode<T> is ~0.9 KB (float) / ~1.4 KB (double) per node, carries 45
pointers and 32 multipole coefficients that shooting never reads, and the
stars a ray interacts with directly are spread over up to nine ranges (the
node and its neighbours). Here every leaf gets

  - a LeafRecord (centre, half length, expansion order, direct-star range),
  - its local coefficients in SoA layout with a fixed stride, and
  - one contiguous list of its node+neighbour stars, padded to 16/32 bytes so
    a star is one or two vector loads.

Nothing else in the code uses these arrays; the tree itself is unchanged
apart from the two index fields written into each node.
******************************************************************************/
namespace ipm_leaf
{
	template <typename T>
	struct alignas(4 * sizeof(T)) StarXYM
	{
		T x;
		T y;
		T m;
		T pad;
	};

	template <typename T>
	struct LeafRecord
	{
		T cx;
		T cy;
		T half_length;
		int expansion_order;
		int direct_begin;
		int direct_count;
	};

	/******************************************************************************
	number of direct stars of a node: its own plus those of its neighbours
	******************************************************************************/
	template <typename T>
	__device__ __forceinline__ int direct_count(const TreeNode<T>* node)
	{
		int n = node->numstars;
		for (int j = 0; j < node->num_neighbors; j++)
		{
			n += node->neighbors[j]->numstars;
		}
		return n;
	}

	/******************************************************************************
	assign every leaf an index and a slot in the direct-star list
	******************************************************************************/
	template <typename T>
	__global__ void count_leaves_kernel(TreeNode<T>* nodes, int numnodes, int* num_leaves, int* num_direct)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numnodes; i += blockDim.x * gridDim.x)
		{
			TreeNode<T>* node = &nodes[i];
			if (node->num_children == 0)
			{
				node->leaf_index = atomicAdd(num_leaves, 1);
				node->direct_begin = atomicAdd(num_direct, direct_count(node));
			}
			else
			{
				node->leaf_index = -1;
				node->direct_begin = 0;
			}
		}
	}

	/******************************************************************************
	fill the records, coefficients and star lists
	******************************************************************************/
	template <typename T>
	__global__ void fill_leaves_kernel(const TreeNode<T>* nodes, int numnodes, const star<T>* stars,
		LeafRecord<T>* leaves, T* coef_re, T* coef_im, int coef_stride, StarXYM<T>* direct)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numnodes; i += blockDim.x * gridDim.x)
		{
			const TreeNode<T>* node = &nodes[i];
			if (node->leaf_index < 0) continue;

			LeafRecord<T> rec;
			rec.cx = node->center.re;
			rec.cy = node->center.im;
			rec.half_length = node->half_length;
			rec.expansion_order = node->expansion_order < coef_stride - 1 ? node->expansion_order : coef_stride - 1;
			rec.direct_begin = node->direct_begin;

			T* cre = coef_re + 1LL * node->leaf_index * coef_stride;
			T* cim = coef_im + 1LL * node->leaf_index * coef_stride;
			for (int k = 0; k < coef_stride; k++)
			{
				cre[k] = k <= rec.expansion_order ? node->local_coeffs[k].re : T(0);
				cim[k] = k <= rec.expansion_order ? node->local_coeffs[k].im : T(0);
			}

			int n = 0;
			StarXYM<T>* out = direct + node->direct_begin;
			for (int k = 0; k < node->numstars; k++, n++)
			{
				const star<T> s = stars[node->stars + k];
				out[n] = StarXYM<T>{s.position.re, s.position.im, s.mass, T(0)};
			}
			for (int j = 0; j < node->num_neighbors; j++)
			{
				const TreeNode<T>* nb = node->neighbors[j];
				for (int k = 0; k < nb->numstars; k++, n++)
				{
					const star<T> s = stars[nb->stars + k];
					out[n] = StarXYM<T>{s.position.re, s.position.im, s.mass, T(0)};
				}
			}
			rec.direct_count = n;

			leaves[node->leaf_index] = rec;
		}
	}

#ifdef __CUDACC__ //the host driver uses kernel launches; the kernels above also compile as plain host code for testing
	/******************************************************************************
	the compact arrays (device) and the two counters that size them (managed).
	grow-only (see reserve_buffer); capacities are in elements
	******************************************************************************/
	template <typename T>
	struct LeafBuffers
	{
		LeafRecord<T>* leaves = nullptr;
		size_t leaves_cap = 0;
		T* coef_re = nullptr;
		size_t coef_re_cap = 0;
		T* coef_im = nullptr;
		size_t coef_im_cap = 0;
		StarXYM<T>* direct = nullptr;
		size_t direct_cap = 0;
		int* counters = nullptr; //number of leaves and of direct star entries
		size_t counters_cap = 0;

		/******************************************************************************
		free every buffer

		\param return_on_error -- whether to check for and return on a CUDA error

		\return bool -- false if return_on_error is true and a cudaFree failed
		******************************************************************************/
		bool release(bool return_on_error)
		{
			if (!release_buffer(leaves, leaves_cap, return_on_error, "cudaFree(*leaves)")) return false;
			if (!release_buffer(coef_re, coef_re_cap, return_on_error, "cudaFree(*coef_re)")) return false;
			if (!release_buffer(coef_im, coef_im_cap, return_on_error, "cudaFree(*coef_im)")) return false;
			if (!release_buffer(direct, direct_cap, return_on_error, "cudaFree(*direct)")) return false;
			if (!release_buffer(counters, counters_cap, return_on_error, "cudaFree(*counters)")) return false;
			return true;
		}
	};

	/******************************************************************************
	host driver. makes the buffers large enough (they only grow) and fills the
	compact arrays.

	\param tree -- per-level node arrays
	\param num_nodes -- number of nodes per level
	\param stars -- star array (sorted by node)
	\param expansion_order -- global expansion order (coefficient stride is +1)
	\param buf -- buffers of the compact arrays and the counters
	\param num_leaves, num_direct -- output sizes
	\param headroom -- factor applied to the sizes when a compact array grows
	\param stream -- stream the kernels run on
	******************************************************************************/
	template <typename T>
	bool build_leaf_records(const std::vector<TreeNode<T>*>& tree, const std::vector<int>& num_nodes, const star<T>* stars,
		int expansion_order, LeafBuffers<T>& buf, int& num_leaves, int& num_direct, double headroom, cudaStream_t stream, int verbose)
	{
		if (!reserve_buffer(buf.counters, buf.counters_cap, 2, BufferKind::managed, 1.0, "cudaMallocManaged(*counters)")) return false;
		int* counters = buf.counters;
		counters[0] = 0;
		counters[1] = 0;

		dim3 threads;
		dim3 blocks;
		for (size_t i = 0; i < tree.size(); i++)
		{
			threads = get_threads(count_leaves_kernel<T>, 256);
			blocks = get_blocks(threads, num_nodes[i]);
			count_leaves_kernel<T> <<<blocks, threads, 0, stream>>> (tree[i], num_nodes[i], &counters[0], &counters[1]);
			if (cuda_error("count_leaves_kernel", true, __FILE__, __LINE__, stream)) return false;
		}
		num_leaves = counters[0];
		num_direct = counters[1];
		const int coef_stride = expansion_order + 1;
		print_verbose("Compact tree: " << num_leaves << " leaves, " << num_direct << " direct star entries ("
			<< (num_leaves * (sizeof(LeafRecord<T>) + 2 * coef_stride * sizeof(T)) + num_direct * sizeof(StarXYM<T>)) / (1024.0 * 1024.0)
			<< " MB)\n", verbose, 2);

		if (!reserve_buffer(buf.leaves, buf.leaves_cap, 1ULL * num_leaves, BufferKind::device, headroom, "cudaMalloc(*leaves)")) return false;
		if (!reserve_buffer(buf.coef_re, buf.coef_re_cap, 1ULL * num_leaves * coef_stride, BufferKind::device, headroom, "cudaMalloc(*coef_re)")) return false;
		if (!reserve_buffer(buf.coef_im, buf.coef_im_cap, 1ULL * num_leaves * coef_stride, BufferKind::device, headroom, "cudaMalloc(*coef_im)")) return false;
		if (!reserve_buffer(buf.direct, buf.direct_cap, 1ULL * (num_direct > 0 ? num_direct : 1), BufferKind::device, headroom, "cudaMalloc(*direct)")) return false;

		for (size_t i = 0; i < tree.size(); i++)
		{
			threads = get_threads(fill_leaves_kernel<T>, 256);
			blocks = get_blocks(threads, num_nodes[i]);
			fill_leaves_kernel<T> <<<blocks, threads, 0, stream>>> (tree[i], num_nodes[i], stars, buf.leaves, buf.coef_re, buf.coef_im, coef_stride, buf.direct);
			if (cuda_error("fill_leaves_kernel", true, __FILE__, __LINE__, stream)) return false;
		}

		return true;
	}
#endif
}
