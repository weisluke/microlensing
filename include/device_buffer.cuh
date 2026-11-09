#pragma once

#include "util/util.cuh"

#include <cmath> //for std::ceil
#include <cstddef> //for std::ptrdiff_t
#include <new> //for std::bad_alloc
#include <vector>

#include <cuda_runtime.h>


/******************************************************************************
kind of memory a grow-only buffer lives in
******************************************************************************/
enum class BufferKind { device, managed };

/******************************************************************************
make sure a buffer holds at least n elements. the buffer only grows: if it is
already large enough it is left untouched

\param ptr -- reference to the pointer to the buffer. may be nullptr
\param cap -- reference to the capacity of the buffer, in elements
\param n -- number of elements needed. 0 counts as 1
\param kind -- device memory (cudaMalloc) or managed memory (cudaMallocManaged)
\param headroom -- factor applied to n when growing, so later small increases
				   don't reallocate. at least 1
\param name -- to print in error msg

\return bool -- true if the buffer holds at least n elements, false if
				allocation failed (ptr is then nullptr and cap is 0)
******************************************************************************/
template <typename P>
bool reserve_buffer(P*& ptr, size_t& cap, size_t n, BufferKind kind, double headroom, const char* name)
{
	if (n == 0)
	{
		n = 1;
	}
	if (ptr != nullptr && cap >= n)
	{
		return true;
	}

	if (ptr != nullptr)
	{
		cudaFree(ptr);
		if (cuda_error("cudaFree(*ptr)", false, __FILE__, __LINE__)) {ptr = nullptr; cap = 0; return false;}
	}
	ptr = nullptr;
	cap = 0;

	size_t grown = static_cast<size_t>(std::ceil(static_cast<double>(n) * headroom));
	size_t new_cap = (grown > n) ? grown : n;

	cudaError_t err = (kind == BufferKind::managed)
		? cudaMallocManaged(&ptr, new_cap * sizeof(P))
		: cudaMalloc(&ptr, new_cap * sizeof(P));
	//a failed allocation leaves its error as the last error, which cuda_error reports
	if (cuda_error(name, false, __FILE__, __LINE__) || err != cudaSuccess)
	{
		ptr = nullptr;
		cap = 0;
		return false;
	}
	cap = new_cap;
	return true;
}

/******************************************************************************
free a buffer made by reserve_buffer

\param ptr -- reference to the pointer to the buffer. set to nullptr
\param cap -- reference to the capacity of the buffer. set to 0
\param return_on_error -- whether to check for and return on a CUDA error
\param name -- to print in error msg

\return bool -- false if return_on_error is true and cudaFree failed, else true
******************************************************************************/
template <typename P>
bool release_buffer(P*& ptr, size_t& cap, bool return_on_error, const char* name)
{
	bool failed = false;
	if (ptr != nullptr)
	{
		cudaFree(ptr);
		failed = return_on_error && cuda_error(name, false, __FILE__, __LINE__);
	}
	ptr = nullptr;
	cap = 0;
	return !failed;
}

/******************************************************************************
Thrust allocator backed by one grow-only device buffer, so repeated Thrust
calls don't cudaMalloc and cudaFree their temporary storage.

one Thrust call can hold several temporary allocations at once
(thrust::minmax_element holds two), so the buffer is a bump allocator:
allocate hands out the next aligned part of the buffer, deallocate keeps it,
and reset, called before each Thrust call, starts again at the beginning of
the buffer (allocate does so itself when nothing is in use). a request that
does not fit while other parts are in use gets a buffer of its own, which
deallocate frees; once nothing is in use any more, the buffer grows to all
that the call needed, so the next call of the same size fits. the buffer only
grows while nothing is in use. release frees everything
******************************************************************************/
struct ThrustScratch
{
	using value_type = char;

	static constexpr size_t ALIGNMENT = 256; //alignment cudaMalloc gives

	char* ptr = nullptr;
	size_t cap = 0; //bytes
	size_t used = 0; //bytes of ptr handed out since the last reset
	size_t needed = 0; //bytes requested since the last reset, each rounded up to ALIGNMENT
	int live = 0; //parts of ptr in use
	std::vector<char*> extra; //buffers of the requests that did not fit, until their deallocate
	std::vector<size_t> extra_cap;

	bool in_use() const
	{
		return live != 0 || !extra.empty();
	}

	/******************************************************************************
	start handing out the buffer from its beginning again. does nothing while a
	part of it or an extra buffer is in use
	******************************************************************************/
	void reset()
	{
		if (in_use()) return;
		used = 0;
		needed = 0;
	}

	char* allocate(std::ptrdiff_t n)
	{
		const size_t bytes = (static_cast<size_t>(n) + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
		if (!in_use())
		{
			reset();
			if (!reserve_buffer(ptr, cap, bytes, BufferKind::device, 1.0, "thrust scratch"))
			{
				throw std::bad_alloc();
			}
		}
		needed += bytes;
		if (used + bytes <= cap)
		{
			char* p = ptr + used;
			used += bytes;
			live++;
			return p;
		}

		char* p = nullptr;
		size_t c = 0;
		if (!reserve_buffer(p, c, bytes, BufferKind::device, 1.0, "thrust scratch (extra)"))
		{
			throw std::bad_alloc();
		}
		extra.push_back(p);
		extra_cap.push_back(c);
		return p;
	}

	void deallocate(char* p, size_t)
	{
		bool was_extra = false;
		for (size_t i = 0; i < extra.size() && !was_extra; i++)
		{
			if (extra[i] == p)
			{
				release_buffer(extra[i], extra_cap[i], false, "cudaFree(thrust scratch (extra))");
				extra.erase(extra.begin() + i);
				extra_cap.erase(extra_cap.begin() + i);
				was_extra = true;
			}
		}
		if (!was_extra && live > 0) live--;

		//after a call that did not fit, grow to all it needed. if that fails, ptr is nullptr and the next
		//allocate tries again
		if (!in_use() && needed > cap)
		{
			reserve_buffer(ptr, cap, needed, BufferKind::device, 1.0, "thrust scratch");
		}
	}

	bool release()
	{
		bool ok = true;
		for (size_t i = 0; i < extra.size(); i++)
		{
			ok = release_buffer(extra[i], extra_cap[i], true, "cudaFree(thrust scratch (extra))") && ok;
		}
		extra.clear();
		extra_cap.clear();
		ok = release_buffer(ptr, cap, true, "cudaFree(thrust scratch)") && ok;
		used = 0;
		needed = 0;
		live = 0;
		return ok;
	}
};
