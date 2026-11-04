// Tests for ipm_fast::add_row_prefix_kernel (include/ipm_functions_fast.cuh). Needs a GPU:
//   tests/run_cuda_tests.sh /tmp/ipm_bench/cuda_tests tests/gpu/test_row_prefix.cu
// Prints one "PASS <name>" or "FAIL <name>: <detail>" line per case.
// Pixels and steps are multiples of 1/4 small enough that every partial sum is exact in float, so the kernel must
// match the host reference exactly: the map, the number of clamped pixels and their total.
#include "ipm_functions_fast.cuh"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;
static void report(const std::string& name, bool ok, const std::string& detail = "")
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
}

struct Clamp
{
	unsigned long long n = 0;
	double total = 0;
};

//per row, inclusive prefix of steps added to pixels; negative results set to 0 and counted
template <typename T>
static void reference(std::vector<T>& pixels, const std::vector<T>& steps, int nrows, int ncols, Clamp& c)
{
	for (int r = 0; r < nrows; r++)
	{
		double run = 0;
		for (int i = 0; i < ncols; i++)
		{
			const size_t k = static_cast<size_t>(r) * ncols + i;
			run += steps[k];
			double v = pixels[k] + run;
			if (v < 0)
			{
				c.n++;
				c.total += v;
				v = 0;
			}
			pixels[k] = static_cast<T>(v);
		}
	}
}

template <typename T>
static void fill(std::vector<T>& pixels, std::vector<T>& steps, int ncols, std::mt19937& rng)
{
	std::uniform_int_distribution<int> pix(0, 8), step(-4, 3);   //steps drift downward: some row sums go below 0
	for (auto& p : pixels) p = static_cast<T>(pix(rng) * 0.25);
	for (auto& s : steps) s = static_cast<T>(step(rng) * 0.25);
	steps[steps.size() - ncols] = -4;   //pixels are <= 2: the first pixel of the last row is always clamped
}

template <typename T>
static void run(const char* tname)
{
	const int shapes[][2] = {{1, 1}, {3, 7}, {2, 31}, {2, 32}, {3, 33}, {5, 100}, {40, 1000}, {9, 3000}};   //rows x columns
	std::mt19937 rng(20261015);
	for (auto& s : shapes)
	{
		const int nrows = s[0], ncols = s[1];
		const size_t n = static_cast<size_t>(nrows) * ncols;
		const std::string shape = std::to_string(nrows) + "x" + std::to_string(ncols);
		//production launch (one warp per row, ROW_PREFIX_BLOCK threads), and one small block that strides over the rows
		const unsigned int prod_blocks = static_cast<unsigned int>((1LL * nrows * 32 + ipm_fast::ROW_PREFIX_BLOCK - 1) / ipm_fast::ROW_PREFIX_BLOCK);
		const unsigned int launches[2][2] = {{prod_blocks, ipm_fast::ROW_PREFIX_BLOCK}, {1, 64}};
		for (auto& l : launches)
		{
			//two maps through the same counters, as for the two parity maps
			std::vector<T> p1(n), s1(n), p2(n), s2(n);
			fill(p1, s1, ncols, rng);
			fill(p2, s2, ncols, rng);
			T *dp1, *ds1, *dp2, *ds2;
			unsigned long long* dn;
			double* dt;
			cudaMalloc(&dp1, n * sizeof(T)); cudaMalloc(&ds1, n * sizeof(T));
			cudaMalloc(&dp2, n * sizeof(T)); cudaMalloc(&ds2, n * sizeof(T));
			cudaMalloc(&dn, sizeof(unsigned long long)); cudaMalloc(&dt, sizeof(double));
			cudaMemcpy(dp1, p1.data(), n * sizeof(T), cudaMemcpyHostToDevice);
			cudaMemcpy(ds1, s1.data(), n * sizeof(T), cudaMemcpyHostToDevice);
			cudaMemcpy(dp2, p2.data(), n * sizeof(T), cudaMemcpyHostToDevice);
			cudaMemcpy(ds2, s2.data(), n * sizeof(T), cudaMemcpyHostToDevice);
			cudaMemset(dn, 0, sizeof(unsigned long long));
			cudaMemset(dt, 0, sizeof(double));
			const Complex<int> npix(ncols, nrows);
			ipm_fast::add_row_prefix_kernel<T> <<<l[0], l[1]>>> (dp1, ds1, npix, dn, dt);
			ipm_fast::add_row_prefix_kernel<T> <<<l[0], l[1]>>> (dp2, ds2, npix, dn, dt);
			const cudaError_t err = cudaDeviceSynchronize();

			Clamp want;
			reference(p1, s1, nrows, ncols, want);
			reference(p2, s2, nrows, ncols, want);
			std::vector<T> g1(n), g2(n);
			Clamp got;
			cudaMemcpy(g1.data(), dp1, n * sizeof(T), cudaMemcpyDeviceToHost);
			cudaMemcpy(g2.data(), dp2, n * sizeof(T), cudaMemcpyDeviceToHost);
			cudaMemcpy(&got.n, dn, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
			cudaMemcpy(&got.total, dt, sizeof(double), cudaMemcpyDeviceToHost);
			size_t bad = 0;
			for (size_t k = 0; k < n; k++) bad += (g1[k] != p1[k]) + (g2[k] != p2[k]);
			const std::string name = std::string("row_prefix_") + tname + "_" + shape + "_grid" + std::to_string(l[0]) + "x" + std::to_string(l[1]);
			const std::string detail = "cuda " + std::string(cudaGetErrorString(err)) + ", " + std::to_string(bad) + " pixels differ, clamped "
				+ std::to_string(got.n) + " (want " + std::to_string(want.n) + "), total " + std::to_string(got.total)
				+ " (want " + std::to_string(want.total) + ")";
			report(name, err == cudaSuccess && bad == 0 && got.n == want.n && got.total == want.total && want.n > 0, detail);
			cudaFree(dp1); cudaFree(ds1); cudaFree(dp2); cudaFree(ds2); cudaFree(dn); cudaFree(dt);
		}
	}
}

int main()
{
	run<float>("float");
	run<double>("double");
	return n_fail ? 1 : 0;
}
