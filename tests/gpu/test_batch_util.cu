// Tests for the helpers behind batched map generation: the stream-aware cuda_error() and the thread-local last
// error code (include/util/util.cuh), the per-thread device-property cache, and the grow-only buffers of
// include/device_buffer.cuh. Needs the GPU (two GPUs for the device-property case, which skips otherwise).
// The take_last_cuda_error case makes a 1 PiB cudaMalloc fail on purpose, so cuda_error() printing an
// "Error code: 2 (out of memory)" message to stderr is expected output.
// Prints one "PASS <name>" or "FAIL <name>: <detail>" line per case and exits 1 if any case fails.
#include "device_buffer.cuh"
#include "spin.cuh"

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

static int n_fail = 0;
static void report(const std::string& name, bool ok, const std::string& detail = "")
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
	std::fflush(stdout);
}

template <typename F>
static double seconds(F f)
{
	auto t0 = std::chrono::steady_clock::now();
	f();
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

__global__ void empty_kernel() {}

static void stream_sync_waits_only_for_its_stream()
{
	cudaStream_t a, b;
	cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
	//load both kernels first: a lazy module load while the spin kernel runs would wait for it
	launch_spin(0.0, a);
	empty_kernel<<<1, 1, 0, b>>>();
	cudaDeviceSynchronize();
	launch_spin(1.0, a);
	empty_kernel<<<1, 1, 0, b>>>();
	bool failed = false;
	double t = seconds([&] { failed = cuda_error("b", true, __FILE__, __LINE__, b); });
	cudaDeviceSynchronize();
	cudaStreamDestroy(a);
	cudaStreamDestroy(b);
	report("stream_sync_waits_only_for_its_stream", !failed && t < 0.25,
		failed ? "cuda_error returned true" : "took " + std::to_string(t) + " s");
}

static void legacy_sync_still_waits_for_device()
{
	cudaStream_t a;
	cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
	launch_spin(0.5, a);
	double t = seconds([&] { cuda_error("x", true, __FILE__, __LINE__); });
	cudaDeviceSynchronize();
	cudaStreamDestroy(a);
	report("legacy_sync_still_waits_for_device", t >= 0.4, "took " + std::to_string(t) + " s");
}

static void take_last_cuda_error_reports_and_resets()
{
	take_last_cuda_error();
	void* p = nullptr;
	cudaMalloc(&p, 1ULL << 50);
	bool reported = cuda_error("cudaMalloc", false, __FILE__, __LINE__);
	int first = take_last_cuda_error();
	int second = take_last_cuda_error();
	int other = -1;
	std::thread([&] { other = take_last_cuda_error(); }).join();
	report("take_last_cuda_error_reports_and_resets",
		reported && first == cudaErrorMemoryAllocation && second == 0 && other == 0,
		!reported ? "cuda_error returned false" : first != cudaErrorMemoryAllocation ? "first code " + std::to_string(first)
		: second != 0 ? "second code " + std::to_string(second) : "another thread saw code " + std::to_string(other));
}

static int bus_id(int device)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	return prop.pciBusID;
}

static void device_prop_is_per_thread()
{
	int n = 0;
	cudaGetDeviceCount(&n);
	if (n < 2)
	{
		std::printf("SKIP device_prop_is_per_thread: needs 2 devices, found %d\n", n);
		return;
	}
	int b0 = bus_id(0);
	int b1 = bus_id(1);
	int in_thread = -1;
	std::thread([&] { cudaSetDevice(1); in_thread = cached_device_prop().pciBusID; }).join();
	cudaSetDevice(0);
	int m0 = cached_device_prop().pciBusID;
	cudaSetDevice(1);
	int m1 = cached_device_prop().pciBusID;
	cudaSetDevice(0);
	report("device_prop_is_per_thread", b0 != b1 && in_thread == b1 && m0 == b0 && m1 == b1,
		"thread " + std::to_string(in_thread) + " (want " + std::to_string(b1) + "), main " + std::to_string(m0) + "/" + std::to_string(m1)
		+ " (want " + std::to_string(b0) + "/" + std::to_string(b1) + ")");
}

static void reserve_buffer_grows_and_reuses()
{
	int* p = nullptr;
	size_t cap = 0;
	bool ok = reserve_buffer(p, cap, 100, BufferKind::device, 1.0, "t");
	bool a = ok && p != nullptr && cap == 100;
	int* first = p;
	ok = reserve_buffer(p, cap, 50, BufferKind::device, 1.0, "t");
	bool b = ok && p == first && cap == 100;
	ok = reserve_buffer(p, cap, 200, BufferKind::device, 1.25, "t");
	bool c = ok && p != nullptr && cap == 250;
	bool d = release_buffer(p, cap, true, "t") && p == nullptr && cap == 0;

	float* m = nullptr;
	size_t mcap = 0;
	bool e = reserve_buffer(m, mcap, 0, BufferKind::managed, 1.0, "m") && m != nullptr && mcap == 1;
	if (e) { m[0] = 3.5f; e = m[0] == 3.5f; }
	bool f = release_buffer(m, mcap, true, "m") && m == nullptr && mcap == 0;
	report("reserve_buffer_grows_and_reuses", a && b && c && d && e && f,
		std::string("steps ") + (a ? "" : "[100]") + (b ? "" : "[50 reuse]") + (c ? "" : "[200 headroom]")
		+ (d ? "" : "[release]") + (e ? "" : "[managed]") + (f ? "" : "[managed release]") + " failed");
}

static void release_of_null_makes_no_cuda_call()
{
	cudaDeviceSynchronize();
	launch_spin(1.0, 0);
	int* p = nullptr;
	size_t cap = 0;
	bool ok = false;
	double t = seconds([&] { ok = release_buffer(p, cap, true, "x"); });
	cudaDeviceSynchronize();
	report("release_of_null_makes_no_cuda_call", ok && t < 0.1,
		!ok ? "returned false" : "took " + std::to_string(t) + " s");
}

int main()
{
	stream_sync_waits_only_for_its_stream();
	legacy_sync_still_waits_for_device();
	take_last_cuda_error_reports_and_resets();
	device_prop_is_per_thread();
	reserve_buffer_grows_and_reuses();
	release_of_null_makes_no_cuda_call();

	std::printf("%s: %d failed\n", n_fail ? "FAILED" : "OK", n_fail);
	return n_fail ? 1 : 0;
}
