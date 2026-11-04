// Tests for running IPM<T> objects as batch workers (include/microlensing.cuh, include/ipm.cuh, include/ipm_leaf.cuh):
//   - a map made on the last visible device on a created stream must match one made by a default object, and an
//     invalid device must make run() fail;
//   - a reused object (reuse_memory = 1) keeps its buffers between maps: once warmed up, a run makes no call that
//     waits for other work on the device, its maps match those of fresh objects, it gives out and saves no parity
//     map or histogram of an earlier map, and release_memory() gives every buffer back; a default object keeps
//     today's footprint and makes no CUDA call when it never runs;
//   - thrust::minmax_element holds two temporary allocations at once; ThrustScratch, reset before each call, gives
//     them parts of its buffer that do not overlap and, once warmed up, allocates nothing.
// Needs the GPU; with one visible GPU all objects share device 0. The timing cases occupy the legacy stream of
// device 0 with a spin kernel: a call that synchronizes the device would wait for it.
// The invalid-device case prints an error message to stderr, which is expected output.
// float and double cannot share one program: the kernels of fmm.cuh declare their extern shared array with the
// same name for both types. This file therefore tests float; tests/gpu/test_ipm_batch_double.cu defines
// IPM_BATCH_TEST_DOUBLE and includes it to test double.
// Prints one "PASS <name>" or "FAIL <name>: <detail>" line per case (and "INFO" lines with measured times) and
// exits 1 if any case fails.
#include "device_buffer.cuh"
#include "ipm.cuh"
#include "spin.cuh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h> //for mkdtemp
#include <utility>
#include <vector>

static int n_fail = 0;
static void report(const std::string& name, bool ok, const std::string& detail = "")
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
	std::fflush(stdout);
}

template <typename T>
static const char* type_name()
{
	return sizeof(T) == sizeof(float) ? "float" : "double";
}

template <typename T>
static std::string case_name(const char* name)
{
	return std::string(name) + "_" + type_name<T>();
}

// the loose bounds on two maps of the same inputs (two serial runs are not bit-identical). the sums of the float
// parity maps vary more between serial runs: over 120 runs of map B by fresh default objects and a reused object,
// the saddle map's sum spread over 1.6e-5 (relative), and 1-2 % of pairs of runs differed by more than 1e-5,
// so their bound is 5e-5. the double parity maps' sums spread over 3e-15
template <typename T> struct Tolerance;
template <> struct Tolerance<float> { static constexpr double sum = 1e-5; static constexpr double parity_sum = 5e-5; static constexpr double diff = 1e-2; };
template <> struct Tolerance<double> { static constexpr double sum = 1e-12; static constexpr double parity_sum = 1e-12; static constexpr double diff = 1e-8; };

static double seconds_since(std::chrono::steady_clock::time_point start)
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// the lens of the brief: kappa 0.4, shear 0.4, kappa_star 0.2, half-lengths 10, 600x600 pixels, seed 7
template <typename T>
static void configure(IPM<T>& ipm)
{
	ipm.kappa_tot = static_cast<T>(0.4);
	ipm.shear = static_cast<T>(0.4);
	ipm.kappa_star = static_cast<T>(0.2);
	ipm.half_length_y = Complex<T>(10, 10);
	ipm.num_pixels_y = Complex<int>(600, 600);
	ipm.random_seed = 7;
	ipm.write_stars = 0;
}

// every input of IPM<T> set to the value of a fresh object: run() changes some inputs (the seed, m_lower and
// m_upper), so a reused object gets all of them again before each map
template <typename T>
static void set_default_inputs(IPM<T>& ipm)
{
	const IPM<T> d;
	ipm.kappa_tot = d.kappa_tot;
	ipm.shear = d.shear;
	ipm.kappa_star = d.kappa_star;
	ipm.theta_star = d.theta_star;
	ipm.mass_function_str = d.mass_function_str;
	ipm.m_solar = d.m_solar;
	ipm.m_lower = d.m_lower;
	ipm.m_upper = d.m_upper;
	ipm.rectangular = d.rectangular;
	ipm.approx = d.approx;
	ipm.safety_scale = d.safety_scale;
	ipm.starfile = d.starfile;
	ipm.random_seed = d.random_seed;
	ipm.write_stars = d.write_stars;
	ipm.outfile_prefix = d.outfile_prefix;
	ipm.light_loss = d.light_loss;
	ipm.center_y = d.center_y;
	ipm.half_length_y = d.half_length_y;
	ipm.num_pixels_y = d.num_pixels_y;
	ipm.num_rays_y = d.num_rays_y;
	ipm.write_maps = d.write_maps;
	ipm.write_parities = d.write_parities;
	ipm.write_histograms = d.write_histograms;
}

// map A: the 600x600 map with seed 11
template <typename T>
static void set_map_a(IPM<T>& ipm)
{
	set_default_inputs(ipm);
	configure(ipm);
	ipm.random_seed = 11;
}

// map B: 800x500 pixels, half-lengths 20 x 12.5, parities, Salpeter masses, rectangular star field, seed 12
template <typename T>
static void set_map_b(IPM<T>& ipm)
{
	set_default_inputs(ipm);
	configure(ipm);
	ipm.num_pixels_y = Complex<int>(800, 500);
	ipm.half_length_y = Complex<T>(static_cast<T>(20), static_cast<T>(12.5));
	ipm.write_parities = 1;
	ipm.mass_function_str = "salpeter";
	ipm.m_solar = static_cast<T>(1);
	ipm.rectangular = 1;
	ipm.approx = 1;
	ipm.random_seed = 12;
}

// the order of the stars inside the tree is not reproducible even for two default objects (the star sort
// places stars with atomics), so star fields are compared as sets: sorted by position and mass, bit for bit
template <typename T>
static std::vector<star<T>> sorted_stars(IPM<T>& ipm)
{
	std::vector<star<T>> v(ipm.get_stars(), ipm.get_stars() + ipm.get_num_stars());
	std::sort(v.begin(), v.end(), [](const star<T>& p, const star<T>& q)
	{
		if (p.position.re != q.position.re) return p.position.re < q.position.re;
		if (p.position.im != q.position.im) return p.position.im < q.position.im;
		return p.mass < q.mass;
	});
	return v;
}

template <typename T>
static bool same_stars(IPM<T>& a, IPM<T>& b)
{
	if (a.get_num_stars() != b.get_num_stars()) return false;
	const std::vector<star<T>> sa = sorted_stars(a);
	const std::vector<star<T>> sb = sorted_stars(b);
	return std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(star<T>)) == 0;
}

// map b matches map a within the loose bounds: |sum(b)/sum(a) - 1| < sum_tol (Tolerance::sum for a total map,
// Tolerance::parity_sum for a parity map), and fewer than 1e-3 of the pixels with b >= 1e-3 * mean(a) differ by
// more than Tolerance::diff relative to a
template <typename T>
static bool maps_match(const std::vector<T>& ma, const std::vector<T>& mb, std::string& detail, double sum_tol = Tolerance<T>::sum)
{
	if (ma.size() != mb.size() || ma.empty()) { detail = "maps of different sizes"; return false; }
	const size_t n = ma.size();
	double sa = 0, sb = 0;
	for (size_t i = 0; i < n; i++) { sa += ma[i]; sb += mb[i]; }
	const double mean_a = sa / n;
	const double sum_ratio = std::abs(sb / sa - 1);
	size_t considered = 0, bad = 0;
	for (size_t i = 0; i < n; i++)
	{
		if (!(mb[i] >= 1e-3 * mean_a)) continue;
		considered++;
		if (!(std::abs(1.0 * mb[i] - ma[i]) <= Tolerance<T>::diff * std::abs(1.0 * ma[i]))) bad++;
	}
	const double fraction = considered ? 1.0 * bad / considered : 1.0;
	detail = "sum ratio - 1 = " + std::to_string(sum_ratio) + ", fraction of differing pixels = " + std::to_string(fraction);
	return sum_ratio < sum_tol && fraction < 1e-3;
}

// copy one of the maps (0 total, 1 minima, 2 saddles) of an object that ran
template <typename T>
static bool copy_map(const IPM<T>& ipm, int which, std::vector<T>& out)
{
	out.assign(1ULL * ipm.num_pixels_y.re * ipm.num_pixels_y.im, T(0));
	if (which == 0) return ipm.copy_pixels(out.data());
	if (which == 1) return ipm.copy_pixels_minima(out.data());
	return ipm.copy_pixels_saddles(out.data());
}

template <typename T>
static void own_stream_on_last_device_matches_default()
{
	const std::string name = case_name<T>("own_stream_on_last_device_matches_default");
	int n_devices = 0;
	cudaGetDeviceCount(&n_devices);
	IPM<T> a;
	IPM<T> b;
	b.device = n_devices - 1;
	if (!b.create_stream()) { report(name, false, "create_stream failed"); return; }
	configure(a);
	configure(b);
	if (!a.run(0) || !b.run(0)) { report(name, false, "run failed"); return; }
	if (!same_stars(a, b)) { report(name, false, "star fields differ"); return; }

	std::vector<T> ma, mb;
	if (!copy_map(a, 0, ma) || !copy_map(b, 0, mb)) { report(name, false, "copy_pixels failed"); return; }
	std::string detail;
	const bool ok = maps_match(ma, mb, detail);
	report(name, ok, detail);
}

#ifndef IPM_BATCH_TEST_DOUBLE
static void set_cuda_devices_rejects_invalid_device()
{
	int n_devices = 0;
	cudaGetDeviceCount(&n_devices);
	IPM<float> c;
	configure(c);
	c.device = n_devices;
	const bool ran = c.run(0);
	IPM<float> d;
	configure(d);
	d.device = -1;
	const bool ran_negative = d.run(0);
	report("set_cuda_devices_rejects_invalid_device", !ran && !ran_negative);
}
#endif

// load the spin kernel (lazy module loading makes the first launch of a kernel wait for running work)
static void warm_up_spin()
{
	cudaSetDevice(0);
	launch_spin(0.001, 0);
	cudaDeviceSynchronize();
}

// a warmed-up worker runs a map with no cudaMalloc, cudaFree, device synchronization or legacy-stream call: each
// of them would wait for the spin kernel on the legacy stream. the brief's case is the 600x600 map; with parities,
// the worker runs map B (three maps, three min/max passes and the sum of the parity maps) and copies all three maps
template <typename T>
static void warm_worker_run_makes_no_device_sync(bool parities)
{
	const std::string name = case_name<T>(parities ? "warm_worker_parity_run_makes_no_device_sync" : "warm_worker_run_makes_no_device_sync");
	IPM<T> w;
	w.reuse_memory = 1;
	if (!w.create_stream()) { report(name, false, "create_stream failed"); return; }
	auto set_inputs = [&]() { if (parities) set_map_b(w); else configure(w); };
	const int n_maps = parities ? 3 : 1;
	set_inputs();
	if (!w.run(0)) { report(name, false, "warm-up run failed"); return; } //also loads the IPM kernels
	std::vector<T> warm_maps[3];
	for (int m = 0; m < n_maps; m++)
	{
		if (!copy_map(w, m, warm_maps[m])) { report(name, false, "a copy after the warm-up run failed"); return; }
	}
	std::vector<T> host_bufs[3];
	for (int m = 0; m < n_maps; m++) host_bufs[m].assign(warm_maps[m].size(), T(0));
	unsigned long long p0 = 0, s0 = 0, t0 = 0;
	w.allocated_bytes(p0, s0, t0);
	warm_up_spin();
	set_inputs();

	launch_spin(1.5, 0);
	const auto start = std::chrono::steady_clock::now();
	bool ok = w.run(0) && w.copy_pixels(host_bufs[0].data());
	if (parities) ok = ok && w.copy_pixels_minima(host_bufs[1].data()) && w.copy_pixels_saddles(host_bufs[2].data());
	const double t = seconds_since(start);
	cudaDeviceSynchronize();
	const double t_spin = seconds_since(start);

	unsigned long long p1 = 0, s1 = 0, t1 = 0;
	w.allocated_bytes(p1, s1, t1);
	const bool same_buffers = p0 == p1 && s0 == s1 && t0 == t1;
	if (!same_buffers)
	{
		std::printf("INFO %s: bytes after the warm-up run %llu %llu %llu, after the timed run %llu %llu %llu\n",
			name.c_str(), p0, s0, t0, p1, s1, t1);
	}
	std::printf("INFO %s: run + copies %.3f s while the spin kernel ran (spin ended after %.3f s)\n", name.c_str(), t, t_spin);
	//the timed run made the same maps as the warm-up run
	std::string detail;
	bool same_maps = ok;
	for (int m = 0; m < n_maps && same_maps; m++)
	{
		same_maps = maps_match(warm_maps[m], host_bufs[m], detail, m == 0 ? Tolerance<T>::sum : Tolerance<T>::parity_sum);
		if (!same_maps) detail = "map " + std::to_string(m) + ": " + detail;
	}
	report(name, ok && t < 0.5 && same_buffers && same_maps,
		!ok ? "run or a copy failed" : !same_buffers ? "the buffers changed in the second run"
		: !same_maps ? "the timed run made a different map: " + detail
		: "run + copies took " + std::to_string(t) + " s");
}

// one worker runs A, B, A; every result matches a fresh default object given the same inputs
template <typename T>
static void reused_worker_matches_fresh_objects()
{
	const std::string name = case_name<T>("reused_worker_matches_fresh_objects");
	int n_devices = 0;
	cudaGetDeviceCount(&n_devices);
	IPM<T> w;
	w.device = n_devices - 1;
	w.reuse_memory = 1;
	if (!w.create_stream()) { report(name, false, "create_stream failed"); return; }

	const char* labels[3] = {"A", "B", "A again"};
	for (int k = 0; k < 3; k++)
	{
		const bool is_b = (k == 1);
		IPM<T> f;
		if (is_b) { set_map_b(w); set_map_b(f); }
		else { set_map_a(w); set_map_a(f); }
		const std::string label = labels[k];
		if (!w.run(0)) { report(name, false, label + ": worker run failed"); return; }
		if (!f.run(0)) { report(name, false, label + ": fresh run failed"); return; }
		if (!same_stars(w, f)) { report(name, false, label + ": star fields differ"); return; }
		const int n_maps = is_b ? 3 : 1;
		for (int m = 0; m < n_maps; m++)
		{
			std::vector<T> mw, mf;
			if (!copy_map(w, m, mw) || !copy_map(f, m, mf)) { report(name, false, label + ": map copy failed"); return; }
			std::string detail;
			if (!maps_match(mf, mw, detail, m == 0 ? Tolerance<T>::sum : Tolerance<T>::parity_sum))
			{
				report(name, false, label + ", map " + std::to_string(m) + ": " + detail);
				return;
			}
		}
		//without parities the worker has no parity maps to give, although it keeps the buffers of map B
		std::vector<T> none;
		if (!is_b && (copy_map(w, 1, none) || copy_map(w, 2, none)))
		{
			report(name, false, label + ": the worker returned a parity map of an earlier run");
			return;
		}
	}

	//nor does it save them, or their histograms, when write_parities is switched on after the run: save() fails
	//as it does for a fresh object
	char tmpl[] = "/tmp/test_ipm_batch_XXXXXX";
	if (mkdtemp(tmpl) == nullptr) { report(name, false, "mkdtemp failed"); return; }
	const std::string dir = tmpl;
	IPM<T> f;
	set_map_a(f);
	if (!f.run(0)) { report(name, false, "fresh run for save failed"); return; }
	//no CUDA error either: a stale buffer would be read (out of bounds, if the last histogram is longer)
	bool saved[2] = {true, true};
	int cuda_errors[2] = {0, 0};
	bool minima_files[2] = {true, true};
	IPM<T>* objects[2] = {&w, &f};
	for (int k = 0; k < 2; k++)
	{
		objects[k]->write_parities = 1;
		objects[k]->write_maps = 0;
		objects[k]->outfile_prefix = dir + "/" + std::to_string(k) + "_";
		std::fprintf(stderr, "(%s: the next error is expected)\n", name.c_str());
		take_last_cuda_error();
		saved[k] = objects[k]->save(0);
		cuda_errors[k] = take_last_cuda_error();
		minima_files[k] = std::filesystem::exists(objects[k]->outfile_prefix + "ipm_mags_numpixels_minima.txt");
	}
	std::filesystem::remove_all(dir);
	for (int k = 0; k < 2; k++)
	{
		if (saved[k] || cuda_errors[k] != 0 || minima_files[k])
		{
			report(name, false, std::string("save() with parities switched on after the run, ") + (k == 0 ? "worker" : "fresh object") + ": "
				+ (saved[k] ? "returned true" : cuda_errors[k] != 0 ? "CUDA error " + std::to_string(cuda_errors[k]) : "wrote a minima histogram"));
			return;
		}
	}
	report(name, true);
}

// a default object frees its step buffer after shooting, as before: its pixel memory is the map alone, and its
// star memory is exactly the curand states, the two star arrays and the binomial coefficients
template <typename T>
static void default_object_keeps_todays_footprint()
{
	const std::string name = case_name<T>("default_object_keeps_todays_footprint");
	IPM<T> d;
	configure(d);
	d.write_histograms = 0;
	const unsigned long long map_bytes = 600ULL * 600ULL * sizeof(T);
	const unsigned long long binomial_bytes = (2 * treenode::MAX_EXPANSION_ORDER * (2 * treenode::MAX_EXPANSION_ORDER + 3) / 2 + 1) * sizeof(long long);
	for (int k = 0; k < 2; k++)
	{
		if (!d.run(0)) { report(name, false, "run " + std::to_string(k + 1) + " failed"); return; }
		unsigned long long p = 0, s = 0, t = 0;
		d.allocated_bytes(p, s, t);
		const unsigned long long star_bytes = 1ULL * d.get_num_stars() * (sizeof(curandState) + 2 * sizeof(star<T>)) + binomial_bytes;
		if (p != map_bytes || s != star_bytes || t == 0)
		{
			report(name, false, "run " + std::to_string(k + 1) + ": pixel " + std::to_string(p) + " (expected " + std::to_string(map_bytes)
				+ "), star " + std::to_string(s) + " (expected " + std::to_string(star_bytes) + "), tree " + std::to_string(t));
			return;
		}
	}
	report(name, true);
}

// creating and destroying an object that never ran makes no CUDA call that waits for the device
template <typename T>
static void default_object_makes_no_cuda_call()
{
	const std::string name = case_name<T>("default_object_makes_no_cuda_call");
	launch_spin(1.0, 0);
	const auto start = std::chrono::steady_clock::now();
	{
		IPM<T> x;
	}
	const double t = seconds_since(start);
	cudaDeviceSynchronize();
	std::printf("INFO %s: %.6f s\n", name.c_str(), t);
	report(name, t < 0.1, "creating and destroying took " + std::to_string(t) + " s");
}

// release_memory() gives every buffer back, and the worker runs again afterwards
template <typename T>
static void release_memory_frees_everything()
{
	const std::string name = case_name<T>("release_memory_frees_everything");
	IPM<T> w;
	w.reuse_memory = 1;
	if (!w.create_stream()) { report(name, false, "create_stream failed"); return; }
	configure(w);
	w.write_parities = 1;
	if (!w.run(0)) { report(name, false, "first run failed"); return; }
	unsigned long long p = 0, s = 0, t = 0;
	w.allocated_bytes(p, s, t);
	if (p == 0 || s == 0 || t == 0) { report(name, false, "a component is 0 after a run"); return; }
	if (!w.release_memory()) { report(name, false, "release_memory failed"); return; }
	w.allocated_bytes(p, s, t);
	if (p != 0 || s != 0 || t != 0)
	{
		report(name, false, "after release_memory: pixel " + std::to_string(p) + ", star " + std::to_string(s) + ", tree " + std::to_string(t));
		return;
	}
	std::vector<T> m;
	if (copy_map(w, 0, m)) { report(name, false, "a map is still there after release_memory"); return; }
	w.write_parities = 0;
	const bool again = w.run(0) && copy_map(w, 0, m);
	report(name, again, "run after release_memory failed");
}

// thrust allocator that counts the calls and the live allocations, checks that no two live allocations overlap,
// and passes them on to a ThrustScratch
struct CountingScratch
{
	using value_type = char;

	ThrustScratch* scratch;
	int allocations = 0;
	int deallocations = 0;
	int max_live = 0;
	bool overlap = false;
	bool outside = false; //an allocation that does not lie in the scratch buffer (it was allocated separately)
	std::vector<std::pair<char*, size_t>> live;

	char* allocate(std::ptrdiff_t n)
	{
		char* p = scratch->allocate(n);
		for (const auto& q : live)
		{
			if (p < q.first + q.second && q.first < p + n) overlap = true;
		}
		if (!(p >= scratch->ptr && p + n <= scratch->ptr + scratch->cap)) outside = true;
		live.push_back({p, static_cast<size_t>(n)});
		allocations++;
		max_live = std::max(max_live, static_cast<int>(live.size()));
		return p;
	}

	void deallocate(char* p, size_t n)
	{
		deallocations++;
		for (size_t i = 0; i < live.size(); i++)
		{
			if (live[i].first == p) { live.erase(live.begin() + i); break; }
		}
		scratch->deallocate(p, n);
	}
};

// the min/max pass of a reused IPM object hands minmax_element a ThrustScratch, reset before each call.
// minmax_element holds more than one temporary allocation at once, so the parts it gets must not overlap, all of
// them must be given back before it returns, and after the first call they must all come from the scratch buffer
// without growing it (no cudaMalloc). the first call's counts are printed
template <typename T>
static void thrust_minmax_scratch_allocations()
{
	const std::string name = case_name<T>("thrust_minmax_scratch_allocations");
	const size_t n = 600ULL * 600ULL;
	std::vector<T> h(n);
	for (size_t i = 0; i < n; i++) h[i] = static_cast<T>(1 + (i * 7919 % 1000) * 1e-3);
	const size_t i_min = 123457, i_max = 300001;
	h[i_min] = static_cast<T>(0.25);
	h[i_max] = static_cast<T>(9.5);

	cudaSetDevice(0);
	T* d = nullptr;
	cudaStream_t s = 0;
	if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess || cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess
		|| cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		report(name, false, "setup failed");
		return;
	}
	//a cudaMemcpy from pageable memory can return before the data has arrived, and s does not wait for the legacy stream
	cudaDeviceSynchronize();
	ThrustScratch scratch;
	thrust::device_ptr<T> dp = thrust::device_pointer_cast(d);
	bool ok = true;
	std::string detail;
	size_t cap_after_first = 0;
	for (int k = 1; k <= 3 && ok; k++)
	{
		CountingScratch counting{&scratch};
		scratch.reset();
		auto mm = thrust::minmax_element(thrust::cuda::par_nosync(counting).on(s), dp, dp + n);
		const bool right = static_cast<size_t>(mm.first - dp) == i_min && static_cast<size_t>(mm.second - dp) == i_max;
		if (k == 1) cap_after_first = scratch.cap;
		ok = right && !counting.overlap && counting.live.empty() && counting.allocations == counting.deallocations
			&& !scratch.in_use() && (k == 1 || (!counting.outside && scratch.cap == cap_after_first));
		detail = "call " + std::to_string(k) + ": " + std::to_string(counting.allocations) + " allocations, "
			+ std::to_string(counting.deallocations) + " deallocations, at most " + std::to_string(counting.max_live)
			+ " live, " + std::to_string(counting.live.size()) + " live after return, " + (counting.overlap ? "" : "no ")
			+ "overlap, " + (counting.outside ? "" : "none ") + "outside the scratch buffer, scratch " + std::to_string(scratch.cap)
			+ " bytes, result " + (right ? "right" : "wrong");
		if (k == 1 || !ok) std::printf("INFO %s: %s\n", name.c_str(), detail.c_str());
	}
	ok = scratch.release() && ok;
	cudaStreamDestroy(s);
	cudaFree(d);
	report(name, ok, detail);
}

// a worker with every write_* flag set, running the 600x600 map; w.device is the last visible device
template <typename T>
static bool make_snapshot_worker(IPM<T>& w)
{
	int n_devices = 0;
	cudaGetDeviceCount(&n_devices);
	w.device = n_devices - 1;
	w.reuse_memory = 1;
	if (!w.create_stream()) return false;
	set_default_inputs(w);
	configure(w);
	w.write_stars = 1;
	w.write_maps = 1;
	w.write_parities = 1;
	w.write_histograms = 1;
	return w.run(0);
}

static std::string slurp(const std::filesystem::path& p)
{
	std::ifstream f(p, std::ios::binary);
	return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// the three maps of a worker, copied for save_host
template <typename T>
static bool copy_three_maps(const IPM<T>& w, std::vector<T> maps[3])
{
	for (int m = 0; m < 3; m++)
	{
		if (!copy_map(w, m, maps[m])) return false;
	}
	return true;
}

// save() of a worker and save_host() of its snapshot write the same files, byte for byte (one run's data against
// itself, so the star order is the same). 11 files: the parameter info (one file), the stars, three maps, six histograms
template <typename T>
static void snapshot_save_host_matches_save()
{
	const std::string name = case_name<T>("snapshot_save_host_matches_save");
	IPM<T> w;
	if (!make_snapshot_worker(w)) { report(name, false, "worker run failed"); return; }
	char tmpl[] = "/tmp/ipm_snapshot_XXXXXX";
	if (mkdtemp(tmpl) == nullptr) { report(name, false, "mkdtemp failed"); return; }
	const std::string dir = tmpl;
	std::filesystem::create_directory(dir + "/a");
	std::filesystem::create_directory(dir + "/b");
	w.outfile_prefix = dir + "/a/";
	bool ok = w.save(0);
	std::string detail = ok ? "" : "save failed";
	IPM<T>* s = ok ? w.snapshot() : nullptr;
	std::vector<T> maps[3];
	if (ok && (s == nullptr || !copy_three_maps(w, maps))) { ok = false; detail = "snapshot or map copy failed"; }
	if (ok)
	{
		s->outfile_prefix = dir + "/b/";
		ok = s->save_host(maps[0].data(), maps[1].data(), maps[2].data(), 0);
		if (!ok) detail = "save_host failed";
	}
	if (ok)
	{
		int n_a = 0, n_b = 0;
		for (const auto& e : std::filesystem::directory_iterator(dir + "/b")) { n_b++; (void)e; }
		for (const auto& e : std::filesystem::directory_iterator(dir + "/a"))
		{
			n_a++;
			const auto twin = std::filesystem::path(dir + "/b") / e.path().filename();
			if (!std::filesystem::exists(twin) || slurp(e.path()) != slurp(twin))
			{
				ok = false;
				detail = "file " + e.path().filename().string() + " differs or is missing";
				break;
			}
		}
		if (ok && (n_a != n_b || n_a != 11)) { ok = false; detail = "file counts " + std::to_string(n_a) + " and " + std::to_string(n_b); }
	}
	delete s;
	std::filesystem::remove_all(dir);
	report(name, ok, detail);
}

// creating a snapshot and saving from it must not wait for the spin kernel on the legacy stream of device 0
template <typename T>
static void snapshot_and_save_host_make_no_device_sync()
{
	const std::string name = case_name<T>("snapshot_and_save_host_make_no_device_sync");
	IPM<T> w;
	if (!make_snapshot_worker(w)) { report(name, false, "worker run failed"); return; }
	std::vector<T> maps[3];
	if (!copy_three_maps(w, maps)) { report(name, false, "map copy failed"); return; }
	char tmpl[] = "/tmp/ipm_snapshot_XXXXXX";
	if (mkdtemp(tmpl) == nullptr) { report(name, false, "mkdtemp failed"); return; }
	const std::string dir = tmpl;
	//the first snapshot, outside the timed region, loads whatever the copies need
	{
		IPM<T>* warm = w.snapshot();
		delete warm;
	}
	warm_up_spin();
	launch_spin(1.5, 0);
	const auto start = std::chrono::steady_clock::now();
	IPM<T>* s2 = w.snapshot();
	bool ok = s2 != nullptr;
	if (ok)
	{
		s2->outfile_prefix = dir + "/";
		ok = s2->save_host(maps[0].data(), maps[1].data(), maps[2].data(), 0);
	}
	delete s2;
	const double t = seconds_since(start);
	cudaDeviceSynchronize();
	std::filesystem::remove_all(dir);
	std::printf("INFO %s: snapshot + save_host + delete %.3f s while the spin kernel ran\n", name.c_str(), t);
	report(name, ok && t < 0.5, !ok ? "snapshot or save_host failed" : "took " + std::to_string(t) + " s");
}

template <typename T>
static void snapshot_getters_match()
{
	const std::string name = case_name<T>("snapshot_getters_match");
	IPM<T> w;
	if (!make_snapshot_worker(w)) { report(name, false, "worker run failed"); return; }
	w.mass_function_str = "salpeter";
	w.m_lower = static_cast<T>(0.1);
	w.m_solar = static_cast<T>(0.5);
	if (!w.run(0)) { report(name, false, "second run failed"); return; }
	IPM<T>* s = w.snapshot();
	if (s == nullptr) { report(name, false, "snapshot failed"); return; }
	const Complex<T> cw = w.get_corner(), cs = s->get_corner();
	const star<T>* sw = w.get_stars();
	const star<T>* ss = s->get_stars();
	const bool stars_ok = sw != nullptr && ss != nullptr && ss != sw
		&& std::memcmp(sw, ss, 1ULL * w.get_num_stars() * sizeof(star<T>)) == 0;
	const bool ok = s->get_num_stars() == w.get_num_stars() && cw.re == cs.re && cw.im == cs.im
		&& s->get_t_shoot_cells() == w.get_t_shoot_cells() && s->random_seed == w.random_seed
		&& s->kappa_star == w.kappa_star && s->m_lower == w.m_lower && s->device == w.device
		&& s->get_perimeter_deposit_used() == w.get_perimeter_deposit_used();
	delete s;
	report(name, ok && stars_ok, !stars_ok ? "the snapshot's stars differ from the worker's" : "a getter or input differs");
}

template <typename T>
static void reset_inputs_restores_defaults()
{
	const std::string name = case_name<T>("reset_inputs_restores_defaults");
	IPM<T> w;
	w.reuse_memory = 1;
	w.device = 0;
	w.kappa_tot = 0.9; w.shear = 0.1; w.kappa_star = 0.5; w.theta_star = 2;
	w.mass_function_str = "kroupa"; w.m_solar = 3; w.m_lower = 0.2; w.m_upper = 20;
	w.rectangular = 1; w.approx = 0; w.safety_scale = 2; w.starfile = "x.bin"; w.random_seed = 99;
	w.write_stars = 0; w.outfile_prefix = "/nowhere/";
	w.light_loss = 0.005; w.center_y = Complex<T>(1, 2); w.half_length_y = Complex<T>(3, 4);
	w.num_pixels_y = Complex<int>(7, 9); w.num_rays_y = 5;
	w.write_maps = 0; w.write_parities = 1; w.write_histograms = 0;
	w.reset_inputs();
	const IPM<T> d;
	const bool ok = w.kappa_tot == d.kappa_tot && w.shear == d.shear && w.kappa_star == d.kappa_star
		&& w.theta_star == d.theta_star && w.mass_function_str == d.mass_function_str && w.m_solar == d.m_solar
		&& w.m_lower == d.m_lower && w.m_upper == d.m_upper && w.rectangular == d.rectangular && w.approx == d.approx
		&& w.safety_scale == d.safety_scale && w.starfile == d.starfile && w.random_seed == d.random_seed
		&& w.write_stars == d.write_stars && w.outfile_prefix == d.outfile_prefix && w.light_loss == d.light_loss
		&& w.center_y.re == d.center_y.re && w.center_y.im == d.center_y.im
		&& w.half_length_y.re == d.half_length_y.re && w.half_length_y.im == d.half_length_y.im
		&& w.num_pixels_y.re == d.num_pixels_y.re && w.num_pixels_y.im == d.num_pixels_y.im
		&& w.num_rays_y == d.num_rays_y && w.write_maps == d.write_maps && w.write_parities == d.write_parities
		&& w.write_histograms == d.write_histograms;
	report(name, ok && w.reuse_memory == 1 && w.device == 0, ok ? "reuse_memory or device changed" : "an input is not the default");
}

template <typename T>
static void save_host_refuses_non_snapshot()
{
	const std::string name = case_name<T>("save_host_refuses_non_snapshot");
	IPM<T> w;
	if (!make_snapshot_worker(w)) { report(name, false, "worker run failed"); return; }
	std::vector<T> maps[3];
	if (!copy_three_maps(w, maps)) { report(name, false, "map copy failed"); return; }
	char tmpl[] = "/tmp/ipm_snapshot_XXXXXX";
	if (mkdtemp(tmpl) == nullptr) { report(name, false, "mkdtemp failed"); return; }
	const std::string dir = tmpl;
	w.outfile_prefix = dir + "/";
	const bool refused = !w.save_host(maps[0].data(), nullptr, nullptr, 0);
	//a snapshot refuses a missing map too
	IPM<T>* s = w.snapshot();
	bool refused_null = false;
	if (s != nullptr)
	{
		s->outfile_prefix = dir + "/";
		refused_null = !s->save_host(maps[0].data(), nullptr, maps[2].data(), 0);
	}
	delete s;
	std::filesystem::remove_all(dir);
	report(name, refused && refused_null, refused ? "a snapshot accepted a null map" : "a non-snapshot saved");
}

// shoot_cells sets the flag: the perimeter deposit is used by default and not when forced to the direct deposit
template <typename T>
static void perimeter_flag()
{
	const std::string name = case_name<T>("perimeter_flag");
	IPM<T> w;
	w.reuse_memory = 1;
	if (!w.create_stream()) { report(name, false, "create_stream failed"); return; }
	configure(w);
	bool ok = w.run(0);
	const bool normal = ok && w.get_perimeter_deposit_used();
	IPM<T>::testing_force_direct_deposit = true;
	configure(w);
	ok = ok && w.run(0);
	const bool forced_off = ok && !w.get_perimeter_deposit_used();
	IPM<T>::testing_force_direct_deposit = false;
	report(name, normal && forced_off, !ok ? "a run failed" : !normal ? "flag not set by a normal run" : "flag set by a forced direct deposit run");
}

template <typename T>
static void run_cases()
{
	own_stream_on_last_device_matches_default<T>();
	thrust_minmax_scratch_allocations<T>();
	default_object_keeps_todays_footprint<T>();
	release_memory_frees_everything<T>();
	reused_worker_matches_fresh_objects<T>();
	warm_up_spin();
	warm_worker_run_makes_no_device_sync<T>(false);
	warm_worker_run_makes_no_device_sync<T>(true);
	default_object_makes_no_cuda_call<T>();
	snapshot_save_host_matches_save<T>();
	snapshot_getters_match<T>();
	reset_inputs_restores_defaults<T>();
	save_host_refuses_non_snapshot<T>();
	perimeter_flag<T>();
	warm_up_spin();
	snapshot_and_save_host_make_no_device_sync<T>();
}

int main()
{
#ifdef IPM_BATCH_TEST_DOUBLE
	run_cases<double>();
#else
	run_cases<float>();
	set_cuda_devices_rejects_invalid_device();
#endif
	return n_fail ? 1 : 0;
}
