// Tests for the map accessors of IPM<T> (include/ipm.cuh): copy_pixels*(), the host copies behind get_pixels*(),
// and save() streaming the maps and histograms from device memory. Needs the GPU; the maps are 200x200 (200x120
// for the save case), each run takes well under a second.
// Prints one "PASS <name>" or "FAIL <name>: <detail>" line per case and exits 1 if any case fails.
#include "ipm.cuh"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <new>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

// the global operator new, replaced so that a test can make large allocations fail: while fail_new_from is not 0,
// an allocation of at least that many bytes throws std::bad_alloc and is counted in n_failed_new. host code only:
// nvcc compiles an operator new for the device as well, hence the guard
[[maybe_unused]] static size_t fail_new_from = 0;
[[maybe_unused]] static long n_failed_new = 0;
#ifndef __CUDA_ARCH__
void* operator new(std::size_t n)
{
	if (fail_new_from != 0 && n >= fail_new_from)
	{
		n_failed_new++;
		throw std::bad_alloc();
	}
	if (void* p = std::malloc(n ? n : 1)) return p;
	throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
#endif

static int n_fail = 0;
static void report(const std::string& name, bool ok, const std::string& detail = "")
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
	std::fflush(stdout);
}

using Map = std::vector<float>;

struct Access
{
	const char* name;
	bool (IPM<float>::*copy)(float*) const;
	float* (IPM<float>::*get)();
};
static const Access MAPS[3] = {
	{"pixels", &IPM<float>::copy_pixels, &IPM<float>::get_pixels},
	{"minima", &IPM<float>::copy_pixels_minima, &IPM<float>::get_pixels_minima},
	{"saddles", &IPM<float>::copy_pixels_saddles, &IPM<float>::get_pixels_saddles},
};

// the lens of the brief: kappa 0.4, shear 0.4, kappa_star 0.2, half-lengths 5, 200x200 pixels
static void configure(IPM<float>& ipm, int seed, int parities)
{
	ipm.kappa_tot = 0.4f;
	ipm.shear = 0.4f;
	ipm.kappa_star = 0.2f;
	ipm.half_length_y = Complex<float>(5, 5);
	ipm.num_pixels_y = Complex<int>(200, 200);
	ipm.random_seed = seed;
	ipm.write_parities = parities;
	ipm.write_stars = 0;
}

static size_t num_pixels(const IPM<float>& ipm)
{
	return 1ULL * ipm.num_pixels_y.re * ipm.num_pixels_y.im;
}

// copy_<map>() into out (sized to the map and prefilled with NaN); returns what the copy returned
static bool copy_of(const IPM<float>& ipm, const Access& m, Map& out)
{
	out.assign(num_pixels(ipm), std::nanf(""));
	return (ipm.*m.copy)(out.data());
}

// index of the first element where a and b differ, or -1
static long long first_difference(const float* a, const float* b, size_t n)
{
	for (size_t i = 0; i < n; i++)
	{
		if (!(a[i] == b[i])) return static_cast<long long>(i);
	}
	return -1;
}

static std::vector<char> slurp(const std::string& f)
{
	std::ifstream in(f, std::ios::binary);
	return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// "value count" lines of a histogram file
static std::map<int, long long> read_histogram(const std::string& f, bool& ok)
{
	std::map<int, long long> h;
	std::ifstream in(f);
	ok = in.is_open();
	int value;
	long long count;
	while (in >> value >> count) h[value] += count;
	return h;
}

static void copy_matches_get_pixels(IPM<float>& ipm, bool ran, Map first[3])
{
	Map copies[3];
	for (int k = 0; k < 3; k++)
	{
		const Access& m = MAPS[k];
		std::string name = std::string("copy_matches_get_pixels_") + m.name;
		bool copied = copy_of(ipm, m, copies[k]);
		float* g = (ipm.*m.get)();
		if (!ran || !copied || g == nullptr)
		{
			report(name, false, !ran ? "run() returned false" : !copied ? "copy returned false" : "get returned nullptr");
			continue;
		}
		long long d = first_difference(copies[k].data(), g, copies[k].size());
		bool positive = std::any_of(copies[k].begin(), copies[k].end(), [](float v) { return v > 0; });
		bool same_copy = (ipm.*m.get)() == g; //later calls return the copy made by the first
		report(name, d < 0 && positive && same_copy,
			d >= 0 ? "differs at " + std::to_string(d) : !positive ? "no positive value" : "second call returned another pointer");
		first[k].assign(g, g + copies[k].size());
	}

	//with the step buffers, d_pixels holds the minima's steps until it is overwritten by minima + saddles
	long long bad = ran ? -1 : 0;
	for (size_t i = 0; bad < 0 && i < copies[0].size(); i++)
	{
		if (!(copies[0][i] == copies[1][i] + copies[2][i])) bad = static_cast<long long>(i);
	}
	report("copy_total_is_minima_plus_saddles", bad < 0, "differs at " + std::to_string(bad));
}

static void run_twice_refreshes_host_copy(IPM<float>& ipm, const Map first[3])
{
	ipm.random_seed = 2;
	bool ran = ipm.run(0);
	for (int k = 0; k < 3; k++)
	{
		const Access& m = MAPS[k];
		std::string name = std::string("run_twice_refreshes_host_copy_") + m.name;
		Map c;
		bool copied = copy_of(ipm, m, c);
		float* g = (ipm.*m.get)();
		if (!ran || !copied || g == nullptr || first[k].size() != c.size())
		{
			report(name, false, !ran ? "run() returned false" : !copied ? "copy returned false" : g == nullptr ? "get returned nullptr" : "no first map");
			continue;
		}
		long long d = first_difference(c.data(), g, c.size());
		bool changed = first_difference(first[k].data(), g, c.size()) >= 0;
		report(name, d < 0 && changed, d >= 0 ? "host copy differs from copy_ at " + std::to_string(d) : "host copy still holds the seed 1 map");
	}
}

static void check_no_parity_maps(IPM<float>& ipm, bool ran, const std::string& name)
{
	Map c, minima, saddles;
	bool total = copy_of(ipm, MAPS[0], c);
	float* g = ipm.get_pixels();
	bool copied_minima = copy_of(ipm, MAPS[1], minima);
	bool copied_saddles = copy_of(ipm, MAPS[2], saddles);
	bool null_minima = ipm.get_pixels_minima() == nullptr;
	bool null_saddles = ipm.get_pixels_saddles() == nullptr;
	bool ok = ran && total && g != nullptr && first_difference(c.data(), g, c.size()) < 0
		&& !copied_minima && !copied_saddles && null_minima && null_saddles;
	report(name, ok, !ran ? "run() returned false" : !total || g == nullptr ? "total map missing or different"
		: copied_minima || copied_saddles ? "a parity copy returned true" : "a parity host copy is not nullptr");
}

static void minima_without_parities_is_false(IPM<float>& reused)
{
	//an object that made parity maps and their host copies before
	reused.write_parities = 0;
	reused.random_seed = 1;
	check_no_parity_maps(reused, reused.run(0), "minima_without_parities_is_false_after_parity_run");

	IPM<float> fresh;
	configure(fresh, 1, 0);
	check_no_parity_maps(fresh, fresh.run(0), "minima_without_parities_is_false");
}

static void no_map_before_run()
{
	IPM<float> ipm;
	configure(ipm, 1, 1);
	bool ok = true;
	for (const Access& m : MAPS)
	{
		Map c;
		ok = ok && !copy_of(ipm, m, c) && (ipm.*m.get)() == nullptr;
	}
	report("no_map_before_run", ok, "a copy returned true or a host copy is not nullptr");
}

static void copies_do_not_share_maps(IPM<float>& ipm)
{
	bool ok = true;
	{
		IPM<float> copied(ipm);
		IPM<float> assigned;
		assigned = ipm;
		for (const Access& m : MAPS)
		{
			Map c;
			ok = ok && !copy_of(copied, m, c) && (copied.*m.get)() == nullptr;
			ok = ok && !copy_of(assigned, m, c) && (assigned.*m.get)() == nullptr;
		}
	}
	//the original still has its map after the copies are destroyed
	Map c;
	float* g = ipm.get_pixels();
	bool original = copy_of(ipm, MAPS[0], c) && g != nullptr && first_difference(c.data(), g, c.size()) < 0;
	report("copies_do_not_share_maps", ok && original, !ok ? "a copied object has a map" : "the original lost its map");
}

static void save_streams_maps_and_histograms()
{
	char tmpl[] = "/tmp/test_ipm_api_XXXXXX";
	if (mkdtemp(tmpl) == nullptr)
	{
		report("save_streams_maps_and_histograms", false, "mkdtemp failed");
		return;
	}
	const std::string dir = tmpl;

	IPM<float> ipm;
	configure(ipm, 1, 1);
	ipm.half_length_y = Complex<float>(5, 3);
	ipm.num_pixels_y = Complex<int>(200, 120); //rows = num_pixels_y2 = 120, columns = num_pixels_y1 = 200
	ipm.write_maps = 1;
	ipm.write_histograms = 1;
	ipm.outfile_prefix = dir + "/";
	bool ran = ipm.run(0);
	bool saved = ran && ipm.save(0);
	if (!saved)
	{
		report("save_streams_maps_and_histograms", false, ran ? "save() returned false" : "run() returned false");
		std::filesystem::remove_all(dir);
		return;
	}

	const char* suffix[3] = {"", "_minima", "_saddles"};
	Map maps[3];
	int min_mag = INT_MAX; //as create_histograms: the smallest round(min * 1000) of the three maps, in T arithmetic
	for (int k = 0; k < 3; k++)
	{
		bool copied = copy_of(ipm, MAPS[k], maps[k]);
		std::string name = std::string("save_streams_map_") + MAPS[k].name;
		std::vector<char> file = slurp(dir + "/ipm_magnifications" + suffix[k] + ".bin");
		std::vector<char> expected(2 * sizeof(int) + maps[k].size() * sizeof(float));
		const int header[2] = {120, 200};
		std::memcpy(expected.data(), header, sizeof(header));
		std::memcpy(expected.data() + sizeof(header), maps[k].data(), maps[k].size() * sizeof(float));
		report(name, copied && file == expected, !copied ? "copy returned false" : "file is " + std::to_string(file.size())
			+ " bytes, expected " + std::to_string(expected.size()) + " (or same size, different bytes)");
		float lo = *std::min_element(maps[k].begin(), maps[k].end());
		min_mag = std::min(min_mag, static_cast<int>(std::round(lo * 1000)));
	}

	for (int k = 0; k < 3; k++)
	{
		//histogram_kernel's binning, which is exact in double arithmetic
		std::map<int, long long> expected;
		for (float v : maps[k]) expected[static_cast<int>(std::round(1.0 * v * 1000 - min_mag)) + min_mag]++;
		bool opened;
		std::map<int, long long> got = read_histogram(dir + "/ipm_mags_numpixels" + suffix[k] + ".txt", opened);
		report(std::string("save_streams_histogram_") + MAPS[k].name, opened && got == expected,
			!opened ? "no file" : std::to_string(got.size()) + " bins, expected " + std::to_string(expected.size()));

		//log10 differs between host and device, so only the pixel count is checked
		std::map<int, long long> got_log = read_histogram(dir + "/ipm_log_mags_numpixels" + suffix[k] + ".txt", opened);
		long long total = 0;
		for (auto& kv : got_log) total += kv.second;
		report(std::string("save_streams_log_histogram_") + MAPS[k].name, opened && total == static_cast<long long>(maps[k].size()),
			!opened ? "no file" : "counts sum to " + std::to_string(total));
	}

	std::filesystem::remove_all(dir);
}

// a failed host allocation in save() (the copy of a histogram, the only allocation there above 64 KiB) makes it
// return false rather than throw through the C interface, and a later save() works
static void save_returns_false_on_a_failed_allocation()
{
	char tmpl[] = "/tmp/test_ipm_api_XXXXXX";
	if (mkdtemp(tmpl) == nullptr)
	{
		report("save_returns_false_on_a_failed_allocation", false, "mkdtemp failed");
		return;
	}
	const std::string dir = tmpl;
	IPM<float> ipm;
	configure(ipm, 1, 0);
	ipm.write_histograms = 1;
	ipm.outfile_prefix = dir + "/";
	bool ran = ipm.run(0);
	bool threw = false, saved = true;
	n_failed_new = 0;
	fail_new_from = 64 << 10;
	try
	{
		saved = ipm.save(0);
	}
	catch (const std::bad_alloc&)
	{
		threw = true;
	}
	fail_new_from = 0;
	bool again = ran && ipm.save(0);
	bool opened;
	read_histogram(dir + "/ipm_mags_numpixels.txt", opened);
	report("save_returns_false_on_a_failed_allocation", ran && !threw && !saved && n_failed_new > 0 && again && opened,
		!ran ? "run() returned false" : threw ? "std::bad_alloc escaped save()" : saved ? "save() returned true"
		: n_failed_new == 0 ? "no allocation was made to fail" : !again ? "the later save() returned false" : "no histogram file");
	std::filesystem::remove_all(dir);
}

int main()
{
	no_map_before_run();

	IPM<float> ipm;
	configure(ipm, 1, 1);
	Map first[3];
	copy_matches_get_pixels(ipm, ipm.run(0), first);
	run_twice_refreshes_host_copy(ipm, first);
	copies_do_not_share_maps(ipm);
	minima_without_parities_is_false(ipm);
	save_streams_maps_and_histograms();
	save_returns_false_on_a_failed_allocation();

	std::printf("%s: %d failed\n", n_fail ? "FAILED" : "OK", n_fail);
	return n_fail ? 1 : 0;
}
