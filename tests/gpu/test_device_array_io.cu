// Tests for DeviceArrayWriter and write_device_array (include/device_array_io.cuh).
// Prints one "PASS <name>" or "FAIL <name>: <detail>" line per case.
#include "array_functions.cuh"
#include "device_array_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <utility>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int n_fail = 0;
static void report(const std::string& name, bool ok, const std::string& detail = "")
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
}

static std::vector<char> slurp(const std::string& f)
{
	std::ifstream in(f, std::ios::binary);
	return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

template <typename T>
static void run(const char* tname, const std::string& dir)
{
	const int shapes[3][2] = {{1, 1}, {3, 5}, {1000, 1001}};
	const size_t chunks[3] = {64, 4096, 64ULL << 20};
	for (auto& s : shapes)
		for (size_t ck : chunks)
		{
			int nr = s[0], nc = s[1];
			size_t n = static_cast<size_t>(nr) * nc;
			std::vector<T> h(n);
			for (size_t i = 0; i < n; i++) h[i] = static_cast<T>(i * 0.37 + 1.0);
			T* d = nullptr;
			cudaMalloc(&d, n * sizeof(T));
			cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
			std::string fa = dir + "/a.bin", fb = dir + "/b.bin";
			bool ok1 = write_array<T>(h.data(), nr, nc, fa);
			bool ok2 = write_device_array<T>(d, nr, nc, fb, ck);
			std::string name = std::string("matches_write_array_") + tname + "_" + std::to_string(nr) + "x" + std::to_string(nc) + "_chunk" + std::to_string(ck);
			bool same = ok1 && ok2 && slurp(fa) == slurp(fb) && !slurp(fa).empty();
			report(name, same, ok2 ? "bytes differ" : "write_device_array returned false");
			cudaFree(d);
		}

	T* d = nullptr;
	cudaMalloc(&d, 16 * sizeof(T));
	cudaMemset(d, 0, 16 * sizeof(T));
	report(std::string("rejects_wrong_extension_") + tname, !write_device_array<T>(d, 4, 4, dir + "/x.txt"), "returned true");
	bool bad = write_device_array<T>(d, 4, 4, "/nonexistent_dir/x.bin", 64);
	bool good = write_device_array<T>(d, 4, 4, dir + "/c.bin", 64) && slurp(dir + "/c.bin").size() == 8 + 16 * sizeof(T);
	report(std::string("unwritable_path_fails_cleanly_") + tname, !bad && good, bad ? "returned true" : "later call failed");
	cudaFree(d);

	// header-only arrays
	for (auto& sh : {std::pair<int, int>(0, 7), std::pair<int, int>(7, 0)})
	{
		std::string fa = dir + "/a.bin", fb = dir + "/b.bin";
		bool ok1 = write_array<T>(nullptr, sh.first, sh.second, fa);
		bool ok2 = write_device_array<T>(nullptr, sh.first, sh.second, fb);
		report(std::string("empty_array_header_only_") + tname + "_" + std::to_string(sh.first) + "x" + std::to_string(sh.second),
			ok1 && ok2 && slurp(fa) == slurp(fb) && slurp(fb).size() == 8, "mismatch or false");
	}

	// failure after buffers exist: /dev/full opens fine and fails on write
	{
		std::string full = dir + "/full.bin";
		std::remove(full.c_str());
		bool linked = symlink("/dev/full", full.c_str()) == 0;
		size_t n = 1 << 20;
		T* dd = nullptr;
		cudaMalloc(&dd, n * sizeof(T));
		cudaMemset(dd, 0, n * sizeof(T));
		bool multi = write_device_array<T>(dd, 1024, 1024, full, 4096);     // many chunks, fails mid-stream
		bool single = write_device_array<T>(dd, 4, 4, full, 64);            // fails at flush/close
		bool after = write_device_array<T>(dd, 1024, 1024, dir + "/d.bin", 4096)
			&& slurp(dir + "/d.bin").size() == 8 + n * sizeof(T);
		bool cuda_clean = cudaGetLastError() == cudaSuccess;
		std::remove(full.c_str());
		report(std::string("dev_full_fails_cleanly_") + tname, linked && !multi && !single && after && cuda_clean,
			!linked ? "symlink failed" : multi ? "mid-stream returned true" : single ? "small returned true" : !after ? "later call failed" : "stale CUDA error");
		cudaFree(dd);
	}

	// several arrays through one writer: each file is byte-identical to write_array's. starting with the largest
	// array, its two buffers serve all the later ones; starting with a small one, they are grown once
	{
		const std::vector<std::pair<int, int>> big_first = {{1000, 1001}, {3, 5}, {200, 300}, {1, 1}, {0, 7}, {1000, 1001}};
		const std::vector<std::pair<int, int>> small_first = {{3, 5}, {1000, 1001}, {1, 1}, {200, 300}};
		for (int seq = 0; seq < 2; seq++)
		{
			const auto& shapes = seq == 0 ? big_first : small_first;
			DeviceArrayWriter<T> w(4096); // 4096-byte chunks: the large arrays take many
			bool ok = true;
			std::string detail;
			for (size_t i = 0; i < shapes.size(); i++)
			{
				int nr = shapes[i].first, nc = shapes[i].second;
				size_t n = static_cast<size_t>(nr) * nc;
				std::vector<T> h(n);
				for (size_t j = 0; j < n; j++) h[j] = static_cast<T>(j * 0.37 + i + 1.0);
				T* d = nullptr;
				if (n > 0)
				{
					cudaMalloc(&d, n * sizeof(T));
					cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
				}
				std::string fa = dir + "/seq_a.bin", fb = dir + "/seq_b.bin";
				bool ok1 = write_array<T>(h.data(), nr, nc, fa);
				bool ok2 = w.write(d, nr, nc, fb);
				if (!(ok1 && ok2 && slurp(fa) == slurp(fb)))
				{
					ok = false;
					detail += "array " + std::to_string(i) + " (" + std::to_string(nr) + "x" + std::to_string(nc) + ")"
						+ (ok2 ? " bytes differ; " : " write returned false; ");
				}
				cudaFree(d);
			}
			const int want = seq == 0 ? 2 : 3; // two buffers, plus the single one of the 3x5 array that was outgrown
			if (w.allocations() != want)
			{
				ok = false;
				detail += std::to_string(w.allocations()) + " pinned allocations, want " + std::to_string(want) + "; ";
			}
			bool released = w.release();
			report(std::string("writer_sequence_matches_write_array_") + (seq == 0 ? "big_first_" : "small_first_") + tname,
				ok && released && cudaGetLastError() == cudaSuccess, released ? detail : detail + "release failed");
		}
	}

	// a failed file in the middle of a sequence leaves the writer usable, with the same buffers
	{
		std::string full = dir + "/full_seq.bin";
		std::remove(full.c_str());
		bool linked = symlink("/dev/full", full.c_str()) == 0;
		const int nr = 1024, nc = 1024;
		size_t n = static_cast<size_t>(nr) * nc;
		std::vector<T> h(n);
		for (size_t j = 0; j < n; j++) h[j] = static_cast<T>(j * 0.11 + 2.0);
		T* dd = nullptr;
		cudaMalloc(&dd, n * sizeof(T));
		cudaMemcpy(dd, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
		DeviceArrayWriter<T> w(4096);
		std::string fa = dir + "/seq_ref.bin", f1 = dir + "/seq_1.bin", f2 = dir + "/seq_2.bin", f3 = dir + "/seq_3.bin";
		bool ref = write_array<T>(h.data(), nr, nc, fa);
		bool first = w.write(dd, nr, nc, f1);
		bool multi = w.write(dd, nr, nc, full);   // many chunks, fails mid-stream
		bool second = w.write(dd, nr, nc, f2);
		bool single = w.write(dd, 4, 4, full);    // fails at flush/close
		bool third = w.write(dd, nr, nc, f3);
		bool same = ref && slurp(f1) == slurp(fa) && slurp(f2) == slurp(fa) && slurp(f3) == slurp(fa);
		bool cuda_clean = cudaGetLastError() == cudaSuccess;
		int allocs = w.allocations();
		std::remove(full.c_str());
		report(std::string("writer_survives_a_failed_file_") + tname,
			linked && first && !multi && second && !single && third && same && cuda_clean && allocs == 2,
			!linked ? "symlink failed" : multi ? "mid-stream returned true" : single ? "small returned true"
			: !(first && second && third) ? "a good write returned false" : !same ? "bytes differ"
			: !cuda_clean ? "stale CUDA error" : std::to_string(allocs) + " pinned allocations, want 2");
		cudaFree(dd);
	}

	// a pending error fails one write of a writer, and the next write works
	{
		T* dd = nullptr;
		cudaMalloc(&dd, 16 * sizeof(T));
		cudaMemset(dd, 0, 16 * sizeof(T));
		DeviceArrayWriter<T> w(64);
		std::string f = dir + "/pending_w.bin";
		bool before = w.write(dd, 4, 4, f);
		std::remove(f.c_str());
		cudaMemcpy(dd, dd, 16 * sizeof(T), (cudaMemcpyKind)12345); // invalid, non-sticky error
		bool r = w.write(dd, 4, 4, f);
		bool exists = std::filesystem::exists(f);
		bool again = w.write(dd, 4, 4, f) && slurp(f).size() == 8 + 16 * sizeof(T);
		report(std::string("writer_pending_error_fails_one_write_") + tname, before && !r && !exists && again,
			!before ? "first write failed" : r ? "returned true" : exists ? "file written" : "retry failed");
		cudaFree(dd);
	}

	// pending error is reported, nothing is written
	{
		T* dd = nullptr;
		cudaMalloc(&dd, 16 * sizeof(T));
		std::string f = dir + "/pending.bin";
		std::remove(f.c_str());
		cudaMemcpy(dd, dd, 16 * sizeof(T), (cudaMemcpyKind)12345); // invalid, non-sticky error
		bool r = write_device_array<T>(dd, 4, 4, f);
		bool exists = std::filesystem::exists(f);
		bool again = write_device_array<T>(dd, 4, 4, f); // error was consumed
		report(std::string("pending_error_is_reported_") + tname, !r && !exists && again, r ? "returned true" : exists ? "file written" : "retry failed");
		cudaFree(dd);
	}
}

int main()
{
	std::string dir = "/tmp/ipm_bench/cuda_tests";
	if (const char* e = std::getenv("DEVICE_IO_TEST_DIR")) dir = e;
	std::filesystem::create_directories(dir);
	run<float>("float", dir);
	run<double>("double", dir);
	return n_fail ? 1 : 0;
}
