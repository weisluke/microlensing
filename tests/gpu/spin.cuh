// Busy-wait kernel for timing tests: keeps a stream occupied for a known time without touching memory.
#pragma once

#include <cuda_runtime.h>

__global__ void spin_kernel(long long cycles)
{
	long long start = clock64();
	while (clock64() - start < cycles) {}
}

// occupy stream s for about the given number of seconds (one block, one thread), using the current device's clock rate
void launch_spin(double seconds, cudaStream_t s)
{
	int device = 0;
	int khz = 0;
	cudaGetDevice(&device);
	cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, device);
	long long cycles = static_cast<long long>(seconds * khz * 1000.0);
	spin_kernel<<<1, 1, 0, s>>>(cycles);
}
