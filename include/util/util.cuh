#pragma once

#include <algorithm> //for std::find, std::transform
#include <cctype> //for std::tolower
#include <cmath> //for std::cbrt, std::ceil
#include <iostream>
#include <string>


/******************************************************************************
determine if a string exists within some array of char*s

\param begin -- starting pointer of array
\param end -- ending pointer of array
\param option -- string to find within the array

\return bool -- true if string is in array, false if not
******************************************************************************/
bool cmd_option_exists(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

/******************************************************************************
determine if a string is a valid option from some options array

\param begin -- starting pointer of array of string options
\param end -- ending pointer of array of string options
\param option -- string to find within the array

\return bool -- true if string was found in array, false if not
******************************************************************************/
bool cmd_option_valid(const std::string* begin, const std::string* end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

/******************************************************************************
determine the parameter value of an option in array of char*s
assumed to be placed immediately after the option in some range

\param begin -- starting pointer of array
\param end -- ending pointer of array
\param option -- string to find the value of

\return char* -- array of chars of the value after the option
******************************************************************************/
char* cmd_option_value(char** begin, char** end, const std::string& option)
{
	char** itr = std::find(begin, end, option);

	/******************************************************************************
	if found option doesn't equal end and there is something in the following
	position (uses ++itr to increment before checking)
	******************************************************************************/
	if (itr != end && ++itr != end)
	{
		return *itr;
	}
	return nullptr;
}

/******************************************************************************
function to make a string lowercase

\param what -- reference to the string to make lowercase

\return lowercase version of the string
******************************************************************************/
std::string make_lowercase(const std::string& what)
{
	std::string lowercase = what;
	std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(), [](unsigned char c) { return std::tolower(c); });
	return lowercase;
}

/******************************************************************************
function to print a message

\param what -- what to print
\param verbose -- verbose level
\param level -- level of the message. only shown if verbose >= level

levels correspond to:
0 -- always present
1 -- low level
2 -- mid level
3 -- high level
******************************************************************************/
#define print_verbose(what, verbose, level) do {if (verbose >= level) {std::cout << what;}} while (false)

/******************************************************************************
function to set a parameter value and print message if necessary

\param name -- name of the parameter
\param param -- reference to the parameter
\param what -- value to set the parameter equal to
\param verbose -- verbose level
******************************************************************************/
template <typename T, typename U>
void set_param(const std::string& name, T& param, U what, int verbose)
{
	param = static_cast<T>(what);
	print_verbose(name << " set to: " << param << "\n", verbose, 2);
}

/******************************************************************************
function to print out progress bar of loops
examples: [====    ] 50 %       [=====  ] 73 %

\param verbose -- verbose level
\param icurr -- current position in the loop
\param imax -- maximum position in the loop
\param num_bars -- number of = symbols inside the bar
				   default value: 50
******************************************************************************/
__host__ __device__ void print_progress(int verbose, unsigned long long int icurr, unsigned long long int imax, int num_bars = 50)
{
	if (verbose < 1) {return;}

#ifdef __CUDA_ARCH__
	printf("\r%llu %%", icurr * 100 / imax);
#else
	std::cout << "\r[";
	for (int i = 0; i < num_bars; i++)
	{
		if (i <= icurr * num_bars / imax)
		{
			std::cout << "=";
		}
		else
		{
			std::cout << " ";
		}
	}
	std::cout << "] " << icurr * 100 / imax << " %" << std::flush;
#endif
}

/******************************************************************************
code of the last error reported by cuda_error on this thread, 0 if none
******************************************************************************/
inline thread_local int last_cuda_error_code = 0;

/******************************************************************************
return the code of the last error cuda_error reported on this thread (a
cudaError_t as an int, 0 if none since the last call) and reset it to 0

\return int -- the stored error code
******************************************************************************/
int take_last_cuda_error()
{
	int code = last_cuda_error_code;
	last_cuda_error_code = 0;
	return code;
}

/******************************************************************************
CUDA error checking

\param name -- to print in error msg
\param sync -- boolean of whether to synchronize or not
\param file -- the file being run
\param line -- line number of the source code where the error is given
\param stream -- stream to synchronize if sync is true. the default, 0,
				synchronizes the whole device

every error reported is also stored for take_last_cuda_error

\return bool -- true for error, false for no error
******************************************************************************/
bool cuda_error(const char* name, bool sync, const char* file, const int line, cudaStream_t stream = 0)
{
	cudaError_t err = cudaGetLastError();

	/******************************************************************************
	if the last error message is not a success, print the error code and msg and
	return true (i.e., an error occurred)
	******************************************************************************/
	if (err != cudaSuccess)
	{
		last_cuda_error_code = static_cast<int>(err);
		const char* errMsg = cudaGetErrorString(err);
		std::cerr << "CUDA error check for " << name << " failed at " << file << ":" << line << "\n";
		std::cerr << "Error code: " << err << " (" << errMsg << ")\n";
		return true;
	}

	/******************************************************************************
	if a synchronization is also to be done
	******************************************************************************/
	if (sync)
	{
		/******************************************************************************
		perform the same error checking as initially
		******************************************************************************/
		err = (stream == 0) ? cudaDeviceSynchronize() : cudaStreamSynchronize(stream);
		if (err != cudaSuccess)
		{
			last_cuda_error_code = static_cast<int>(err);
			const char* errMsg = cudaGetErrorString(err);
			std::cerr << "CUDA error check for " << ((stream == 0) ? "cudaDeviceSynchronize" : "cudaStreamSynchronize") << " failed at " << file << ":" << line << "\n";
			std::cerr << "Error code: " << err << " (" << errMsg << ")\n";
			return true;
		}
	}
	return false;
}

/******************************************************************************
properties of the current device, cached per thread because
cudaGetDeviceProperties costs ~0.5-1 ms and the tree/FMM loops called it
hundreds of times. refreshed when the thread's current device changes

\return const cudaDeviceProp& -- reference to this thread's cached properties
******************************************************************************/
const cudaDeviceProp& cached_device_prop()
{
	static thread_local cudaDeviceProp prop;
	static thread_local int cached_device = -1;
	int device;
	cudaGetDevice(&device);
	if (device != cached_device)
	{
		cudaGetDeviceProperties(&prop, device);
		cached_device = device;
	}
	return prop;
}

/******************************************************************************
get the maximum number of threads allowed per block for a kernel

\param kernel -- kernel we want the maximum number of threads for
\param shared_mem -- shared memory used during the kernel launch

\return num_threads
******************************************************************************/
template <typename T>
int get_max_num_threads(T kernel, int shared_mem = 0)
{
	int ideal_grid_size;
	int ideal_block_size;

	cudaOccupancyMaxPotentialBlockSize(&ideal_grid_size, &ideal_block_size, kernel, shared_mem);
	return ideal_block_size;
}

/******************************************************************************
get the number of threads per block for the kernel

\param kernel -- kernel whose maximum number of allowed threads limits the
                 desired number
\param x -- desired number of threads per block in x dimension
\param y -- desired number of threads per block in y dimension
\param z -- desired number of threads per block in z dimension
\param shared_mem -- shared memory used during the kernel launch

\return threads
******************************************************************************/
template <typename T>
dim3 get_threads(T kernel, int x = 1, int y = 1, int z = 1, int shared_mem = 0)
{
	int max_num_threads = get_max_num_threads(kernel, shared_mem);

	dim3 threads(x, y, z);

	if (threads.x * threads.y * threads.z > max_num_threads)
	{
		double factor = std::cbrt((1.0 * max_num_threads) / (threads.x * threads.y * threads.z));

		threads.x = std::ceil(factor * threads.x);
		threads.y = std::ceil(factor * threads.y);
		threads.z = std::ceil(factor * threads.z);

		while (threads.x * threads.y * threads.z > max_num_threads)
		{
			// A distinct maximum
			if (threads.x > threads.y && threads.x > threads.z)
			{
				threads.x -= 1;
			}
			else if (threads.y > threads.x && threads.y > threads.z)
			{
				threads.y -= 1;
			}
			else if (threads.z > threads.x && threads.z > threads.y)
			{
				threads.z -= 1;
			}
			// A tied maximum
			else if (threads.z == threads.y || threads.z == threads.x)
			{
				threads.z -= 1;
			}
			else if (threads.x == threads.y)
			{
				threads.y -= 1;
			}
		}
	}

	return threads;
}

/******************************************************************************
get the number of blocks for the provided threads per block and total number of
threads

\param threads -- threads per block
\param x -- number of total threads in x dimension
\param y -- number of total threads in y dimension
\param z -- number of total threads in z dimension
******************************************************************************/
dim3 get_blocks(dim3 threads, int x = 1, int y = 1, int z = 1)
{
	const cudaDeviceProp& prop = cached_device_prop();

	dim3 blocks(
		(x - 1) / threads.x + 1,
		(y - 1) / threads.y + 1,
		(z - 1) / threads.z + 1
	);

	if (blocks.x * blocks.y * blocks.z > prop.multiProcessorCount)
	{
		double factor = std::cbrt((1.0 * prop.multiProcessorCount) / (blocks.x * blocks.y * blocks.z));

		blocks.x = std::ceil(factor * blocks.x);
		blocks.y = std::ceil(factor * blocks.y);
		blocks.z = std::ceil(factor * blocks.z);

		while (blocks.x * blocks.y * blocks.z > prop.multiProcessorCount)
		{
			// A distinct maximum
			if (blocks.x > blocks.y && blocks.x > blocks.z)
			{
				blocks.x -= 1;
			}
			else if (blocks.y > blocks.x && blocks.y > blocks.z)
			{
				blocks.y -= 1;
			}
			else if (blocks.z > blocks.x && blocks.z > blocks.y)
			{
				blocks.z -= 1;
			}
			// A tied maximum
			else if (blocks.z == blocks.y || blocks.z == blocks.x)
			{
				blocks.z -= 1;
			}
			else if (blocks.x == blocks.y)
			{
				blocks.y -= 1;
			}
		}
	}

	return blocks;
}

/******************************************************************************
display info about a cuda device

\param num -- device number
\param prop -- reference to cuda device property structure
******************************************************************************/
void show_device_info(int num, cudaDeviceProp& prop)
{
	int clock_rate = 0;
	cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, num);
	int memory_clock_rate = 0;
	cudaDeviceGetAttribute(&memory_clock_rate, cudaDevAttrMemoryClockRate, num);
	int perf_ratio = 0;
	cudaDeviceGetAttribute(&perf_ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, num);

	std::cout << "Device Number: " << num << "\n";
	std::cout << "  Device name: " << prop.name << "\n";
	std::cout << "  Compute capability (major.minor): " << prop.major << "." << prop.minor << "\n";
	std::cout << "  Clock rate (kHz): " << clock_rate << "\n";
	std::cout << "  Memory clock rate (kHz): " << memory_clock_rate << "\n";
	std::cout << "  Memory bus width (bits): " << prop.memoryBusWidth << "\n";
	std::cout << "  Peak memory bandwidth (GB/s): " << 2 * memory_clock_rate * (prop.memoryBusWidth / 8) / (1024 * 1024) << "\n";
	std::cout << "  Single to double precision performance ratio: " << perf_ratio << "\n";
	std::cout << "  Total global memory (GB): " << prop.totalGlobalMem / (1024 * 1024 * 1024) << "\n";
	std::cout << "  Shared memory per multiprocessor (kbytes): " << prop.sharedMemPerMultiprocessor / 1024 << "\n";
	std::cout << "  Shared memory per block (kbytes): " << prop.sharedMemPerBlock / 1024 << "\n";
	std::cout << "  Number of multiprocessors: " << prop.multiProcessorCount << "\n";
	std::cout << "  Max blocks per multiprocessor: " << prop.maxBlocksPerMultiProcessor << " blocks\n";
	std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << " threads\n";
	std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << " threads\n";
	std::cout << "  Warp size: " << prop.warpSize << " threads\n";

	std::cout << "  Maximum (x, y, z) dimensions of block: ("
		<< prop.maxThreadsDim[0] << ", "
		<< prop.maxThreadsDim[1] << ", "
		<< prop.maxThreadsDim[2] << ")\n";

	std::cout << "  Maximum (x, y, z) dimensions of grid: ("
		<< prop.maxGridSize[0] << ", "
		<< prop.maxGridSize[1] << ", "
		<< prop.maxGridSize[2] << ")\n";
}

