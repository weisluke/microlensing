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
CUDA error checking

\param name -- to print in error msg
\param sync -- boolean of whether device needs synchronized or not
\param file -- the file being run
\param line -- line number of the source code where the error is given

\return bool -- true for error, false for no error
******************************************************************************/
bool cuda_error(const char* name, bool sync, const char* file, const int line)
{
	cudaError_t err = cudaGetLastError();

	/******************************************************************************
	if the last error message is not a success, print the error code and msg and
	return true (i.e., an error occurred)
	******************************************************************************/
	if (err != cudaSuccess)
	{
		const char* errMsg = cudaGetErrorString(err);
		std::cerr << "CUDA error check for " << name << " failed at " << file << ":" << line << "\n";
		std::cerr << "Error code: " << err << " (" << errMsg << ")\n";
		return true;
	}

	/******************************************************************************
	if a device synchronization is also to be done
	******************************************************************************/
	if (sync)
	{
		/******************************************************************************
		perform the same error checking as initially
		******************************************************************************/
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			const char* errMsg = cudaGetErrorString(err);
			std::cerr << "CUDA error check for cudaDeviceSynchronize failed at " << file << ":" << line << "\n";
			std::cerr << "Error code: " << err << " (" << errMsg << ")\n";
			return true;
		}
	}
	return false;
}

/******************************************************************************
set the number of threads per block for the kernel

\param threads -- reference to threads
\param x -- number of threads per block in x dimension
\param y -- number of threads per block in y dimension
\param z -- number of threads per block in z dimension
******************************************************************************/
void set_threads(dim3& threads, int x = 1, int y = 1, int z = 1)
{
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	threads.x = x;
	threads.y = y;
	threads.z = z;

	if (threads.x * threads.y * threads.z > prop.maxThreadsPerBlock)
	{
		int t_x;
		int t_y;
		int t_z;
		do
		{
			t_x = threads.x;
			t_y = threads.y;
			t_z = threads.z;

			double factor = std::cbrt((1.0 * prop.maxThreadsPerBlock) / (t_x * t_y * t_z));

			threads.x = std::ceil(factor * t_x);
			threads.y = std::ceil(factor * t_y);
			threads.z = std::ceil(factor * t_z);
		} while (threads.x != t_x || threads.y != t_y || threads.z != t_z);
	}
}

/******************************************************************************
set the number of blocks for the kernel

\param threads -- reference to threads
\param blocks -- reference to blocks
\param x -- number of total threads in x dimension
\param y -- number of total threads in y dimension
\param z -- number of total threads in z dimension
******************************************************************************/
void set_blocks(dim3& threads, dim3& blocks, int x = 1, int y = 1, int z = 1)
{
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	blocks.x = (x - 1) / threads.x + 1;
	blocks.y = (y - 1) / threads.y + 1;
	blocks.z = (z - 1) / threads.z + 1;

	if (blocks.x * blocks.y * blocks.z > prop.multiProcessorCount)
	{
		int b_x;
		int b_y;
		int b_z;
		do
		{
			b_x = blocks.x;
			b_y = blocks.y;
			b_z = blocks.z;

			double factor = std::cbrt((1.0 * prop.multiProcessorCount) / (b_x * b_y * b_z));

			blocks.x = std::ceil(factor * b_x);
			blocks.y = std::ceil(factor * b_y);
			blocks.z = std::ceil(factor * b_z);
		} while (blocks.x != b_x || blocks.y != b_y || blocks.z != b_z);
	}
}

/******************************************************************************
display info about a cuda device

\param num -- device number
\param prop -- reference to cuda device property structure
******************************************************************************/
void show_device_info(int num, cudaDeviceProp& prop)
{
	std::cout << "Device Number: " << num << "\n";
	std::cout << "  Device name: " << prop.name << "\n";
	std::cout << "  Compute capability (major.minor): " << prop.major << "." << prop.minor << "\n";
	std::cout << "  Clock rate (kHz): " << prop.clockRate << "\n";
	std::cout << "  Memory clock rate (kHz): " << prop.memoryClockRate << "\n";
	std::cout << "  Memory bus width (bits): " << prop.memoryBusWidth << "\n";
	std::cout << "  Peak memory bandwidth (GB/s): " << 2 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / (1024 * 1024) << "\n";
	std::cout << "  Single to double precision performance ratio: " << prop.singleToDoublePrecisionPerfRatio << "\n";
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

