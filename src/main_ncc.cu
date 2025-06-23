/******************************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

******************************************************************************/


#include "ncc.cuh"
#include "util/util.cuh"

#include <iostream>
#include <string>


#if defined(is_float) && !defined(is_double)
using dtype = float; //type to be used throughout this program. float or double
#elif !defined(is_float) && defined(is_double)
using dtype = double; //type to be used throughout this program. float or double
#else
#error "Error. One, and only one, of is_float or is_double must be defined"
#endif

NCC<dtype> ncc;

/******************************************************************************
constants to be used
******************************************************************************/
constexpr int OPTS_SIZE = 2 * 13;
const std::string OPTS[OPTS_SIZE] =
{
	"-h", "--help",
	"-v", "--verbose",
	"-ip", "--infile_prefix",
	"-cy1", "--center_y1",
	"-cy2", "--center_y2",
	"-hly1", "--half_length_y1",
	"-hly2", "--half_length_y2",
	"-npy1", "--num_pixels_y1",
	"-npy2", "--num_pixels_y2",
	"-os", "--over_sample",
	"-wm", "--write_maps",
	"-wh", "--write_histograms",
	"-o", "--outfile_prefix"
};

/******************************************************************************
default input option values
******************************************************************************/
int verbose = 1;



/******************************************************************************
Print the program usage help message

\param name -- name of the executable
******************************************************************************/
void display_usage(char* name)
{
	if (name)
	{
		std::cout << "Usage: " << name << " opt1 val1 opt2 val2 opt3 val3 ...\n";
	}
	else
	{
		std::cout << "Usage: programname opt1 val1 opt2 val2 opt3 val3 ...\n";
	}
	std::cout
		<< "                                                                               \n"
		<< "Options:\n"
		<< "  -h,--help                Show this help message.\n"
		<< "  -v,--verbose             Specify verbosity of output from 0 (none) to 1, 2,\n"
		<< "                           or 3 (low, medium, high). Default value: " << verbose << "\n"
		<< "  -ip,--infile_prefix      Specify the prefix to be used when reading in files.\n"
		<< "                           Default value: " << ncc.infile_prefix << "\n"
		<< "  -cy1, --center_y1        Specify the y1 and y2 coordinates of the center of\n"
		<< "  -cy2, --center_y2        the number of caustic crossings map.\n"
		<< "                           Default value: " << ncc.center_y << "\n"
		<< "  -hly1,--half_length_y1   Specify the y1 and y2 extent of the half-length of\n"
		<< "  -hly2,--half_length_y2   the number of caustic crossings map.\n"
		<< "                           Default value: " << ncc.half_length_y << "\n"
		<< "  -npy1,--num_pixels_y1    Specify the number of pixels per side for the\n"
		<< "  -npy2,--num_pixels_y2    number of caustic crossings map.\n"
		<< "                           Default value: " << ncc.num_pixels_y << "\n"
		<< "  -os,--over_sample        Specify the power of 2 by which to oversample the\n"
		<< "                           final pixels. E.g., an input of 4 means the final\n"
		<< "                           pixel array will initially be oversampled by a value\n"
		<< "                           of 2^4 = 16 along both axes. This will require\n"
		<< "                           16*16 = 256 times more memory. Default value: " << ncc.over_sample << "\n"
		<< "  -wm,--write_maps         Specify whether to write number of caustic crossings\n"
		<< "                           maps (1) or not (0). Default value: " << ncc.write_maps << "\n"
		<< "  -wh,--write_histograms   Specify whether to write histograms (1) or not (0).\n"
		<< "                           Default value: " << ncc.write_histograms << "\n"
		<< "  -o,--outfile_prefix      Specify the prefix to be used in output file names.\n"
		<< "                           Default value: " << ncc.outfile_prefix << "\n";
}



int main(int argc, char* argv[])
{
	/******************************************************************************
	if help option has been input, display usage message
	******************************************************************************/
	if (cmd_option_exists(argv, argv + argc, "-h") || cmd_option_exists(argv, argv + argc, "--help"))
	{
		display_usage(argv[0]);
		return -1;
	}

	/******************************************************************************
	if there are input options, but not an even number (since all options take a
	parameter), display usage message and exit
	subtract 1 to take into account that first argument array value is program name
	******************************************************************************/
	if ((argc - 1) % 2 != 0)
	{
		std::cerr << "Error. Invalid input syntax.\n";
		display_usage(argv[0]);
		return -1;
	}

	/******************************************************************************
	check that all options given are valid. use step of 2 since all input options
	take parameters (assumed to be given immediately after the option). start at 1,
	since first array element, argv[0], is program name
	******************************************************************************/
	for (int i = 1; i < argc; i += 2)
	{
		if (!cmd_option_valid(OPTS, OPTS + OPTS_SIZE, argv[i]))
		{
			std::cerr << "Error. Invalid input syntax. Unknown option " << argv[i] << "\n";
			display_usage(argv[0]);
			return -1;
		}
		if (argv[i] == std::string("-v") || argv[i] == std::string("--verbose"))
		{
			char* cmdinput = cmd_option_value(argv, argv + argc, std::string(argv[i]));
			try
			{
				set_param("verbose", verbose, std::stoi(cmdinput), std::stoi(cmdinput));
			}
			catch (...)
			{
				std::cerr << "Error. Invalid verbose input.\n";
				return -1;
			}
		}
	}


	/******************************************************************************
	BEGIN read in options and values, checking correctness and exiting if necessary
	******************************************************************************/

	for (int i = 1; i < argc; i += 2)
	{
		char* cmdinput = cmd_option_value(argv, argv + argc, std::string(argv[i]));

		if (argv[i] == std::string("-ip") || argv[i] == std::string("--infile_prefix"))
		{
			set_param("infile_prefix", ncc.infile_prefix, cmdinput, verbose);
		}
		else if (argv[i] == std::string("-cy1") || argv[i] == std::string("--center_y1"))
		{
			try
			{
				set_param("center_y1", ncc.center_y.re, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid center_y1 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-cy2") || argv[i] == std::string("--center_y2"))
		{
			try
			{
				set_param("center_y2", ncc.center_y.im, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid center_y2 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-hly1") || argv[i] == std::string("--half_length_y1"))
		{
			try
			{
				set_param("half_length_y1", ncc.half_length_y.re, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid half_length_y1 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-hly2") || argv[i] == std::string("--half_length_y2"))
		{
			try
			{
				set_param("half_length_y2", ncc.half_length_y.im, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid half_length_y2 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-npy1") || argv[i] == std::string("--num_pixels_y1"))
		{
			try
			{
				set_param("num_pixels_y1", ncc.num_pixels_y.re, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_pixels_y1 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-npy2") || argv[i] == std::string("--num_pixels_y2"))
		{
			try
			{
				set_param("num_pixels_y2", ncc.num_pixels_y.im, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_pixels_y2 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-os") || argv[i] == std::string("--over_sample"))
		{
			try
			{
				set_param("over_sample", ncc.over_sample, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid over_sample input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wm") || argv[i] == std::string("--write_maps"))
		{
			try
			{
				set_param("write_maps", ncc.write_maps, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_maps input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wh") || argv[i] == std::string("--write_histograms"))
		{
			try
			{
				set_param("write_histograms", ncc.write_histograms, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_histograms input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			set_param("outfile_prefix", ncc.outfile_prefix, cmdinput, verbose);
		}
	}

	print_verbose("\n", verbose, 2);

	/******************************************************************************
	END read in options and values, checking correctness and exiting if necessary
	******************************************************************************/


	/******************************************************************************
	run and save files
	******************************************************************************/
	if (!ncc.run(verbose)) return -1;
	if (!ncc.save(verbose)) return -1;


	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}

