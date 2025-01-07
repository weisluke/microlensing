/******************************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

******************************************************************************/


#include "macro_ipm.cuh"
#include "util.cuh"

#include <iostream>
#include <string>


#if defined(is_float) && !defined(is_double)
using dtype = float; //type to be used throughout this program. float or double
#elif !defined(is_float) && defined(is_double)
using dtype = double; //type to be used throughout this program. float or double
#else
#error "Error. One, and only one, of is_float or is_double must be defined"
#endif

MacroIPM<dtype> map_maker;

/******************************************************************************
constants to be used
******************************************************************************/
constexpr int OPTS_SIZE = 2 * 31;
const std::string OPTS[OPTS_SIZE] =
{
	"-h", "--help",
	"-v", "--verbose",
	"-p11", "--p11",
	"-p111", "--p111",
	"-p112", "--p112",
	"-p111", "--p122",
	"-p111", "--p122",
	"-ks", "--kappa_star",
	"-t", "--theta_star",
	"-mf", "--mass_function",
	"-ms", "--m_solar",
	"-ml", "--m_lower",
	"-mh", "--m_upper",
	"-ll", "--light_loss",
	"-r", "--rectangular",
	"-a", "--approx",
	"-ss", "--safety_scale",
	"-sf", "--starfile",
	"-cy1", "--center_y1",
	"-cy2", "--center_y2",
	"-hly1", "--half_length_y1",
	"-hly2", "--half_length_y2",
	"-npy1", "--num_pixels_y1",
	"-npy2", "--num_pixels_y2",
	"-nry", "--num_rays_y",
	"-rs", "--random_seed",
	"-ws", "--write_stars",
	"-wm", "--write_maps",
	"-wp", "--write_parities",
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
		<< "  -p11                     \n"
		<< "  -p111                    \n"
		<< "  -p112                    \n"
		<< "  -p122                    \n"
		<< "  -p222                    \n"
		<< "  -ks,--kappa_star         Specify the convergence in point mass lenses. If\n"
		<< "                           provided, this overrides any supplied value for the\n"
		<< "                           smooth fraction. Default value: " << map_maker.kappa_star << "\n"
		<< "  -t,--theta_star          Specify the size of the Einstein radius of a unit\n"
		<< "                           mass point lens in arbitrary units. Default value: " << map_maker.theta_star << "\n"
		<< "  -mf,--mass_function      Specify the mass function to use for the point mass\n"
		<< "                           lenses. Options are: equal, uniform, Salpeter,\n"
		<< "                           Kroupa, and optical_depth. Default value: " << map_maker.mass_function_str << "\n"
		<< "  -ms,--m_solar            Specify the solar mass in arbitrary units.\n"
		<< "                           Default value: " << map_maker.m_solar << "\n"
		<< "  -ml,--m_lower            Specify the lower mass cutoff in solar mass units.\n"
		<< "                           Default value: " << map_maker.m_lower << "\n"
		<< "  -mh,--m_upper            Specify the upper mass cutoff in solar mass units.\n"
		<< "                           Default value: " << map_maker.m_upper << "\n"
		<< "  -ll,--light_loss         Allowed average fraction of light lost due to\n"
		<< "                           scatter by the microlenses in the large deflection\n"
		<< "                           limit. Default value: " << map_maker.light_loss << "\n"
		<< "  -r,--rectangular         Specify whether the star field should be\n"
		<< "                           rectangular (1) or circular (0). Default value: " << map_maker.rectangular << "\n"
		<< "  -a,--approx              Specify whether terms for alpha_smooth should be\n"
		<< "                           approximated (1) or exact (0). Default value: " << map_maker.approx << "\n"
		<< "  -ss,--safety_scale       Specify the ratio of the size of the star field to\n"
		<< "                           the size of the shooting region.\n"
		<< "                           Default value: " << map_maker.safety_scale << "\n"
		<< "  -sf,--starfile           Specify the location of a binary file containing\n"
		<< "                           values for num_stars, rectangular, corner,\n"
		<< "                           theta_star, and the star positions and masses, in an\n"
		<< "                           order as defined in this source code.\n"
		<< "                           A whitespace delimited text file where each line\n"
		<< "                           contains the x1 and x2 coordinates and the mass of a\n"
		<< "                           microlens, in units where theta_star = 1, is also\n"
		<< "                           accepted. If provided, this takes precedence for all\n"
		<< "                           star information.\n"
		<< "  -cy1, --center_y1        Specify the y1 and y2 coordinates of the center of\n"
		<< "  -cy2, --center_y2        the magnification map.\n"
		<< "                           Default value: " << map_maker.center_y << "\n"
		<< "  -hly1,--half_length_y1   Specify the y1 and y2 extent of the half-length of\n"
		<< "  -hly2,--half_length_y2   the magnification map.\n"
		<< "                           Default value: " << map_maker.half_length_y << "\n"
		<< "  -npy1,--num_pixels_y1    Specify the number of pixels per side for the\n"
		<< "  -npy2,--num_pixels_y2    magnification map.\n"
		<< "                           Default value: " << map_maker.num_pixels_y << "\n"
		<< "  -nry,--num_rays_y        Specify the average number of rays per pixel in the\n"
		<< "                           source plane in the absence of lensing.\n"
		<< "                           Default value: " << map_maker.num_rays_y << "\n"
		<< "  -rs,--random_seed        Specify the random seed for star field generation.\n"
		<< "                           A value of 0 is reserved for star input files.\n"
		<< "  -ws,--write_stars        Specify whether to write stars (1) or not (0).\n"
		<< "                           Default value: " << map_maker.write_stars << "\n"
		<< "  -wm,--write_maps         Specify whether to write magnification maps (1) or\n"
		<< "                           not (0). Default value: " << map_maker.write_maps << "\n"
		<< "  -wp,--write_parities     Specify whether to write parity specific\n"
		<< "                           magnification maps (1) or not (0). Default value: " << map_maker.write_parities << "\n"
		<< "  -wh,--write_histograms   Specify whether to write histograms (1) or not (0).\n"
		<< "                           Default value: " << map_maker.write_histograms << "\n"
		<< "  -o,--outfile_prefix      Specify the prefix to be used in output file names.\n"
		<< "                           Default value: " << map_maker.outfile_prefix << "\n";
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

		if (argv[i] == std::string("-p11") || argv[i] == std::string("-p11"))
		{
			try
			{
				set_param("p11", map_maker.p11, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid p11 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-p111") || argv[i] == std::string("-p111"))
		{
			try
			{
				set_param("p111", map_maker.p111, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid p111 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-p112") || argv[i] == std::string("-p112"))
		{
			try
			{
				set_param("p112", map_maker.p112, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid p112 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-p122") || argv[i] == std::string("-p122"))
		{
			try
			{
				set_param("p122", map_maker.p122, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid p122 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-p222") || argv[i] == std::string("-p222"))
		{
			try
			{
				set_param("p222", map_maker.p222, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid p222 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ks") || argv[i] == std::string("--kappa_star"))
		{
			if (cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile"))
			{
				continue;
			}
			try
			{
				set_param("kappa_star", map_maker.kappa_star, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid kappa_star input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-t") || argv[i] == std::string("--theta_star"))
		{
			if (cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile"))
			{
				continue;
			}
			try
			{
				set_param("theta_star", map_maker.theta_star, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid theta_star input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-mf") || argv[i] == std::string("--mass_function"))
		{
			if (cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile"))
			{
				continue;
			}
			set_param("mass_function", map_maker.mass_function_str, make_lowercase(cmdinput), verbose);
		}
		else if (argv[i] == std::string("-ms") || argv[i] == std::string("--m_solar"))
		{
			if (cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile"))
			{
				continue;
			}
			try
			{
				set_param("m_solar", map_maker.m_solar, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid m_solar input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ml") || argv[i] == std::string("--m_lower"))
		{
			if (cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile"))
			{
				continue;
			}
			try
			{
				set_param("m_lower", map_maker.m_lower, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid m_lower input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-mh") || argv[i] == std::string("--m_upper"))
		{
			if (cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile"))
			{
				continue;
			}
			try
			{
				set_param("m_upper", map_maker.m_upper, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid m_upper input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ll") || argv[i] == std::string("--light_loss"))
		{
			try
			{
				set_param("light_loss", map_maker.light_loss, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid light_loss input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-r") || argv[i] == std::string("--rectangular"))
		{
			if (cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile"))
			{
				continue;
			}
			try
			{
				set_param("rectangular", map_maker.rectangular, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid rectangular input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-a") || argv[i] == std::string("--approx"))
		{
			try
			{
				set_param("approx", map_maker.approx, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid approx input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ss") || argv[i] == std::string("--safety_scale"))
		{
			try
			{
				set_param("safety_scale", map_maker.safety_scale, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid safety_scale input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-sf") || argv[i] == std::string("--starfile"))
		{
			set_param("starfile", map_maker.starfile, cmdinput, verbose);
		}
		else if (argv[i] == std::string("-cy1") || argv[i] == std::string("--center_y1"))
		{
			try
			{
				set_param("center_y1", map_maker.center_y.re, std::stod(cmdinput), verbose);
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
				set_param("center_y2", map_maker.center_y.im, std::stod(cmdinput), verbose);
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
				set_param("half_length_y1", map_maker.half_length_y.re, std::stod(cmdinput), verbose);
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
				set_param("half_length_y2", map_maker.half_length_y.im, std::stod(cmdinput), verbose);
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
				set_param("num_pixels_y1", map_maker.num_pixels_y.re, std::stoi(cmdinput), verbose);
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
				set_param("num_pixels_y2", map_maker.num_pixels_y.im, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_pixels_y2 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-nry") || argv[i] == std::string("--num_rays_y"))
		{
			try
			{
				set_param("num_rays_y", map_maker.num_rays_y, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_rays_y input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-rs") || argv[i] == std::string("--random_seed"))
		{
			if (cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile"))
			{
				continue;
			}
			try
			{
				set_param("random_seed", map_maker.random_seed, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid random_seed input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ws") || argv[i] == std::string("--write_stars"))
		{
			try
			{
				set_param("write_stars", map_maker.write_stars, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_stars input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wm") || argv[i] == std::string("--write_maps"))
		{
			try
			{
				set_param("write_maps", map_maker.write_maps, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_maps input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wp") || argv[i] == std::string("--write_parities"))
		{
			try
			{
				set_param("write_parities", map_maker.write_parities, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_parities input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wh") || argv[i] == std::string("--write_histograms"))
		{
			try
			{
				set_param("write_histograms", map_maker.write_histograms, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_histograms input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			set_param("outfile_prefix", map_maker.outfile_prefix, cmdinput, verbose);
		}
	}

	if (!(cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile")) &&
		!(cmd_option_exists(argv, argv + argc, "-ks") || cmd_option_exists(argv, argv + argc, "--kappa_star")) &&
		(cmd_option_exists(argv, argv + argc, "-s") || cmd_option_exists(argv, argv + argc, "--smooth_fraction")))
	{
		set_param("kappa_star", map_maker.kappa_star, (1 - smooth_fraction) * map_maker.kappa_tot, verbose);
	}

	print_verbose("\n", verbose, 2);

	/******************************************************************************
	END read in options and values, checking correctness and exiting if necessary
	******************************************************************************/


	/******************************************************************************
	run and save files
	******************************************************************************/
	if (!map_maker.run(verbose)) return -1;
	if (!map_maker.save(verbose)) return -1;


	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}

