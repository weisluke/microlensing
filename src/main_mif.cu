/******************************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

******************************************************************************/


#include "mif.cuh"
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

MIF<dtype> mif;

/******************************************************************************
constants to be used
******************************************************************************/
constexpr int OPTS_SIZE = 2 * 26;
const std::string OPTS[OPTS_SIZE] =
{
	"-h", "--help",
	"-v", "--verbose",
	"-k", "--kappa_tot",
	"-y", "--shear",
	"-s", "--smooth_fraction", //provided as a courtesy in this executable. not part of the ipm or irs classes
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
	"-y1", "--y1",
	"-y2", "--y2",
	"-v1", "--v1",
	"-v2", "--v2",
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
dtype smooth_fraction = static_cast<dtype>(1 - mif.kappa_star / mif.kappa_tot);



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
		<< "  -k,--kappa_tot           Specify the total convergence. Default value: " << mif.kappa_tot << "\n"
		<< "  -y,--shear               Specify the shear. Default value: " << mif.shear << "\n"
		<< "  -s,--smooth_fraction     Specify the fraction of convergence due to smoothly\n"
		<< "                           distributed mass. Default value: " << smooth_fraction << "\n"
		<< "  -ks,--kappa_star         Specify the convergence in point mass lenses. If\n"
		<< "                           provided, this overrides any supplied value for the\n"
		<< "                           smooth fraction. Default value: " << mif.kappa_star << "\n"
		<< "  -t,--theta_star          Specify the size of the Einstein radius of a unit\n"
		<< "                           mass point lens in arbitrary units. Default value: " << mif.theta_star << "\n"
		<< "  -mf,--mass_function      Specify the mass function to use for the point mass\n"
		<< "                           lenses. Options are: equal, uniform, Salpeter,\n"
		<< "                           Kroupa, and optical_depth. Default value: " << mif.mass_function_str << "\n"
		<< "  -ms,--m_solar            Specify the solar mass in arbitrary units.\n"
		<< "                           Default value: " << mif.m_solar << "\n"
		<< "  -ml,--m_lower            Specify the lower mass cutoff in solar mass units.\n"
		<< "                           Default value: " << mif.m_lower << "\n"
		<< "  -mh,--m_upper            Specify the upper mass cutoff in solar mass units.\n"
		<< "                           Default value: " << mif.m_upper << "\n"
		<< "  -ll,--light_loss         Allowed average fraction of light lost due to\n"
		<< "                           scatter by the microlenses in the large deflection\n"
		<< "                           limit. Default value: " << mif.light_loss << "\n"
		<< "  -r,--rectangular         Specify whether the star field should be\n"
		<< "                           rectangular (1) or circular (0). Default value: " << mif.rectangular << "\n"
		<< "  -a,--approx              Specify whether terms for alpha_smooth should be\n"
		<< "                           approximated (1) or exact (0). Default value: " << mif.approx << "\n"
		<< "  -ss,--safety_scale       Specify the ratio of the size of the star field to\n"
		<< "                           the radius of convergence for alpha_smooth.\n"
		<< "                           Default value: " << mif.safety_scale << "\n"
		<< "  -sf,--starfile           Specify the location of a binary file containing\n"
		<< "                           values for num_stars, rectangular, corner,\n"
		<< "                           theta_star, and the star positions and masses, in an\n"
		<< "                           order as defined in this source code.\n"
		<< "                           A whitespace delimited text file where each line\n"
		<< "                           contains the x1 and x2 coordinates and the mass of a\n"
		<< "                           microlens, in units where theta_star = 1, is also\n"
		<< "                           accepted. If provided, this takes precedence for all\n"
		<< "                           star information.\n"
		<< "  -y1, --y1                Specify the y1 and y2 coordinates of the source, or\n"
		<< "  -y2, --y2                a position that the source travels through.\n"
		<< "                           Default value: " << mif.w0 << "\n"
		<< "  -v1, --v1                Specify the y1 and y2 components of the source\n"
		<< "  -v2, --v2                velocity.\n"
		<< "                           Default value: " << mif.v << "\n"
		<< "  -rs,--random_seed        Specify the random seed for star field generation.\n"
		<< "                           A value of 0 is reserved for star input files.\n"
		<< "  -ws,--write_stars        Specify whether to write stars (1) or not (0).\n"
		<< "                           Default value: " << mif.write_stars << "\n"
		<< "  -wm,--write_maps         Specify whether to write magnification maps (1) or\n"
		<< "                           not (0). Default value: " << mif.write_maps << "\n"
		<< "  -wp,--write_parities     Specify whether to write parity specific\n"
		<< "                           magnification maps (1) or not (0). Default value: " << mif.write_parities << "\n"
		<< "  -wh,--write_histograms   Specify whether to write histograms (1) or not (0).\n"
		<< "                           Default value: " << mif.write_histograms << "\n"
		<< "  -o,--outfile_prefix      Specify the prefix to be used in output file names.\n"
		<< "                           Default value: " << mif.outfile_prefix << "\n";
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

		if (argv[i] == std::string("-k") || argv[i] == std::string("--kappa_tot"))
		{
			try
			{
				set_param("kappa_tot", mif.kappa_tot, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid kappa_tot input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-y") || argv[i] == std::string("--shear"))
		{
			try
			{
				set_param("shear", mif.shear, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid shear input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-s") || argv[i] == std::string("--smooth_fraction"))
		{
			if (cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile"))
			{
				continue;
			}
			if (cmd_option_exists(argv, argv + argc, "-ks") || cmd_option_exists(argv, argv + argc, "--kappa_star"))
			{
				continue;
			}
			try
			{
				set_param("smooth_fraction", smooth_fraction, std::stod(cmdinput), verbose);
				if (smooth_fraction < 0)
				{
					std::cerr << "Error. Invalid smooth_fraction input. smooth_fraction must be >= 0\n";
					return -1;
				}
				else if (smooth_fraction >= 1)
				{
					std::cerr << "Error. Invalid smooth_fraction input. smooth_fraction must be < 1\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid smooth_fraction input.\n";
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
				set_param("kappa_star", mif.kappa_star, std::stod(cmdinput), verbose);
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
				set_param("theta_star", mif.theta_star, std::stod(cmdinput), verbose);
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
			set_param("mass_function", mif.mass_function_str, make_lowercase(cmdinput), verbose);
		}
		else if (argv[i] == std::string("-ms") || argv[i] == std::string("--m_solar"))
		{
			if (cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile"))
			{
				continue;
			}
			try
			{
				set_param("m_solar", mif.m_solar, std::stod(cmdinput), verbose);
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
				set_param("m_lower", mif.m_lower, std::stod(cmdinput), verbose);
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
				set_param("m_upper", mif.m_upper, std::stod(cmdinput), verbose);
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
				set_param("light_loss", mif.light_loss, std::stod(cmdinput), verbose);
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
				set_param("rectangular", mif.rectangular, std::stoi(cmdinput), verbose);
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
				set_param("approx", mif.approx, std::stoi(cmdinput), verbose);
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
				set_param("safety_scale", mif.safety_scale, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid safety_scale input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-sf") || argv[i] == std::string("--starfile"))
		{
			set_param("starfile", mif.starfile, cmdinput, verbose);
		}
		else if (argv[i] == std::string("-y1") || argv[i] == std::string("--y1"))
		{
			try
			{
				set_param("y1", mif.w0.re, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid y1 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-y2") || argv[i] == std::string("--y2"))
		{
			try
			{
				set_param("y2", mif.w0.im, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid y2 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-v1") || argv[i] == std::string("--v1"))
		{
			try
			{
				set_param("v1", mif.v.re, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid v1 input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-v2") || argv[i] == std::string("--v2"))
		{
			try
			{
				set_param("v2", mif.v.im, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid v2 input.\n";
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
				set_param("random_seed", mif.random_seed, std::stoi(cmdinput), verbose);
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
				set_param("write_stars", mif.write_stars, std::stoi(cmdinput), verbose);
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
				set_param("write_maps", mif.write_maps, std::stoi(cmdinput), verbose);
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
				set_param("write_parities", mif.write_parities, std::stoi(cmdinput), verbose);
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
				set_param("write_histograms", mif.write_histograms, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_histograms input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			set_param("outfile_prefix", mif.outfile_prefix, cmdinput, verbose);
		}
	}

	if (!(cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile")) &&
		!(cmd_option_exists(argv, argv + argc, "-ks") || cmd_option_exists(argv, argv + argc, "--kappa_star")) &&
		(cmd_option_exists(argv, argv + argc, "-s") || cmd_option_exists(argv, argv + argc, "--smooth_fraction")))
	{
		set_param("kappa_star", mif.kappa_star, (1 - smooth_fraction) * mif.kappa_tot, verbose);
	}

	print_verbose("\n", verbose, 2);

	/******************************************************************************
	END read in options and values, checking correctness and exiting if necessary
	******************************************************************************/


	/******************************************************************************
	run and save files
	******************************************************************************/
	if (!mif.run(verbose)) return -1;
	if (!mif.save(verbose)) return -1;


	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}

