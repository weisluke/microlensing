/******************************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

******************************************************************************/


#include "ccf.cuh"
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

CCF<dtype> ccf;

/******************************************************************************
constants to be used
******************************************************************************/
constexpr int OPTS_SIZE = 2 * 24;
const std::string OPTS[OPTS_SIZE] =
{
	"-h", "--help",
	"-v", "--verbose",
	"-k", "--kappa_tot",
	"-y", "--shear",
	"-s", "--smooth_fraction", //provided as a courtesy in this executable. not part of the ccf class
	"-ks", "--kappa_star",
	"-t", "--theta_star",
	"-mf", "--mass_function",
	"-ms", "--m_solar",
	"-ml", "--m_lower",
	"-mh", "--m_upper",
	"-r", "--rectangular",
	"-a", "--approx",
	"-ss", "--safety_scale",
	"-ns", "--num_stars",
	"-sf", "--starfile",
	"-np", "--num_phi",
	"-nb", "--num_branches",
	"-rs", "--random_seed",
	"-ws", "--write_stars",
	"-wcc", "--write_critical_curves",
	"-wc", "--write_caustics",
	"-wls", "--write_length_scales",
	"-o", "--outfile_prefix"
};

/******************************************************************************
default input option values
******************************************************************************/
int verbose = 1;
dtype smooth_fraction = static_cast<dtype>(1 - ccf.kappa_star / ccf.kappa_tot);



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
		<< "  -h,--help                  Show this help message\n"
		<< "  -v,--verbose               Specify verbosity of output from 0 (none) to 1, 2,\n"
		<< "                             or 3 (low, medium, high). Default value: " << verbose << "\n"
		<< "  -k,--kappa_tot             Specify the total convergence. Default value: " << ccf.kappa_tot << "\n"
		<< "  -y,--shear                 Specify the shear. Default value: " << ccf.shear << "\n"
		<< "  -s,--smooth_fraction       Specify the fraction of convergence due to\n"
		<< "                             smoothly distributed mass. Default value: " << smooth_fraction << "\n"
		<< "  -ks,--kappa_star           Specify the convergence in point mass lenses. If\n"
		<< "                             provided, this overrides any supplied value for\n"
		<< "                             the smooth fraction. Default value: " << ccf.kappa_star << "\n"
		<< "  -t,--theta_star            Specify the size of the Einstein radius of a unit\n"
		<< "                             mass point lens in arbitrary units.\n"
		<< "                             Default value: " << ccf.theta_star << "\n"
		<< "  -mf,--mass_function        Specify the mass function to use for the point\n"
		<< "                             mass lenses. Options are: equal, uniform,\n"
		<< "                             Salpeter, and, Kroupa, or optical_depth.\n"
		<< "                             Default value: " << ccf.mass_function_str << "\n"
		<< "  -ms,--m_solar              Specify the solar mass in arbitrary units.\n"
		<< "                             Default value: " << ccf.m_solar << "\n"
		<< "  -ml,--m_lower              Specify the lower mass cutoff in solar mass units.\n"
		<< "                             Default value: " << ccf.m_lower << "\n"
		<< "  -mh,--m_upper              Specify the upper mass cutoff in solar mass units.\n"
		<< "                             Default value: " << ccf.m_upper << "\n"
		<< "  -r,--rectangular           Specify whether the star field should be\n"
		<< "                             rectangular (1) or circular (0). Default value: " << ccf.rectangular << "\n"
		<< "  -a,--approx                Specify whether terms for alpha_smooth should be\n"
		<< "                             approximated (1) or exact (0). Default value: " << ccf.approx << "\n"
		<< "  -ss,--safety_scale         Specify the ratio of the size of the star field to\n"
		<< "                             the radius of convergence for alpha_smooth.\n"
		<< "                             Default value: " << ccf.safety_scale << "\n"
		<< "  -ns,--num_stars            Specify the number of stars desired.\n"
		<< "                             Default value: " << ccf.num_stars << "\n"
		<< "  -sf,--starfile             Specify the location of a binary file containing\n"
		<< "                             values for num_stars, rectangular, corner,\n"
		<< "                             theta_star, and the star positions and masses, in\n"
		<< "                             an order as defined in this source code.\n"
		<< "                             A whitespace delimited text file where each line\n"
		<< "                             contains the x1 and x2 coordinates and the mass of\n"
		<< "                             a microlens, in units where theta_star = 1, is\n"
		<< "                             also accepted. If provided, this takes precedence\n"
		<< "                             for all star information.\n"
		<< "  -np,--num_phi              Specify the number of steps used to vary phi in\n"
		<< "                             the range [0, 2*pi]. Default value: " << ccf.num_phi << "\n"
		<< "  -nb,--num_branches         Specify the number of branches to use for phi in\n"
		<< "                             the range [0, 2*pi]. Default value: " << ccf.num_branches << "\n"
		<< "  -rs,--random_seed          Specify the random seed for star field generation.\n"
		<< "                             A value of 0 is reserved for star input files.\n"
		<< "  -ws,--write_stars          Specify whether to write stars (1) or not (0).\n"
		<< "                             Default value: " << ccf.write_stars << "\n"
		<< "  -wcc,                      Specify whether to write critical curves (1) or\n"
		<< "   --write_critical_curves   not (0). Default value: " << ccf.write_critical_curves << "\n"
		<< "  -wc,--write_caustics       Specify whether to write caustics (1) or not (0).\n"
		<< "                             Default value: " << ccf.write_caustics << "\n"
		<< "  -wls,                      Specify whether to write magnification length\n"
		<< "   --write_length_scales     scales (1) or not (0). Default value: " << ccf.write_mu_length_scales << "\n"
		<< "  -o,--outfile_prefix        Specify the prefix to be used in output\n"
		<< "                             filenames. Default value: " << ccf.outfile_prefix << "\n";
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
		char* cmdinput = cmd_option_value(argv, argv + argc, argv[i]);

		if (argv[i] == std::string("-k") || argv[i] == std::string("--kappa_tot"))
		{
			try
			{
				set_param("kappa_tot", ccf.kappa_tot, std::stod(cmdinput), verbose);
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
				set_param("shear", ccf.shear, std::stod(cmdinput), verbose);
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
				set_param("kappa_star", ccf.kappa_star, std::stod(cmdinput), verbose);
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
				set_param("theta_star", ccf.theta_star, std::stod(cmdinput), verbose);
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
			set_param("mass_function", ccf.mass_function_str, make_lowercase(cmdinput), verbose);
		}
		else if (argv[i] == std::string("-ms") || argv[i] == std::string("--m_solar"))
		{
			try
			{
				set_param("m_solar", ccf.m_solar, std::stod(cmdinput), verbose);
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
				set_param("m_lower", ccf.m_lower, std::stod(cmdinput), verbose);
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
				set_param("m_upper", ccf.m_upper, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid m_upper input.\n";
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
				set_param("rectangular", ccf.rectangular, std::stoi(cmdinput), verbose);
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
				set_param("approx", ccf.approx, std::stoi(cmdinput), verbose);
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
				set_param("safety_scale", ccf.safety_scale, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid safety_scale input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ns") || argv[i] == std::string("--num_stars"))
		{
			if (cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile"))
			{
				continue;
			}
			try
			{
				set_param("num_stars", ccf.num_stars, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_stars input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-sf") || argv[i] == std::string("--starfile"))
		{
			set_param("starfile", ccf.starfile, cmdinput, verbose);
		}
		else if (argv[i] == std::string("-np") || argv[i] == std::string("--num_phi"))
		{
			try
			{
				set_param("num_phi", ccf.num_phi, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_phi input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-nb") || argv[i] == std::string("--num_branches"))
		{
			try
			{
				set_param("num_branches", ccf.num_branches, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_branches input.\n";
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
				set_param("random_seed", ccf.random_seed, std::stoi(cmdinput), verbose);
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
				set_param("write_stars", ccf.write_stars, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_stars input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wcc") || argv[i] == std::string("--write_critical_curves"))
		{
			try
			{
				set_param("write_critical_curves", ccf.write_critical_curves, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_critical_curves input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wc") || argv[i] == std::string("--write_caustics"))
		{
			try
			{
				set_param("write_caustics", ccf.write_caustics, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_caustics input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wls") || argv[i] == std::string("--write_length_scales"))
		{
			try
			{
				set_param("write_length_scales", ccf.write_mu_length_scales, std::stoi(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_length_scales input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			set_param("outfile_prefix", ccf.outfile_prefix, cmdinput, verbose);
		}
	}

	if (!(cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--starfile")) &&
		!(cmd_option_exists(argv, argv + argc, "-ks") || cmd_option_exists(argv, argv + argc, "--kappa_star")) &&
		(cmd_option_exists(argv, argv + argc, "-s") || cmd_option_exists(argv, argv + argc, "--smooth_fraction")))
	{
		set_param("kappa_star", ccf.kappa_star, (1 - smooth_fraction) * ccf.kappa_tot, verbose);
	}

	print_verbose("\n", verbose, 2);

	/******************************************************************************
	END read in options and values, checking correctness and exiting if necessary
	******************************************************************************/


	/******************************************************************************
	run and save files
	******************************************************************************/
	if (!ccf.run(verbose)) return -1;
	if (!ccf.save(verbose)) return -1;


	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}

