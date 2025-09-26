#pragma once

#include "complex.cuh"
#include "mass_functions.cuh"
#include "mass_functions/mass_function_base.cuh" //for massfunctions::MassFunction<T>
#include "star.cuh"
#include "stopwatch.hpp"
#include "tree_node.cuh"

#include <limits> //for std::numeric_limits
#include <memory> //for std::shared_ptr
#include <string>
#include <vector>


template <typename T>
class Microlensing
{

public:
	/******************************************************************************
	default input variables
	******************************************************************************/
	T kappa_tot = static_cast<T>(0.3);
	T shear = static_cast<T>(0.3);
	T kappa_star = static_cast<T>(0.27);
	T theta_star = static_cast<T>(1);
	std::string mass_function_str = "equal";
	T m_solar = static_cast<T>(1);
	T m_lower = static_cast<T>(0.01);
	T m_upper = static_cast<T>(50);
	int rectangular = 0; //whether star field is rectangular or circular
	int approx = 1; //whether terms for alpha_smooth are exact or approximate
    //for IPM, ratio of the size of the star field to the size of the shooting region
    //for CCF, ratio of the size of the star field to the radius of convergence for alpha_smooth
	T safety_scale = static_cast<T>(1.37);
	std::string starfile = "";
	int random_seed = 0;
	int write_stars = 1;
	std::string outfile_prefix = "./";


protected:
	/******************************************************************************
	protected for IPM, will be made public for CCF
	******************************************************************************/
	int num_stars = 137;

	/******************************************************************************
	constant variables
	******************************************************************************/
	const std::string outfile_type = ".bin";
	//arbitrary limit to the expansion order to avoid numerical precision loss from high degree polynomials
	const int MAX_TAYLOR_SMOOTH = 101;

	/******************************************************************************
	variables for cuda device, kernel threads, and kernel blocks
	******************************************************************************/
	cudaDeviceProp cuda_device_prop;
	dim3 threads;
	dim3 blocks;

	/******************************************************************************
	stopwatch for timing purposes
	******************************************************************************/
	Stopwatch stopwatch;
	double t_elapsed;

	/******************************************************************************
	derived variables
	******************************************************************************/
	std::shared_ptr<massfunctions::MassFunction<T>> mass_function;
	T mean_mass; //<m>
	T mean_mass2; //<m^2>
	T mean_mass2_ln_mass; //<m^2 * ln(m)>

	T kappa_star_actual;
	T m_lower_actual;
	T m_upper_actual;
	T mean_mass_actual;
	T mean_mass2_actual;
	T mean_mass2_ln_mass_actual;

	T mu_ave;
	Complex<T> corner;
	int taylor_smooth;

	T alpha_error; //error in the deflection angle

	int expansion_order;

	T root_half_length;
	int tree_levels;
	std::vector<TreeNode<T>*> tree; //members of the tree will need their memory freed
	std::vector<int> num_nodes;

	/******************************************************************************
	dynamic memory
	******************************************************************************/
	curandState* states = nullptr;
	star<T>* stars = nullptr;
	star<T>* temp_stars = nullptr;

	int* binomial_coeffs = nullptr;



    bool set_cuda_devices(int verbose)
	{
		print_verbose("Setting device...\n", verbose, 3);

		/******************************************************************************
		check that a CUDA capable device is present
		******************************************************************************/
		int n_devices = 0;

		cudaGetDeviceCount(&n_devices);
		if (cuda_error("cudaGetDeviceCount", false, __FILE__, __LINE__)) return false;

		if (n_devices < 1)
		{
			std::cerr << "Error. No CUDA capable devices detected.\n";
			return false;
		}

		if (verbose >= 3)
		{
			std::cout << "Available CUDA capable devices:\n\n";

			for (int i = 0; i < n_devices; i++)
			{
				cudaDeviceProp prop;
				cudaGetDeviceProperties(&prop, i);
				if (cuda_error("cudaGetDeviceProperties", false, __FILE__, __LINE__)) return false;

				show_device_info(i, prop);
			}
		}

		if (n_devices > 1)
		{
			print_verbose("More than one CUDA capable device detected. Defaulting to first device.\n\n", verbose, 2);
		}
		cudaSetDevice(0);
		if (cuda_error("cudaSetDevice", false, __FILE__, __LINE__)) return false;
		cudaGetDeviceProperties(&cuda_device_prop, 0);
		if (cuda_error("cudaGetDeviceProperties", false, __FILE__, __LINE__)) return false;

		print_verbose("Done setting device.\n\n", verbose, 3);
		return true;
	}

    //optional return or not, so memory can be cleared in destructor without error checking
	bool clear_memory(int verbose, bool return_on_error = true)
	{
		print_verbose("Clearing memory...\n", verbose, 3);
		
		/******************************************************************************
		free memory and set variables to nullptr
		******************************************************************************/

		cudaFree(states);
		if (return_on_error && cuda_error("cudaFree(*states)", false, __FILE__, __LINE__)) return false;
		states = nullptr;
		
		cudaFree(stars);
		if (return_on_error && cuda_error("cudaFree(*stars)", false, __FILE__, __LINE__)) return false;
		stars = nullptr;
		
		cudaFree(temp_stars);
		if (return_on_error && cuda_error("cudaFree(*temp_stars)", false, __FILE__, __LINE__)) return false;
		temp_stars = nullptr;
		
		cudaFree(binomial_coeffs);
		if (return_on_error && cuda_error("cudaFree(*binomial_coeffs)", false, __FILE__, __LINE__)) return false;
		binomial_coeffs = nullptr;

		for	(int i = 0; i < tree.size(); i++) //for every level in the tree, free the memory for the nodes
		{
			cudaFree(tree[i]);
			if (return_on_error && cuda_error("cudaFree(*tree[i])", false, __FILE__, __LINE__)) return false;
			tree[i] = nullptr;
		}

		print_verbose("Done clearing memory.\n\n", verbose, 3);
		return true;
	}

	bool check_input_params(int verbose)
	{
		print_verbose("Checking input parameters...\n", verbose, 3);

		if (kappa_tot < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. kappa_tot must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (kappa_star < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. kappa_star must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}
		if (starfile == "" && kappa_star > kappa_tot)
		{
			std::cerr << "Error. kappa_star must be <= kappa_tot\n";
			return false;
		}

		if (starfile == "" && theta_star < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. theta_star must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (starfile == "" && !massfunctions::MASS_FUNCTIONS<T>.count(mass_function_str))
		{
			std::cerr << "Error. mass_function must be equal, uniform, Salpeter, Kroupa, or optical_depth.\n";
			return false;
		}

		if (starfile == "" && m_solar < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. m_solar must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (starfile == "" && m_lower < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. m_lower must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (starfile == "" && m_upper < m_lower)
		{
			std::cerr << "Error. m_upper must be >= m_lower.\n";
			return false;
		}

		if (starfile == "" && rectangular != 0 && rectangular != 1)
		{
			std::cerr << "Error. rectangular must be 1 (rectangular) or 0 (circular).\n";
			return false;
		}

		if (approx != 0 && approx != 1)
		{
			std::cerr << "Error. approx must be 1 (approximate) or 0 (exact).\n";
			return false;
		}

		/******************************************************************************
		if the alpha_smooth comes from a rectangular mass sheet, finding the caustics
		requires a Taylor series approximation to alpha_smooth. a bound on the error of
		that series necessitates having some minimum cutoff here for the ratio of the
		size of the star field to (IPM) the size of the shooting rectangle or (CCF) the
		radius of convergence for alpha_smooth
		******************************************************************************/
		if (safety_scale < 1.1)
		{
			std::cerr << "Error. safety_scale must be >= 1.1\n";
			return false;
		}

		if (write_stars != 0 && write_stars != 1)
		{
			std::cerr << "Error. write_stars must be 1 (true) or 0 (false).\n";
			return false;
		}

		print_verbose("Done checking input parameters.\n\n", verbose, 3);
		return true;
	}

	bool calculate_derived_params(int verbose)
	{
		print_verbose("Calculating derived parameters...\n", verbose, 3);
		stopwatch.start();

		/******************************************************************************
		average magnification of the system
		******************************************************************************/
		set_param("mu_ave", mu_ave, 1 / ((1 - kappa_tot) * (1 - kappa_tot) - shear * shear), verbose);

		/******************************************************************************
		if star file is not specified, set the mass function, mean_mass, mean_mass2,
		and mean_mass2_ln_mass
		******************************************************************************/
		if (starfile == "")
		{
			if (mass_function_str == "equal")
			{
				set_param("m_lower", m_lower, 1, verbose);
				set_param("m_upper", m_upper, 1, verbose);
			}
			else
			{
				set_param("m_lower", m_lower, m_lower * m_solar, verbose);
				set_param("m_upper", m_upper, m_upper * m_solar, verbose);
			}

			/******************************************************************************
			determine mass function, <m>, <m^2>, and <m^2 * ln(m)>
			******************************************************************************/
			mass_function = massfunctions::MASS_FUNCTIONS<T>.at(mass_function_str);
			set_param("mean_mass", mean_mass, mass_function->mean_mass(m_lower, m_upper, m_solar), verbose);
			set_param("mean_mass2", mean_mass2, mass_function->mean_mass2(m_lower, m_upper, m_solar), verbose);
			set_param("mean_mass2_ln_mass", mean_mass2_ln_mass, mass_function->mean_mass2_ln_mass(m_lower, m_upper, m_solar), verbose);
		}
		/******************************************************************************
		if star file is specified, check validity of values and set num_stars,
		rectangular, corner, theta_star, stars, kappa_star, m_lower, m_upper,
		mean_mass, mean_mass2, and mean_mass2_ln_mass based on star information
		******************************************************************************/
		else
		{
			print_verbose("Calculating some parameter values based on star input file " << starfile << "\n", verbose, 3);

			if (!read_star_file<T>(num_stars, rectangular, corner, theta_star, stars,
				kappa_star, m_lower, m_upper, mean_mass, mean_mass2, mean_mass2_ln_mass, starfile))
			{
				std::cerr << "Error. Unable to read star field parameters from file " << starfile << "\n";
				return false;
			}

			set_param("num_stars", num_stars, num_stars, verbose);
			set_param("rectangular", rectangular, rectangular, verbose);
			set_param("corner", corner, corner, verbose);
			set_param("theta_star", theta_star, theta_star, verbose);
			set_param("kappa_star", kappa_star, kappa_star, verbose);
			if (kappa_star > kappa_tot)
			{
				std::cerr << "Warning. kappa_star > kappa_tot\n";
			}
			set_param("m_lower", m_lower, m_lower, verbose);
			set_param("m_upper", m_upper, m_upper, verbose);
			set_param("mean_mass", mean_mass, mean_mass, verbose);
			set_param("mean_mass2", mean_mass2, mean_mass2, verbose);
			set_param("mean_mass2_ln_mass", mean_mass2_ln_mass, mean_mass2_ln_mass, verbose);

			print_verbose("Done calculating some parameter values based on star input file " << starfile << "\n", verbose, 3);
		}

		t_elapsed = stopwatch.stop();
		print_verbose("Done calculating derived parameters. Elapsed time: " << t_elapsed << " seconds.\n\n", verbose, 3);

		return true;
	}


public:

	/******************************************************************************
	class initializer is empty
	******************************************************************************/
	Microlensing()
	{

	}

};