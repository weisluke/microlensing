#pragma once

#include "binomial_coefficients.cuh"
#include "complex.cuh"
#include "fmm.cuh"
#include "mass_functions.cuh"
#include "mass_functions/equal.cuh"
#include "mass_functions/kroupa.cuh"
#include "mass_functions/mass_function_base.cuh" //for massfunctions::MassFunction<T>
#include "mass_functions/salpeter.cuh"
#include "mass_functions/uniform.cuh"
#include "star.cuh"
#include "stopwatch.hpp"
#include "tree_node.cuh"
#include "util/util.cuh"

#include <curand_kernel.h>

#include <algorithm> //for std::min and std::max
#include <chrono> //for setting random seed with clock
#include <cmath>
#include <iostream>
#include <limits> //for std::numeric_limits
#include <memory> //for std::shared_ptr
#include <numbers>
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


private:
	/******************************************************************************
	stopwatch for timing purposes
	******************************************************************************/
	Stopwatch stopwatch;
	double t_elapsed;


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
			std::cout << "Available CUDA capable devices:\n";

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
			print_verbose("More than one CUDA capable device detected. Defaulting to first device.\n", verbose, 2);
		}
		cudaSetDevice(0);
		if (cuda_error("cudaSetDevice", false, __FILE__, __LINE__)) return false;
		cudaGetDeviceProperties(&cuda_device_prop, 0);
		if (cuda_error("cudaGetDeviceProperties", false, __FILE__, __LINE__)) return false;

		print_verbose("Done setting device.\n", verbose, 3);
		return true;
	}

    //optional return or not, so memory can be cleared in destructor without error checking
	bool clear_memory(int verbose, bool return_on_error = true)
	{
		print_verbose("Clearing Microlensing<T> memory...\n", verbose, 3);
		
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

		print_verbose("Done clearing Microlensing<T> memory.\n", verbose, 3);
		return true;
	}

	bool check_input_params(int verbose)
	{
		print_verbose("Checking Microlensing<T> input parameters...\n", verbose, 3);

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

		print_verbose("Done checking Microlensing<T> input parameters.\n", verbose, 3);
		return true;
	}

	bool calculate_derived_params(int verbose)
	{
		print_verbose("Calculating Microlensing<T> derived parameters...\n", verbose, 3);
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
		print_verbose("Done calculating Microlensing<T> derived parameters. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 3);

		return true;
	}

	bool allocate_initialize_memory(int verbose)
	{
		print_verbose("Allocating Microlensing<T> memory...\n", verbose, 3);
		stopwatch.start();

		/******************************************************************************
		allocate memory for stars
		******************************************************************************/
		cudaMallocManaged(&states, num_stars * sizeof(curandState));
		if (cuda_error("cudaMallocManaged(*states)", false, __FILE__, __LINE__)) return false;
		if (stars == nullptr) //if memory wasn't allocated already due to reading a star file
		{
			cudaMallocManaged(&stars, num_stars * sizeof(star<T>));
			if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return false;
		}
		cudaMallocManaged(&temp_stars, num_stars * sizeof(star<T>));
		if (cuda_error("cudaMallocManaged(*temp_stars)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		allocate memory for binomial coefficients
		******************************************************************************/
		cudaMallocManaged(&binomial_coeffs, (2 * treenode::MAX_EXPANSION_ORDER * (2 * treenode::MAX_EXPANSION_ORDER + 3) / 2 + 1) * sizeof(int));
		if (cuda_error("cudaMallocManaged(*binomial_coeffs)", false, __FILE__, __LINE__)) return false;

		t_elapsed = stopwatch.stop();
		print_verbose("Done allocating Microlensing<T> memory. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 3);

		return true;
	}

	bool populate_star_array(int verbose)
	{
		/******************************************************************************
		BEGIN populating star array
		******************************************************************************/

		set_threads(threads, 512);
		set_blocks(threads, blocks, num_stars);

		if (starfile == "")
		{
			print_verbose("Generating star field...\n", verbose, 1);
			stopwatch.start();

			/******************************************************************************
			if random seed was not provided, get one based on the time
			******************************************************************************/
			while (random_seed == 0) //in case it randomly chooses 0, try again
			{
				set_param("random_seed", random_seed, std::chrono::system_clock::now().time_since_epoch().count(), verbose);
			}

			/******************************************************************************
			generate random star field if no star file has been given
			******************************************************************************/
			initialize_curand_states_kernel<T> <<<blocks, threads>>> (states, num_stars, random_seed);
			if (cuda_error("initialize_curand_states_kernel", true, __FILE__, __LINE__)) return false;

			/******************************************************************************
			mass function must be a template for the kernel due to polymorphism
			we therefore must check all possible options
			******************************************************************************/
			if (mass_function_str == "equal")
			{
				generate_star_field_kernel<T, massfunctions::Equal<T>> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, m_lower, m_upper, m_solar);
			}
			else if (mass_function_str == "uniform")
			{
				generate_star_field_kernel<T, massfunctions::Uniform<T>> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, m_lower, m_upper, m_solar);
			}
			else if (mass_function_str == "salpeter")
			{
				generate_star_field_kernel<T, massfunctions::Salpeter<T>> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, m_lower, m_upper, m_solar);
			}
			else if (mass_function_str == "kroupa")
			{
				generate_star_field_kernel<T, massfunctions::Kroupa<T>> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, m_lower, m_upper, m_solar);
			}
			else if (mass_function_str == "optical_depth")
			{
				generate_star_field_kernel<T, massfunctions::OpticalDepth<T>> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, m_lower, m_upper, m_solar);
			}
			else
			{
				std::cerr << "Error. mass_function must be equal, uniform, Salpeter, Kroupa, or optical_depth.\n";
				return false;
			}
			if (cuda_error("generate_star_field_kernel", true, __FILE__, __LINE__)) return false;

			t_elapsed = stopwatch.stop();
			print_verbose("Done generating star field. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 1);
		}
		else
		{
			/******************************************************************************
			ensure random seed is 0 to denote that stars come from external file
			******************************************************************************/
			set_param("random_seed", random_seed, 0, verbose);
		}

		/******************************************************************************
		calculate kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual,
		mean_mass2_actual, and mean_mass2_ln_mass_actual based on star information
		******************************************************************************/
		calculate_star_params<T>(num_stars, rectangular, corner, theta_star, stars,
			kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual, mean_mass2_actual, mean_mass2_ln_mass_actual);

		set_param("kappa_star_actual", kappa_star_actual, kappa_star_actual, verbose);
		set_param("m_lower_actual", m_lower_actual, m_lower_actual, verbose);
		set_param("m_upper_actual", m_upper_actual, m_upper_actual, verbose);
		set_param("mean_mass_actual", mean_mass_actual, mean_mass_actual, verbose);
		set_param("mean_mass2_actual", mean_mass2_actual, mean_mass2_actual, verbose);
		set_param("mean_mass2_ln_mass_actual", mean_mass2_ln_mass_actual, mean_mass2_ln_mass_actual, verbose);

		if (starfile == "")
		{
			if (rectangular)
			{
				corner = Complex<T>(std::sqrt(corner.re / corner.im), std::sqrt(corner.im / corner.re));
				corner *= std::sqrt(std::numbers::pi_v<T> * theta_star * theta_star * num_stars * mean_mass_actual / (4 * kappa_star));
				set_param("corner", corner, corner, verbose);
			}
			else
			{
				corner = corner / corner.abs();
				corner *= std::sqrt(theta_star * theta_star * num_stars * mean_mass_actual / kappa_star);
				set_param("corner", corner, corner, verbose);
			}
		}

		/******************************************************************************
		END populating star array
		******************************************************************************/

		return true;
	}

	bool create_tree(int verbose)
	{
		/******************************************************************************
		BEGIN create root node, then create children and sort stars
		******************************************************************************/

		if (rectangular)
		{
			root_half_length = corner.re > corner.im ? corner.re : corner.im;
		}
		else
		{
			root_half_length = corner.abs();
		}
		set_param("root_half_length", root_half_length, root_half_length * 1.1, verbose); //slight buffer for containing all the stars

		//initialize variables
		tree_levels = 0;
		tree = {};
		num_nodes = {};

		/******************************************************************************
		push empty pointer into tree, add 1 to number of nodes, and allocate memory
		******************************************************************************/
		tree.push_back(nullptr);
		num_nodes.push_back(1);
		cudaMallocManaged(&tree.back(), num_nodes.back() * sizeof(TreeNode<T>));
		if (cuda_error("cudaMallocManaged(*tree)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		initialize root node
		******************************************************************************/
		tree[0][0] = TreeNode<T>(Complex<T>(0, 0), root_half_length, 0);
		tree[0][0].numstars = num_stars;


		int* max_num_stars_in_level = nullptr;
		int* min_num_stars_in_level = nullptr;
		int* num_nonempty_nodes = nullptr;
		cudaMallocManaged(&max_num_stars_in_level, sizeof(int));
		if (cuda_error("cudaMallocManaged(*max_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		cudaMallocManaged(&min_num_stars_in_level, sizeof(int));
		if (cuda_error("cudaMallocManaged(*min_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		cudaMallocManaged(&num_nonempty_nodes, sizeof(int));
		if (cuda_error("cudaMallocManaged(*num_nonempty_nodes)", false, __FILE__, __LINE__)) return false;

		print_verbose("Creating children and sorting stars...\n", verbose, 1);
		stopwatch.start();

		do
		{
			print_verbose("Processing level " << tree_levels << "\n", verbose, 3);

			*max_num_stars_in_level = 0;
			*min_num_stars_in_level = num_stars;
			*num_nonempty_nodes = 0;

			set_threads(threads, 512);
			set_blocks(threads, blocks, num_nodes[tree_levels]);
			treenode::get_node_star_info_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels],
				num_nonempty_nodes, min_num_stars_in_level, max_num_stars_in_level);
			if (cuda_error("get_node_star_info_kernel", true, __FILE__, __LINE__)) return false;

			print_verbose("Maximum number of stars in a node and its neighbors is " << *max_num_stars_in_level << "\n", verbose, 3);
			print_verbose("Minimum number of stars in a node and its neighbors is " << *min_num_stars_in_level << "\n", verbose, 3);

			if (*max_num_stars_in_level > treenode::MAX_NUM_STARS_DIRECT)
			{
				print_verbose("Number of non-empty children: " << *num_nonempty_nodes * treenode::MAX_NUM_CHILDREN << "\n", verbose, 3);

				print_verbose("Allocating memory for children...\n", verbose, 3);
				tree.push_back(nullptr);
				num_nodes.push_back(*num_nonempty_nodes * treenode::MAX_NUM_CHILDREN);
				cudaMallocManaged(&tree.back(), num_nodes.back() * sizeof(TreeNode<T>));
				if (cuda_error("cudaMallocManaged(*tree)", false, __FILE__, __LINE__)) return false;

				print_verbose("Creating children...\n", verbose, 3);
				(*num_nonempty_nodes)--; //subtract one since value is size of array, and instead needs to be the first allocatable element
				set_threads(threads, 512);
				set_blocks(threads, blocks, num_nodes[tree_levels]);
				treenode::create_children_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels], num_nonempty_nodes, tree[tree_levels + 1]);
				if (cuda_error("create_children_kernel", true, __FILE__, __LINE__)) return false;

				print_verbose("Sorting stars...\n", verbose, 3);
				set_threads(threads, std::ceil(1.0 * 512 / *max_num_stars_in_level), std::min(512, *max_num_stars_in_level));
				set_blocks(threads, blocks, num_nodes[tree_levels], std::min(512, *max_num_stars_in_level));
				treenode::sort_stars_kernel<T> <<<blocks, threads, (threads.x + threads.x + threads.x * treenode::MAX_NUM_CHILDREN) * sizeof(int)>>> (tree[tree_levels], num_nodes[tree_levels], stars, temp_stars);
				if (cuda_error("sort_stars_kernel", true, __FILE__, __LINE__)) return false;

				tree_levels++;

				print_verbose("Setting neighbors...\n", verbose, 3);
				set_threads(threads, 512);
				set_blocks(threads, blocks, num_nodes[tree_levels]);
				treenode::set_neighbors_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels]);
				if (cuda_error("set_neighbors_kernel", true, __FILE__, __LINE__)) return false;
			}
		} while (*max_num_stars_in_level > treenode::MAX_NUM_STARS_DIRECT);
		set_param("tree_levels", tree_levels, tree_levels, verbose);


		cudaFree(max_num_stars_in_level);
		if (cuda_error("cudaFree(*max_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		max_num_stars_in_level = nullptr;
		
		cudaFree(min_num_stars_in_level);
		if (cuda_error("cudaFree(*min_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		min_num_stars_in_level = nullptr;
		
		cudaFree(num_nonempty_nodes);
		if (cuda_error("cudaFree(*num_nonempty_nodes)", false, __FILE__, __LINE__)) return false;
		num_nonempty_nodes = nullptr;


		t_elapsed = stopwatch.stop();
		print_verbose("Done creating children and sorting stars. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 1);

		/******************************************************************************
		END create root node, then create children and sort stars
		******************************************************************************/

		expansion_order = std::ceil(2 * std::log2(theta_star)
									+ std::log2(mean_mass2) - std::log2(mean_mass)
									+ tree_levels
									- std::log2(root_half_length) - std::log2(alpha_error));
		set_param("expansion_order", expansion_order, expansion_order, verbose);
		if (expansion_order < 3)
		{
			std::cerr << "Error. Expansion order needs to be >= 3\n";
			return false;
		}
		else if (expansion_order > treenode::MAX_EXPANSION_ORDER)
		{
			std::cerr << "Error. Maximum allowed expansion order is " << treenode::MAX_EXPANSION_ORDER << "\n";
			return false;
		}

		print_verbose("Calculating binomial coefficients...\n", verbose, 3);
		calculate_binomial_coeffs(binomial_coeffs, 2 * expansion_order);
		print_verbose("Done calculating binomial coefficients.\n", verbose, 3);


		/******************************************************************************
		BEGIN calculating multipole and local coefficients
		******************************************************************************/

		print_verbose("Calculating multipole and local coefficients...\n", verbose, 1);
		stopwatch.start();

		for (int i = tree_levels; i >= 0; i--)
		{
			set_threads(threads, 16, expansion_order + 1);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1));
			fmm::calculate_multipole_coeffs_kernel<T> <<<blocks, threads, 16 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, stars);

			set_threads(threads, 4, expansion_order + 1, treenode::MAX_NUM_CHILDREN);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1), treenode::MAX_NUM_CHILDREN);
			fmm::calculate_M2M_coeffs_kernel<T> <<<blocks, threads, 4 * treenode::MAX_NUM_CHILDREN * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs);
		}

		/******************************************************************************
		local coefficients are non zero only starting at the second level
		******************************************************************************/
		for (int i = 2; i <= tree_levels; i++)
		{
			set_threads(threads, 16, expansion_order + 1);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1));
			fmm::calculate_L2L_coeffs_kernel<T> <<<blocks, threads, 16 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs);

			set_threads(threads, 1, expansion_order + 1, treenode::MAX_NUM_SAME_LEVEL_INTERACTION_LIST);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1), treenode::MAX_NUM_SAME_LEVEL_INTERACTION_LIST);
			fmm::calculate_M2L_coeffs_kernel<T> <<<blocks, threads, 1 * treenode::MAX_NUM_SAME_LEVEL_INTERACTION_LIST * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs);

			set_threads(threads, 4, expansion_order + 1, treenode::MAX_NUM_DIFFERENT_LEVEL_INTERACTION_LIST);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1), treenode::MAX_NUM_DIFFERENT_LEVEL_INTERACTION_LIST);
			fmm::calculate_P2L_coeffs_kernel<T> <<<blocks, threads, 4 * treenode::MAX_NUM_DIFFERENT_LEVEL_INTERACTION_LIST * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs, stars);
		}
		if (cuda_error("calculate_coeffs_kernels", true, __FILE__, __LINE__)) return false;

		t_elapsed = stopwatch.stop();
		print_verbose("Done calculating multipole and local coefficients. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 1);

		/******************************************************************************
		END calculating multipole and local coefficients
		******************************************************************************/

		return true;
	}

	bool write_files(int verbose, const std::string& class_name = "microlensing")
	{
		std::string fname;

		print_verbose("Writing Microlensing<T> parameter info...\n", verbose, 2);
		fname = outfile_prefix + class_name + "_parameter_info.txt";
		outfile.open(fname);
		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		outfile << "kappa_tot " << kappa_tot << "\n";
		outfile << "shear " << shear << "\n";
		outfile << "mu_ave " << mu_ave << "\n";
		outfile << "smooth_fraction " << (1 - kappa_star / kappa_tot) << "\n";
		outfile << "kappa_star " << kappa_star << "\n";
		if (starfile == "")
		{
			outfile << "kappa_star_actual " << kappa_star_actual << "\n";
		}
		outfile << "theta_star " << theta_star << "\n";
		outfile << "random_seed " << random_seed << "\n";
		if (starfile == "")
		{
			outfile << "mass_function " << mass_function_str << "\n";
			if (mass_function_str == "salpeter" || mass_function_str == "kroupa")
			{
				outfile << "m_solar " << m_solar << "\n";
			}
			outfile << "m_lower " << m_lower << "\n";
			outfile << "m_upper " << m_upper << "\n";
			outfile << "mean_mass " << mean_mass << "\n";
			outfile << "mean_mass2 " << mean_mass2 << "\n";
			outfile << "mean_mass2_ln_mass " << mean_mass2_ln_mass << "\n";
		}
		outfile << "m_lower_actual " << m_lower_actual << "\n";
		outfile << "m_upper_actual " << m_upper_actual << "\n";
		outfile << "mean_mass_actual " << mean_mass_actual << "\n";
		outfile << "mean_mass2_actual " << mean_mass2_actual << "\n";
		outfile << "mean_mass2_ln_mass_actual " << mean_mass2_ln_mass_actual << "\n";
		outfile << "num_stars " << num_stars << "\n";
		if (rectangular)
		{
			outfile << "corner_x1 " << corner.re << "\n";
			outfile << "corner_x2 " << corner.im << "\n";
			if (approx)
			{
				outfile << "taylor_smooth " << taylor_smooth << "\n";
			}
		}
		else
		{
			outfile << "rad " << corner.abs() << "\n";
		}
		outfile << "safety_scale " << safety_scale << "\n";
		outfile << "alpha_error " << alpha_error << "\n";
		outfile << "expansion_order " << expansion_order << "\n";
		outfile << "root_half_length " << root_half_length << "\n";
		outfile << "tree_levels " << tree_levels << "\n";
		outfile.close();
		print_verbose("Done writing Microlensing<T> parameter info to file " << fname << "\n", verbose, 1);

		if (write_stars)
		{
			print_verbose("Writing Microlensing<T> star info...\n", verbose, 2);
			fname = outfile_prefix + class_name + "_stars" + outfile_type;
			if (!write_star_file<T>(num_stars, rectangular, corner, theta_star, stars, fname))
			{
				std::cerr << "Error. Unable to write star info to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing Microlensing<T> star info to file " << fname << "\n", verbose, 1);
		}

		return true;
	}


public:

	/******************************************************************************
	class initializer is empty
	******************************************************************************/
	Microlensing()
	{

	}

	/******************************************************************************
	class destructor clears memory with no output or error checking
	******************************************************************************/
	~Microlensing()
	{
		clear_memory(0, false);
	}

	/******************************************************************************
	copy constructor sets this object's dynamic memory pointers to null
	******************************************************************************/
	Microlensing(const Microlensing& other)
	{
		states = nullptr;
		stars = nullptr;
		temp_stars = nullptr;

		binomial_coeffs = nullptr;

		tree = {};
	}

	/******************************************************************************
	copy assignment sets this object's dynamic memory pointers to null
	******************************************************************************/
	Microlensing& operator=(const Microlensing& other)
	{
        if (this == &other) return *this;

		states = nullptr;
		stars = nullptr;
		temp_stars = nullptr;

		binomial_coeffs = nullptr;

		tree = {};

		return *this;
	}

	int get_num_stars()			{return num_stars;}
	Complex<T> get_corner()		{if (rectangular) {return corner;} else {return Complex<T>(corner.abs(), 0);}}
	star<T>* get_stars()		{return stars;}

};