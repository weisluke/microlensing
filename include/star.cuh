#pragma once

#include "complex.cuh"
#include "mass_functions.cuh"
#include "util/util.cuh"

#include <curand_kernel.h>

#include <algorithm> //for std::min and std::max
#include <cmath>
#include <cstdint> //for std::uintmax_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits> //for std::numeric_limits
#include <new>
#include <numbers>
#include <string>
#include <system_error> //for std::error_code


/******************************************************************************
structure to hold position and mass of a star
******************************************************************************/
template <typename T>
struct star
{
	Complex<T> position;
	T mass;
};


/******************************************************************************
initialize curand states for random star field generation

\param states -- pointer to array of curand states
\param nstars -- number of states (stars) to initialize
\param seed -- random seed to use
******************************************************************************/
template<typename T>
__global__ void initialize_curand_states_kernel(curandState* states, int nstars, int seed)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int i = x_index; i < nstars; i += x_stride)
	{
		curand_init(seed, i, 0, &states[i]);
	}
}

/******************************************************************************
generate random star field

\param states -- pointer to array of curand states
\param stars -- pointer to array of point mass lenses
\param nstars -- number of stars to generate
\param rectangular -- whether the star field is rectangular or not
\param corner -- corner of the rectangular field of point mass lenses
\param m_lower -- lower mass cutoff for the distribution in arbitrary units
\param m_upper -- upper mass cutoff for the distribution in arbitrary units
\param m_solar -- solar mass in arbitrary units
******************************************************************************/
template <typename T, class U>
__global__ void generate_star_field_kernel(curandState* states, star<T>* stars, int nstars, int rectangular, Complex<T> corner, 
	T m_lower, T m_upper, T m_solar)
{
	U mass_function;
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int i = x_index; i < nstars; i += x_stride)
	{
		T x1;
		T x2;

		if (rectangular)
		{
			/******************************************************************************
			random positions in the range [-corner, corner]
			******************************************************************************/
			x1 = curand_uniform_double(&states[i]) * 2 * corner.re - corner.re;
			x2 = curand_uniform_double(&states[i]) * 2 * corner.im - corner.im;
		}
		else
		{
			/******************************************************************************
			random angle
			******************************************************************************/
			T a = curand_uniform_double(&states[i]) * 2 * std::numbers::pi_v<T>;
			/******************************************************************************
			random radius uses square root of random number as numbers need to be evenly
			dispersed in 2-D space
			******************************************************************************/
			T r = sqrt(curand_uniform_double(&states[i])) * corner.abs();

			/******************************************************************************
			transform to Cartesian coordinates
			******************************************************************************/
			x1 = r * cos(a);
			x2 = r * sin(a);
		}

		stars[i].position = Complex<T>(x1, x2);
		stars[i].mass = mass_function.mass(curand_uniform_double(&states[i]), m_lower, m_upper, m_solar);
	}
}

/******************************************************************************
determines star field parameters from the given array

\param nstars -- number of point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- corner of the rectangular field of point mass lenses
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param m_lower -- lower mass cutoff
\param m_upper -- upper mass cutoff
\param mean_mass -- mean mass <m>
\param mean_mass2 -- mean mass squared <m^2>
\param mean_mass2_ln_mass -- mean mass squared * ln(mass) <m^2 * ln(m)>
******************************************************************************/
template <typename T>
void calculate_star_params(int nstars, int rectangular, Complex<T> corner, T theta, star<T>* stars,
	T& kappastar, T& m_lower, T& m_upper, T& mean_mass, T& mean_mass2, T& mean_mass2_ln_mass)
{
	m_lower = std::numeric_limits<T>::max();
	m_upper = std::numeric_limits<T>::min();
	
	/******************************************************************************
	use doubles to avoid issues for floats when all the stars are of unit mass and
	the number of stars is greater than 2^24. incrementation fails at that value
	due to the mantissa needing to be 2^-24, smaller than the 23 mantissa bits of a
	float allow
	******************************************************************************/
	double m_tot = 0;
	double m2_tot = 0;
	double m2_ln_mass_tot = 0;

	for (int i = 0; i < nstars; i++)
	{
		m_tot += stars[i].mass;
		m2_tot += stars[i].mass * stars[i].mass;
		m2_ln_mass_tot += stars[i].mass * stars[i].mass * std::log(stars[i].mass);
		m_lower = std::min(m_lower, stars[i].mass);
		m_upper = std::max(m_upper, stars[i].mass);
	}
	mean_mass = m_tot / nstars;
	mean_mass2 = m2_tot / nstars;
	mean_mass2_ln_mass = m2_ln_mass_tot / nstars;

	if (rectangular)
	{
		kappastar = m_tot * std::numbers::pi_v<T> * theta * theta / (4 * corner.re * corner.im);
	}
	else
	{
		kappastar = m_tot * theta * theta / (corner.abs() * corner.abs());
	}
}

/******************************************************************************
read binary star field file

\param nstars -- number of point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- corner of the rectangular field of point mass lenses
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param starfile -- location of the star field file

\return bool -- true if file is successfully read, false if not
******************************************************************************/
template <typename T>
bool read_star_file_bin(int& nstars, int& rectangular, Complex<T>& corner, T& theta, star<T>*& stars, const std::string& starfile)
{
	std::filesystem::path starpath = starfile;

	if (starpath.extension() != ".bin")
	{
		std::cerr << "Error. Star input file " << starfile << " is not a .bin file.\n";
		return false;
	}

	std::error_code err;
	std::uintmax_t fsize = std::filesystem::file_size(starfile, err);

	if (err)
	{
		std::cerr << "Error determining size of star input file " << starfile << "\n";
		return false;
	}

	std::ifstream infile;
	infile.open(starfile, std::ios_base::binary);
	if (!infile.is_open())
	{
		std::cerr << "Error. Failed to open file " << starfile << "\n";
		return false;
	}

	/******************************************************************************
	first item in the file is the number of stars
	******************************************************************************/
	infile.read((char*)(&nstars), sizeof(int));
	if (nstars < 1)
	{
		std::cerr << "Error. Invalid num_stars input. num_stars must be an integer > 0\n";
		return false;
	}

	/******************************************************************************
	allocate memory for stars if needed
	******************************************************************************/
	if (stars == nullptr)
	{
		cudaMallocManaged(&stars, nstars * sizeof(star<T>));
		if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return false;
	}

	/******************************************************************************
	second item in the file is whether the star field is rectangular
	******************************************************************************/
	infile.read((char*)(&rectangular), sizeof(int));
	if (rectangular != 0 && rectangular != 1)
	{
		std::cerr << "Error. Invalid rectangular input. rectangular must be 1 (rectangular) or 0 (circular).\n";
		return false;
	}


	/******************************************************************************
	objects in the file are nstars + rectangular + corner + theta + stars
	******************************************************************************/
	if (fsize == sizeof(int) + sizeof(int) + sizeof(Complex<T>) + sizeof(T) + nstars * sizeof(star<T>))
	{
		/******************************************************************************
		third item in the file is the corner of the star field
		******************************************************************************/
		infile.read((char*)(&corner), sizeof(Complex<T>));
		/******************************************************************************
		fourth item in the file is the size of the Einstein radius of a unit mass point
		lens
		******************************************************************************/
		infile.read((char*)(&theta), sizeof(T));

		infile.read((char*)stars, nstars * sizeof(star<T>));
	}
	else if (fsize == sizeof(int) + sizeof(int) + sizeof(Complex<float>) + sizeof(float) + nstars * sizeof(star<float>))
	{
		Complex<float> temp_corner;
		infile.read((char*)(&temp_corner), sizeof(Complex<float>));
		corner = temp_corner;

		float temp_theta;
		infile.read((char*)(&temp_theta), sizeof(float));
		theta = static_cast<T>(temp_theta);

		star<float>* temp_stars = new (std::nothrow) star<float>[nstars];
		if (!temp_stars)
		{
			std::cerr << "Error. Memory allocation for *temp_stars failed.\n";
			return false;
		}
		infile.read((char*)temp_stars, nstars * sizeof(star<float>));
		for (int i = 0; i < nstars; i++)
		{
			stars[i].position = temp_stars[i].position;
			stars[i].mass = static_cast<T>(temp_stars[i].mass);
		}
		delete[] temp_stars;
		temp_stars = nullptr;
	}
	else if (fsize == sizeof(int) + sizeof(int) + sizeof(Complex<double>) + sizeof(double) + nstars * sizeof(star<double>))
	{
		Complex<double> temp_corner;
		infile.read((char*)(&temp_corner), sizeof(Complex<double>));
		corner = temp_corner;

		double temp_theta;
		infile.read((char*)(&temp_theta), sizeof(double));
		theta = static_cast<T>(temp_theta);

		star<double>* temp_stars = new (std::nothrow) star<double>[nstars];
		if (!temp_stars)
		{
			std::cerr << "Error. Memory allocation for *temp_stars failed.\n";
			return false;
		}
		infile.read((char*)temp_stars, nstars * sizeof(star<double>));
		for (int i = 0; i < nstars; i++)
		{
			stars[i].position = temp_stars[i].position;
			stars[i].mass = static_cast<T>(temp_stars[i].mass);
		}
		delete[] temp_stars;
		temp_stars = nullptr;
	}
	else
	{
		std::cerr << "Error. Star input file " << starfile << " does not contain validly formatted single or double precision stars and accompanying information.\n";
		return false;
	}

	infile.close();

	if (corner.re < 0 || corner.im < 0)
	{
		std::cerr << "Error. Real and imaginary parts of the corner of the star field must both be >= 0\n";
		return false;
	}
	if (theta < std::numeric_limits<T>::min())
	{
		std::cerr << "Error. theta_star must be >= " << std::numeric_limits<T>::min() << "\n";
		return false;
	}

	return true;
}

/******************************************************************************
read txt star field file

\param nstars -- number of point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- corner of the rectangular field of point mass lenses
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param starfile -- location of the star field file

\return bool -- true if file is successfully read, false if not
******************************************************************************/
template <typename T>
bool read_star_file_txt(int& nstars, int& rectangular, Complex<T>& corner, T& theta, star<T>*& stars, const std::string& starfile)
{
	std::filesystem::path starpath = starfile;

	if (starpath.extension() != ".txt")
	{
		std::cerr << "Error. Star input file " << starfile << " is not a .txt file.\n";
		return false;
	}

	/******************************************************************************
	set parameters that will be modified based on star input file
	******************************************************************************/
	nstars = 0;
	T x1, x2, m;

	std::ifstream infile;
	infile.open(starfile);
	if (!infile.is_open())
	{
		std::cerr << "Error. Failed to open file " << starfile << "\n";
		return false;
	}

	while (infile >> x1 >> x2 >> m)
	{
		nstars++;
	}
	infile.close();

	if (nstars < 1)
	{
		std::cerr << "Error. Invalid num_stars input. num_stars must be an integer > 0\n";
		return false;
	}


	/******************************************************************************
	allocate memory for stars if needed
	******************************************************************************/
	if (stars == nullptr)
	{
		cudaMallocManaged(&stars, nstars * sizeof(star<T>));
		if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return false;
	}


	/******************************************************************************
	set parameters that will be modified based on star input file
	******************************************************************************/
	T max_x1 = 0;
	T max_x2 = 0;
	T max_rad = 0;
	double total_mass = 0;
	double I_stars = 0;

	infile.open(starfile);
	if (!infile.is_open())
	{
		std::cerr << "Error. Failed to open file " << starfile << "\n";
		return false;
	}

	for	(int i = 0; i < nstars; i++)
	{
		infile >> x1 >> x2 >> m;
		stars[i].position = Complex<T>(x1, x2);
		stars[i].mass = m;
		total_mass += m;
		I_stars += m * (x1 * x1 + x2 * x2); //moment of inertia of a point mass
		max_x1 = std::max(max_x1, std::abs(x1));
		max_x2 = std::max(max_x2, std::abs(x2));
		max_rad = std::max(max_rad, std::hypot(x1, x2));
	}
	infile.close();

	/******************************************************************************
	determine whether a star field is circular or rectangular by using the moment
	of inertia
	******************************************************************************/
	double I_circ = total_mass * max_rad * max_rad / 2; //moment of inertia of a disk of radius max_rad
	double I_rect = total_mass * (max_x1 * max_x1 + max_x2 * max_x2) / 3; //moment of inertia of a rectangle with corner (max_x1, max_x2)

	if (I_stars / I_circ > I_stars / I_rect)
	{
		rectangular = 0;
		corner.re = max_rad;
		corner.im = 0;
	}
	else
	{
		rectangular = 1;
		corner.re = max_x1;
		corner.im = max_x2;
	}

	theta = 1; //assume all positions are in units of the Einstein radius of a unit mass
	
	return true;
}

/******************************************************************************
read star field file

\param nstars -- number of point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- corner of the rectangular field of point mass lenses
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param m_lower -- lower mass cutoff
\param m_upper -- upper mass cutoff
\param mean_mass -- mean mass <m>
\param mean_mass2 -- mean mass squared <m^2>
\param mean_mass2_ln_mass -- mean mass squared * ln(mass) <m^2 * ln(m)>
\param starfile -- location of the star field file

\return bool -- true if file is successfully read, false if not
******************************************************************************/
template <typename T>
bool read_star_file(int& nstars, int& rectangular, Complex<T>& corner, T& theta, star<T>*& stars,
	T& kappastar, T& m_lower, T& m_upper, T& mean_mass, T& mean_mass2, T& mean_mass2_ln_mass, const std::string& starfile)
{
	std::filesystem::path starpath = starfile;

	if (starpath.extension() == ".bin")
	{
		if (!read_star_file_bin<T>(nstars, rectangular, corner, theta, stars, starfile))
		{
			return false;
		}
	}
	else if (starpath.extension() == ".txt")
	{
		if (!read_star_file_txt<T>(nstars, rectangular, corner, theta, stars, starfile))
		{
			return false;
		}
	}
	else
	{
		std::cerr << "Error. Star input file " << starfile << " is not a .bin or .txt file.\n";
		return false;
	}

	calculate_star_params<T>(nstars, rectangular, corner, theta, stars, kappastar, 
		m_lower, m_upper, mean_mass, mean_mass2, mean_mass2_ln_mass);

	return true;
}

/******************************************************************************
write star field file

\param nstars -- number of point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- corner of the rectangular field of point mass lenses
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param starfile -- location of the star field file

\return bool -- true if file is successfully written, false if not
******************************************************************************/
template <typename T>
bool write_star_file(int nstars, int rectangular, Complex<T> corner, T theta, star<T>* stars, const std::string& starfile)
{
	std::filesystem::path starpath = starfile;

	if (starpath.extension() != ".bin")
	{
		std::cerr << "Error. Star file " << starfile << " is not a .bin file.\n";
		return false;
	}

	std::ofstream outfile;
	outfile.open(starfile, std::ios_base::binary);

	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << starfile << "\n";
		return false;
	}

	outfile.write((char*)(&nstars), sizeof(int));
	outfile.write((char*)(&rectangular), sizeof(int));
	if (rectangular)
	{
		outfile.write((char*)(&corner), sizeof(Complex<T>));
	}
	else
	{
		Complex<T> tmp_corner = Complex<T>(corner.abs(), 0);
		outfile.write((char*)(&tmp_corner), sizeof(Complex<T>));
	}
	outfile.write((char*)(&theta), sizeof(T));
	outfile.write((char*)stars, nstars * sizeof(star<T>));
	outfile.close();

	return true;
}

