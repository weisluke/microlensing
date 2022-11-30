﻿#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <string>
#include <system_error>

#include "complex.cuh"


/*structure to hold position and mass of a star*/
template <typename T>
struct star
{
	Complex<T> position;
	T mass;
};


/**************************************************
generate random star field

\param stars -- pointer to array of stars
\param nstars -- number of stars to generate
\param rad -- radius within which to generate stars
\param mass -- mass for each star
\param seed -- random seed to use. defaults to seed
			   generated based on current time

\return seed -- the random seed used
**************************************************/
template <typename T>
int generate_circular_star_field(star<T>* stars, int nstars, T rad, T mass, int seed = static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count()))
{
	const T PI = 3.1415926535898;

	/*random number generator seeded according to the provided seed*/
	std::mt19937 gen(seed);

	/*uniform distribution to pick real values between 0 and 1*/
	std::uniform_real_distribution<T> dis(0, 1);

	/*variables to hold randomly chosen angle and radius*/
	T a, r;

	for (int i = 0; i < nstars; i++)
	{
		/*random angle*/
		a = dis(gen) * 2.0 * PI;
		/*random radius uses square root of random number
		as numbers need to be evenly dispersed in 2-D space*/
		r = std::sqrt(dis(gen)) * rad;

		/*transform to Cartesian coordinates*/
		stars[i].position = Complex<T>(r * std::cos(a), r * std::sin(a));
		stars[i].mass = mass;
	}

	return seed;
}

/******************************************************************
determines star field parameters from the given starfile

\param nstars -- number of stars
\param m_low -- lower mass cutoff
\param m_up -- upper mass cutoff
\param meanmass -- mean mass
\param meanmass2 -- mean squared mass
\param starfile -- location of the star field file. the file may
				   be either a whitespace delimited .txt file
				   containing valid values for a star's x
				   coordinate, y coordinate, and mass, in that
				   order, on each line, or a .bin file of star
				   structures (as defined in this source code).

\return bool -- true if file successfully read, false if not
******************************************************************/
template <typename T>
bool read_star_params(int& nstars, T& m_low, T& m_up, T& meanmass, T& meanmass2, const std::string& starfile)
{
	/*set parameters that will be modified based on star input file*/
	nstars = 0;
	m_low = std::numeric_limits<T>::max();
	m_up = std::numeric_limits<T>::min();

	/*set local variables to be used based on star input file
	total mass, total mass^2*/
	T mtot = 0;
	T m2tot = 0;

	std::filesystem::path starpath = starfile;

	std::ifstream infile;

	if (starpath.extension() == ".txt")
	{
		infile.open(starfile);

		if (!infile.is_open())
		{
			std::cerr << "Error. Failed to open file " << starfile << "\n";
			return false;
		}

		T x, y, m;

		while (infile >> x >> y >> m)
		{
			nstars++;
			mtot += m;
			m2tot += m * m;
			m_low = std::fmin(m_low, m);
			m_up = std::fmax(m_up, m);
		}
		infile.close();

		if (nstars < 1)
		{
			std::cerr << "Error. No valid star information found in file " << starfile << "\n";
			return false;
		}
		meanmass = mtot / nstars;
		meanmass2 = m2tot / nstars;
	}
	else if (starpath.extension() == ".bin")
	{
		std::error_code err;
		std::uintmax_t fsize = std::filesystem::file_size(starfile, err);

		if (err)
		{
			std::cerr << "Error determining size of star input file " << starfile << "\n";
			return false;
		}

		infile.open(starfile, std::ios_base::binary);

		if (!infile.is_open())
		{
			std::cerr << "Error. Failed to open file " << starfile << "\n";
			return false;
		}

		infile.read((char*)(&nstars), sizeof(int));

		if (nstars < 1)
		{
			std::cerr << "Error. No valid star information found in file " << starfile << "\n";
			return false;
		}

		star<T>* stars = new (std::nothrow) star<T>[nstars];

		if (!stars)
		{
			std::cerr << "Error. Memory allocation for *stars failed.\n";
			return false;
		}

		if ((fsize - sizeof(int)) == nstars * sizeof(star<T>))
		{
			infile.read((char*)stars, nstars * sizeof(star<T>));
		}
		else if ((fsize - sizeof(int)) == nstars * sizeof(star<float>))
		{
			star<float>* temp_stars = new (std::nothrow) star<float>[nstars];
			if (!temp_stars)
			{
				std::cerr << "Error. Memory allocation for *temp_stars failed.\n";
				return false;
			}
			infile.read((char*)temp_stars, nstars * sizeof(star<float>));
			for (int i = 0; i < nstars; i++)
			{
				stars[i].position = Complex<T>(temp_stars[i].position.re, temp_stars[i].position.im);
				stars[i].mass = temp_stars[i].mass;
			}
			delete temp_stars;
			temp_stars = nullptr;
		}
		else if ((fsize - sizeof(int)) == nstars * sizeof(star<double>))
		{
			star<double>* temp_stars = new (std::nothrow) star<double>[nstars];
			if (!temp_stars)
			{
				std::cerr << "Error. Memory allocation for *temp_stars failed.\n";
				return false;
			}
			infile.read((char*)temp_stars, nstars * sizeof(star<double>));
			for (int i = 0; i < nstars; i++)
			{
				stars[i].position = Complex<T>(temp_stars[i].position.re, temp_stars[i].position.im);
				stars[i].mass = temp_stars[i].mass;
			}
			delete temp_stars;
			temp_stars = nullptr;
		}
		else
		{
			std::cerr << "Error. Binary star file does not contain valid single or double precision stars.\n";
			return false;
		}

		infile.close();

		for (int i = 0; i < nstars; i++)
		{
			mtot += stars[i].mass;
			m2tot += stars[i].mass * stars[i].mass;
			m_low = std::fmin(m_low, stars[i].mass);
			m_up = std::fmax(m_up, stars[i].mass);
		}
		meanmass = mtot / nstars;
		meanmass2 = m2tot / nstars;

		delete stars;
		stars = nullptr;
	}
	else
	{
		std::cerr << "Error. Star input file " << starfile << " is not a .bin or .txt file.\n";
		return false;
	}

	return true;
}

/******************************************************************
read star field file

\param stars -- pointer to array of stars
\param nstars -- number of stars
\param starfile -- location of the star field file. the file may
				   be either a whitespace delimited .txt file
				   containing valid values for a star's x
				   coordinate, y coordinate, and mass, in that
				   order, on each line, or a .bin file of star
				   structures (as defined in this source code).

\return bool -- true if file successfully read, false if not
******************************************************************/
template <typename T>
bool read_star_file(star<T>* stars, int nstars, const std::string& starfile)
{
	std::filesystem::path starpath = starfile;

	std::ifstream infile;

	if (starpath.extension() == ".txt")
	{
		infile.open(starfile);

		if (!infile.is_open())
		{
			std::cerr << "Error. Failed to open file " << starfile << "\n";
			return false;
		}

		T x, y, m;

		for (int i = 0; i < nstars; i++)
		{
			infile >> x >> y >> m;
			stars[i].position = Complex<T>(x, y);
			stars[i].mass = m;
		}
		infile.close();
	}
	else if (starpath.extension() == ".bin")
	{
		std::error_code err;
		std::uintmax_t fsize = std::filesystem::file_size(starfile, err);

		if (err)
		{
			std::cerr << "Error determining size of star input file " << starfile << "\n";
			return false;
		}

		infile.open(starfile, std::ios_base::binary);

		if (!infile.is_open())
		{
			std::cerr << "Error. Failed to open file " << starfile << "\n";
			return false;
		}

		int temp_nstars;
		infile.read((char*)(&temp_nstars), sizeof(int));

		if ((fsize - sizeof(int)) == nstars * sizeof(star<T>))
		{
			infile.read((char*)stars, nstars * sizeof(star<T>));
		}
		else if ((fsize - sizeof(int)) == nstars * sizeof(star<float>))
		{
			star<float>* temp_stars = new (std::nothrow) star<float>[nstars];
			if (!temp_stars)
			{
				std::cerr << "Error. Memory allocation for *temp_stars failed.\n";
				return false;
			}
			infile.read((char*)temp_stars, nstars * sizeof(star<float>));
			for (int i = 0; i < nstars; i++)
			{
				stars[i].position = Complex<T>(temp_stars[i].position.re, temp_stars[i].position.im);
				stars[i].mass = temp_stars[i].mass;
			}
			delete temp_stars;
			temp_stars = nullptr;
		}
		else if ((fsize - sizeof(int)) == nstars * sizeof(star<double>))
		{
			star<double>* temp_stars = new (std::nothrow) star<double>[nstars];
			if (!temp_stars)
			{
				std::cerr << "Error. Memory allocation for *temp_stars failed.\n";
				return false;
			}
			infile.read((char*)temp_stars, nstars * sizeof(star<double>));
			for (int i = 0; i < nstars; i++)
			{
				stars[i].position = Complex<T>(temp_stars[i].position.re, temp_stars[i].position.im);
				stars[i].mass = temp_stars[i].mass;
			}
			delete temp_stars;
			temp_stars = nullptr;
		}
		else
		{
			std::cerr << "Error. Binary star file does not contain valid single or double precision stars.\n";
			return false;
		}

		infile.close();
	}
	else
	{
		std::cerr << "Error. Star input file " << starfile << " is not a .bin or .txt file.\n";
		return false;
	}

	return true;
}

/******************************************************************
write star field file

\param stars -- pointer to array of stars
\param nstars -- number of stars
\param starfile -- location of the star field file. the file may
				   be either a whitespace delimited .txt file which
				   will contain valid values for a star's x
				   coordinate, y coordinate, and mass, in that
				   order, on each line, or a .bin file of star
				   structures (as defined in this source code).

\return bool -- true if file successfully written, false if not
******************************************************************/
template <typename T>
bool write_star_file(star<T>* stars, int nstars, const std::string& starfile)
{
	std::filesystem::path starpath = starfile;

	std::ofstream outfile;

	if (starpath.extension() == ".txt")
	{
		outfile.precision(9);
		outfile.open(starfile);

		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << starfile << "\n";
			return false;
		}

		for (int i = 0; i < nstars; i++)
		{
			outfile << stars[i].position.re << " " << stars[i].position.im << " " << stars[i].mass << "\n";
		}
		outfile.close();
	}
	else if (starpath.extension() == ".bin")
	{
		outfile.open(starfile, std::ios_base::binary);

		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << starfile << "\n";
			return false;
		}

		outfile.write((char*)(&nstars), sizeof(int));
		outfile.write((char*)stars, nstars * sizeof(star<T>));
		outfile.close();
	}
	else
	{
		std::cerr << "Error. Star file " << starfile << " is not a .bin or .txt file.\n";
		return false;
	}

	return true;
}
