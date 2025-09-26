#pragma once

#include <string>


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


public:

	/******************************************************************************
	class initializer is empty
	******************************************************************************/
	Microlensing()
	{

	}

};