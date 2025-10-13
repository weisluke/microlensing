#pragma once

#include "array_functions.cuh"
#include "complex.cuh"
#include "mif_functions.cuh"
#include "microlensing.cuh"
#include "star.cuh"
#include "stopwatch.hpp"
#include "tree_node.cuh"
#include "util/math_util.cuh"
#include "util/util.cuh"

#include <thrust/universal_vector.h> //for thrust::universal_vector

#include <algorithm> //for std::min
#include <cmath>
#include <fstream> //for writing files
#include <iostream>
#include <limits> //for std::numeric_limits
#include <numbers>
#include <numeric> //for std::reduce
#include <string>
#include <vector>


template <typename T>
class MIF : public Microlensing<T>
{

public:
	/******************************************************************************
	default input variables
	******************************************************************************/
	//must include microlensing variables to put them into scope since class is a template
	using Microlensing<T>::kappa_tot;
	using Microlensing<T>::shear;
	using Microlensing<T>::kappa_star;
	using Microlensing<T>::theta_star;
	using Microlensing<T>::mass_function_str;
	using Microlensing<T>::m_solar;
	using Microlensing<T>::m_lower;
	using Microlensing<T>::m_upper;
	using Microlensing<T>::rectangular;
	using Microlensing<T>::approx;
	using Microlensing<T>::safety_scale;
	using Microlensing<T>::starfile;
	using Microlensing<T>::random_seed;
	using Microlensing<T>::write_stars;
	using Microlensing<T>::outfile_prefix;

	T light_loss = static_cast<T>(0.001); //average fraction of light lost due to scatter by the microlenses in the large deflection angle limit
	Complex<T> w0 = Complex<T>();
	Complex<T> v = Complex<T>(2, 3);
	int write_images = 1;
	int write_image_lines = 0;
	int write_magnifications = 1;


private:
	//must include microlensing variables to put them into scope since class is a template
	using Microlensing<T>::num_stars;
	using Microlensing<T>::outfile_type;
	using Microlensing<T>::MAX_TAYLOR_SMOOTH;
	using Microlensing<T>::threads;
	using Microlensing<T>::blocks;
	using Microlensing<T>::mean_mass;
	using Microlensing<T>::mean_mass2;
	using Microlensing<T>::mean_mass2_ln_mass;
	using Microlensing<T>::kappa_star_actual;
	using Microlensing<T>::m_lower_actual;
	using Microlensing<T>::m_upper_actual;
	using Microlensing<T>::mean_mass_actual;
	using Microlensing<T>::mean_mass2_actual;
	using Microlensing<T>::mean_mass2_ln_mass_actual;
	using Microlensing<T>::mu_ave;
	using Microlensing<T>::corner;
	using Microlensing<T>::taylor_smooth;
	using Microlensing<T>::alpha_error;
	using Microlensing<T>::expansion_order;
	using Microlensing<T>::root_half_length;
	using Microlensing<T>::tree_levels;
	using Microlensing<T>::tree;
	using Microlensing<T>::stars;

	/******************************************************************************
	stopwatch for timing purposes
	******************************************************************************/
	Stopwatch stopwatch;
	double t_elapsed;

	/******************************************************************************
	maximum source plane size of the region of images visible for a macro-image
	which on average loses no more than the desired amount of flux
	******************************************************************************/
	T max_r;

	/******************************************************************************
	dynamic memory
	******************************************************************************/
	Complex<T>* image_lines = nullptr;
	Complex<T>* source_lines = nullptr;
	T* image_lines_mags = nullptr;
	std::vector<int> image_lines_lengths;

	std::vector<Complex<T>> images;
	std::vector<Complex<T>> images_mags;



	//optional return or not, so memory can be cleared in destructor without error checking
	bool clear_memory(int verbose, bool return_on_error = true)
	{
		print_verbose("Clearing MIF<T> memory...\n", verbose, 3);
		
		/******************************************************************************
		free memory and set variables to nullptr
		******************************************************************************/

		if (!Microlensing<T>::clear_memory(verbose, return_on_error)) return false;
		
		cudaFree(image_lines);
		if (return_on_error && cuda_error("cudaFree(*image_lines)", false, __FILE__, __LINE__)) return false;
		image_lines = nullptr;
		
		cudaFree(source_lines);
		if (return_on_error && cuda_error("cudaFree(*source_lines)", false, __FILE__, __LINE__)) return false;
		source_lines = nullptr;
		
		cudaFree(image_lines_mags);
		if (return_on_error && cuda_error("cudaFree(*image_lines_mags)", false, __FILE__, __LINE__)) return false;
		image_lines_mags = nullptr;

		image_lines_lengths.clear();
		image_lines_lengths.shrink_to_fit();

		images.clear();
		images.shrink_to_fit();

		images_mags.clear();
		images_mags.shrink_to_fit();

		print_verbose("Done clearing MIF<T> memory.\n", verbose, 3);
		return true;
	}

	bool check_input_params(int verbose)
	{
		print_verbose("Checking MIF<T> input parameters...\n", verbose, 3);

		if (!Microlensing<T>::check_input_params(verbose)) return false;

		if (light_loss < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. light_loss must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}
		else if (light_loss > 0.01)
		{
			std::cerr << "Error. light_loss must be <= 0.01\n";
			return false;
		}

		if (write_images != 0 && write_images != 1)
		{
			std::cerr << "Error. write_images must be 1 (true) or 0 (false).\n";
			return false;
		}

		if (write_image_lines != 0 && write_image_lines != 1)
		{
			std::cerr << "Error. write_image_lines must be 1 (true) or 0 (false).\n";
			return false;
		}

		if (write_magnifications != 0 && write_magnifications != 1)
		{
			std::cerr << "Error. write_magnifications must be 1 (true) or 0 (false).\n";
			return false;
		}

		print_verbose("Done checking MIF<T> input parameters.\n", verbose, 3);
		return true;
	}

	bool calculate_derived_params(int verbose)
	{
		print_verbose("Calculating MIF<T> derived parameters...\n", verbose, 3);
		stopwatch.start();

		if (!Microlensing<T>::calculate_derived_params(verbose)) return false;

		set_param("max_r", max_r, theta_star * std::sqrt(kappa_star * mean_mass2 / (mean_mass * light_loss)), verbose);

		/******************************************************************************
		if stars are not drawn from external file, calculate final number of stars to
		use and corner of the star field
		******************************************************************************/
		if (starfile == "")
		{
			corner = Complex<T>((std::abs(w0.re) + max_r) / std::abs(1 - kappa_tot + shear), 
								(std::abs(w0.im) + max_r) / std::abs(1 - kappa_tot - shear));

			if (rectangular)
			{
				num_stars = std::ceil(
					safety_scale * 2 * corner.re
					* safety_scale * 2 * corner.im
					* kappa_star / (std::numbers::pi_v<T> * theta_star * theta_star * mean_mass)
					);
				set_param("num_stars", num_stars, num_stars, verbose);

				corner = Complex<T>(std::sqrt(corner.re / corner.im), std::sqrt(corner.im / corner.re));
				corner *= std::sqrt(std::numbers::pi_v<T> * theta_star * theta_star * num_stars * mean_mass / (4 * kappa_star));
				set_param("corner", corner, corner, verbose);
			}
			else
			{
				num_stars = std::ceil(
					safety_scale * corner.abs()
					* safety_scale * corner.abs()
					* kappa_star / (theta_star * theta_star * mean_mass)
					);
				set_param("num_stars", num_stars, num_stars, verbose);

				corner = corner / corner.abs();
				corner *= std::sqrt(theta_star * theta_star * num_stars * mean_mass / kappa_star);
				set_param("corner", corner, corner, verbose);
			}
		}
		/******************************************************************************
		otherwise, check that the star file actually has a large enough field of stars
		******************************************************************************/
		else
		{
			Complex<T> tmp_corner = Complex<T>((std::abs(w0.re) + max_r) / std::abs(1 - kappa_tot + shear), 
											   (std::abs(w0.im) + max_r) / std::abs(1 - kappa_tot - shear));

			if (!rectangular)
			{
				corner = tmp_corner / tmp_corner.abs() * corner.abs();
				set_param("corner", corner, corner, verbose);
			}

			if (
				(rectangular && (corner.re < safety_scale * tmp_corner.re || corner.im < safety_scale * tmp_corner.im)) ||
				(!rectangular && (corner.abs() < safety_scale * tmp_corner.abs()))
				)
			{
				std::cerr << "Error. The provided star field is not large enough to cover the necessary image plane region.\n";
				std::cerr << "Try decreasing the safety_scale, or providing a larger field of stars.\n";
				return false;
			}
		}

		//error is 10^-7 einstein radii
		set_param("alpha_error", alpha_error, theta_star * 0.0000001, verbose);

		taylor_smooth = 1;
		while ((kappa_star * std::numbers::inv_pi_v<T> * 4 / (taylor_smooth + 1) * corner.abs() * (safety_scale + 1) / (safety_scale - 1)
				* std::pow(1 / safety_scale, taylor_smooth + 1) > alpha_error)
				&& taylor_smooth <= MAX_TAYLOR_SMOOTH)
		{
			taylor_smooth += 2;
		}
		/******************************************************************************
		if phase * (taylor_smooth - 1), a term in the approximation of alpha_smooth, is
		not in the correct fractional range of pi, increase taylor_smooth
		this is due to NOT wanting cos(phase * (taylor_smooth - 1)) = 0, within errors
		******************************************************************************/
		while ((std::fmod(corner.arg() * (taylor_smooth - 1), std::numbers::pi_v<T>) < 0.1 * std::numbers::pi_v<T> 
				|| std::fmod(corner.arg() * (taylor_smooth - 1), std::numbers::pi_v<T>) > 0.9 * std::numbers::pi_v<T>)
				&& taylor_smooth <= MAX_TAYLOR_SMOOTH)
		{
			taylor_smooth += 2;
		}
		set_param("taylor_smooth", taylor_smooth, taylor_smooth, verbose);
		if (rectangular && taylor_smooth > MAX_TAYLOR_SMOOTH)
		{
			std::cerr << "Error. taylor_smooth must be <= " << MAX_TAYLOR_SMOOTH << "\n";
			return false;
		}

		t_elapsed = stopwatch.stop();
		print_verbose("Done calculating derived parameters. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 3);

		return true;
	}

	bool find_image_line(int verbose)
	{
		print_verbose("Finding images...\n", verbose, 1);

        std::vector<std::vector<Complex<T>>> tmp_image_lines;

        int s = 30; //scale factor for how far true root can be from the tangent estimate
        Complex<T> z, new_z, tmp_dz, dz;
        T max_dz = theta_star * std::sqrt(mean_mass2 / mean_mass) / s;
        T min_dz = max_dz / 1000;
        TreeNode<T>* node;
        T mu1, mu2;
        T dt = 1;

        thrust::universal_vector<int> use_star(num_stars, 1);
        set_threads(threads, 256);
        set_blocks(threads, blocks, num_stars);
		if (write_image_lines)
		{
        	use_star_kernel<T> <<<blocks, threads>>> (stars, num_stars, kappa_tot, shear, w0, v, max_r, &use_star[0]);
        	if (cuda_error("use_star_kernel", true, __FILE__, __LINE__)) return false;
		}
		else
		{
			int N_ANGLES = 1000;
			for (int i = 0; i <= N_ANGLES; i++)
			{
        		use_star_kernel<T> <<<blocks, threads>>> (stars, num_stars, kappa_tot, shear, w0, 
														  v * Complex<T>(std::cos(std::numbers::pi_v<T> / (2 * N_ANGLES) * i),
																		 std::sin(std::numbers::pi_v<T> / (2 * N_ANGLES) * i)),
														  max_r, &use_star[0]);
        		if (cuda_error("use_star_kernel", true, __FILE__, __LINE__)) return false;
			}
		}
        
		/******************************************************************************
		BEGIN finding main image line
		******************************************************************************/
		print_verbose("Finding main image line...\n", verbose, 2);
        stopwatch.start();

        //start in a position such that the velocity vector takes you through the field
        z = Complex<T>(-v.re / (1 - kappa_tot + shear), 
            		   -v.im / (1 - kappa_tot - shear));
        z = z / z.abs() * corner.abs() * safety_scale;

        tmp_image_lines.push_back(std::vector<Complex<T>>());

        z = find_root<T>(z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
						 rectangular, corner, approx, taylor_smooth, w0, v);
        tmp_image_lines.back().push_back(z);

        node = treenode::get_nearest_node(z, tree[0]);
        mu1 = microlensing::mu<T>(z, kappa_tot, shear, theta_star, stars, kappa_star, node,
                        		  rectangular, corner, approx, taylor_smooth);
        do
        {
            new_z = step_tangent<T>(z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                        rectangular, corner, approx, taylor_smooth, w0, v, dt, min_dz, max_dz);
            tmp_dz = (new_z - z);
            new_z = find_root<T>(new_z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                        rectangular, corner, approx, taylor_smooth, w0, v);
            dz = (new_z - z);

            //if actual root is too large relative to the initial tangent step
            if ((dz - tmp_dz).abs() * s > tmp_dz.abs())
            {
                dt /= 2;
                continue;
            }
            //if actual root is very close to the tangent step
            else if ((dz - tmp_dz).abs() * s * s < tmp_dz.abs())
            {
                dt *= 2;
            }

            node = treenode::get_nearest_node(new_z, tree[0]);
            mu2 = microlensing::mu<T>(new_z, kappa_tot, shear, theta_star, stars, kappa_star, node,
                            		  rectangular, corner, approx, taylor_smooth);
            
            if ((mu1 < 0 && mu2 > 0 )
                || (mu1 > 0 && mu2 < 0))
            {
                tmp_dz = dz;
                new_z = z - mu1 / (mu2 - mu1) * dz;
                new_z = find_critical_curve_image<T>(new_z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                            rectangular, corner, approx, taylor_smooth, w0, v);
                dz = (new_z - z);
                dt *= dz.abs() / tmp_dz.abs();
                mu2 = 0;
            }           
            z = new_z;
            mu1 = mu2;
            tmp_image_lines.back().push_back(z);

            if (is_near_star(z, stars, tree[0], dz)
                && is_near_star(z - dz, stars, tree[0], dz))
            {
                int index = get_nearest_star(z - dz / 2, stars, tree[0]);
                use_star[index] = 0;
            }
        } while (z.abs() < safety_scale * corner.abs());
        t_elapsed = stopwatch.stop();
        print_verbose("Done finding main image line. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 2);
		/******************************************************************************
		END finding main image line
		******************************************************************************/

        
		/******************************************************************************
		BEGIN finding secondary image loops
		******************************************************************************/
		print_verbose("Finding secondary image loops...\n", verbose, 2);
        stopwatch.start();
        for (int i = 0; i < num_stars; i++)
        {
            if (use_star[i])
            {
                use_star[i] = 0;
                dz = Complex<T>(-v.re / (1 - kappa_tot + shear), 
                                -v.im / (1 - kappa_tot - shear));
                max_dz = theta_star * theta_star * stars[i].mass / std::abs(macro_parametric_image_line(stars[i].position, kappa_tot, shear, w0, v));
                max_dz /= 2 * s; //diameter to radius
                max_dz = std::min(max_dz, theta_star * std::sqrt(mean_mass2 / mean_mass) / s);
                min_dz = max_dz / 1000;
                dz *= min_dz / dz.abs();
                z = stars[i].position + dz;

                tmp_image_lines.push_back(std::vector<Complex<T>>());

                z = find_root<T>(z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                                    rectangular, corner, approx, taylor_smooth, w0, v);
                tmp_image_lines.back().push_back(z);

                node = treenode::get_nearest_node(z, tree[0]);
                mu1 = microlensing::mu<T>(z, kappa_tot, shear, theta_star, stars, kappa_star, node,
                                		  rectangular, corner, approx, taylor_smooth);
                do
                {
                    new_z = step_tangent<T>(z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                                			rectangular, corner, approx, taylor_smooth, w0, v, dt, min_dz, max_dz);
                    tmp_dz = (new_z - z);

                    if (is_near_star(new_z, stars, tree[0], tmp_dz)
                        && is_near_star(new_z - tmp_dz, stars, tree[0], tmp_dz))
                    {
                        int index = get_nearest_star(new_z - tmp_dz / 2, stars, tree[0]);
                        use_star[index] = 0;
                        if (index == i)
                        {
                            break;
                        }
                    }

                    new_z = find_root<T>(new_z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                                		 rectangular, corner, approx, taylor_smooth, w0, v);
                    dz = (new_z - z);
                    if ((dz - tmp_dz).abs() * s > tmp_dz.abs())
                    {
                        dt /= 2;
                        continue;
                    }
                    else if ((dz - tmp_dz).abs() * s * s < tmp_dz.abs())
                    {
                        dt *= 2;
                    }

                    node = treenode::get_nearest_node(new_z, tree[0]);
                    mu2 = microlensing::mu<T>(new_z, kappa_tot, shear, theta_star, stars, kappa_star, node,
                                    		  rectangular, corner, approx, taylor_smooth);
                    
                    if ((mu1 < 0 && mu2 > 0 )
                        || (mu1 > 0 && mu2 < 0))
                    {
                        tmp_dz = dz;
                        new_z = z - mu1 / (mu2 - mu1) * dz;
                        new_z = find_critical_curve_image<T>(new_z, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
                                    rectangular, corner, approx, taylor_smooth, w0, v);
                        dz = (new_z - z);
                        dt *= dz.abs() / tmp_dz.abs();
                        mu2 = 0;
                    }           
                    z = new_z;
                    mu1 = mu2;
                    tmp_image_lines.back().push_back(z);
                } while (true);
            }
        }
        t_elapsed = stopwatch.stop();
        print_verbose("Done finding secondary image loops. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 2);
		/******************************************************************************
		END finding secondary image loops
		******************************************************************************/

        print_verbose("Done finding images.\n", verbose, 1);


        for (int i = 0; i < tmp_image_lines.size(); i++)
        {
            image_lines_lengths.push_back(tmp_image_lines[i].size());
        }

        int total_image_lines_lengths = std::reduce(image_lines_lengths.begin(), image_lines_lengths.end(), 0);
        
        cudaMallocManaged(&image_lines, total_image_lines_lengths * sizeof(Complex<T>));
        if (cuda_error("cudaMallocManaged(*image_lines)", false, __FILE__, __LINE__)) return false;
        cudaMallocManaged(&source_lines, total_image_lines_lengths * sizeof(Complex<T>));
        if (cuda_error("cudaMallocManaged(*source_lines)", false, __FILE__, __LINE__)) return false;
        cudaMallocManaged(&image_lines_mags, total_image_lines_lengths * sizeof(T));
        if (cuda_error("cudaMallocManaged(*image_lines_mags)", false, __FILE__, __LINE__)) return false;

        print_verbose("Copying image lines...\n", verbose, 2);
        stopwatch.start();
        for (int i = 0; i < tmp_image_lines.size(); i++)
        {
            int start = std::reduce(&image_lines_lengths[0], &image_lines_lengths[i], 0);
            thrust::copy(tmp_image_lines[i].begin(), tmp_image_lines[i].end(), &image_lines[start]);
        }
        t_elapsed = stopwatch.stop();
        print_verbose("Done copying image lines. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 2);


        set_threads(threads, 256);
        set_blocks(threads, blocks, total_image_lines_lengths);

        print_verbose("Mapping image lines...\n", verbose, 2);
        stopwatch.start();
        image_to_source_kernel<T> <<<blocks, threads>>> (image_lines, total_image_lines_lengths, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
            rectangular, corner, approx, taylor_smooth, source_lines);
        if (cuda_error("image_to_source_kernel", true, __FILE__, __LINE__)) return false;
        t_elapsed = stopwatch.stop();
        print_verbose("Done mapping image lines. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 2);

        print_verbose("Calculating magnifications...\n", verbose, 2);
        stopwatch.start();
        magnifications_kernel<T> <<<blocks, threads>>> (image_lines, total_image_lines_lengths, kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
            rectangular, corner, approx, taylor_smooth, image_lines_mags);
        if (cuda_error("magnifications_kernel", true, __FILE__, __LINE__)) return false;
        t_elapsed = stopwatch.stop();
        print_verbose("Done calculating magnifications. Elapsed time: " << t_elapsed << " seconds.\n", verbose, 2);
		
		return true;
	}

	bool find_point_images(int verbose)
	{
		print_verbose("Finding point images...\n", verbose, 1);

		Complex<T> z1, z2, dz, dwdz, dwdzbar;
		T f1, f2;
		TreeNode<T>* node;
		
        stopwatch.start();
		for (int i =0; i < image_lines_lengths.size(); i++)
		{
			int start = std::reduce(&image_lines_lengths[0], &image_lines_lengths[i], 0);

			for (int j = 0; j < image_lines_lengths[i] - 1; j++)
			{
				z1 = image_lines[start + j];
				z2 = image_lines[start + j + 1];
				dz = (z2 - z1);

				node = treenode::get_nearest_node(z2, tree[0]);
				f2 = parametric_image_line(z2, kappa_tot, shear, theta_star, stars, kappa_star, node,
										   rectangular, corner, approx, taylor_smooth, w0, v * Complex<T>(0, 1));
				node = treenode::get_nearest_node(z1, tree[0]);
				f1 = parametric_image_line(z1, kappa_tot, shear, theta_star, stars, kappa_star, node,
										   rectangular, corner, approx, taylor_smooth, w0, v * Complex<T>(0, 1));
				
				//if we are crossing a star, the sign changes
				bool is_star = (is_near_star(z1, stars, tree[0], dz) && is_near_star(z2, stars, tree[0], dz));
				
				if (sgn(f1) != sgn(f2) && !is_star)
				{
					T dt = -f1 / (f2 - f1);
					images.push_back(z1 + dz * dt);
				}

			}
		}
		print_verbose("Number of point images: " << images.size() << "\n", verbose, 2);
		for (int i = 0; i < images.size(); i++)
		{
			images[i] = find_point_image(images[i], kappa_tot, shear, theta_star, stars, kappa_star, tree[0],
					rectangular, corner, approx, taylor_smooth, w0, v);
			node = treenode::get_nearest_node(images[i], tree[0]);
			dwdz = microlensing::d_w_d_z<T>(images[i], kappa_tot, shear, kappa_star,
					rectangular, corner, approx);
			dwdzbar = microlensing::d_w_d_zbar<T>(images[i], kappa_tot, shear, theta_star, stars, kappa_star, node,
					rectangular, corner, approx, taylor_smooth);
			images_mags.push_back(dwdz);
			images_mags.push_back(dwdzbar);
		}
        t_elapsed = stopwatch.stop();
		print_verbose("Done finding point images.\n", verbose, 1);
		
		return true;
	}

	bool write_files(int verbose)
	{
		/******************************************************************************
		stream for writing output files
		set precision to 9 digits
		******************************************************************************/
		std::ofstream outfile;
		outfile.precision(9);
		std::string fname;


		if (!Microlensing<T>::write_files(verbose, "mif")) return false;

		print_verbose("Writing MIF<T> parameter info...\n", verbose, 2);
		fname = outfile_prefix + "mif_parameter_info.txt";
		outfile.open(fname, std::ios_base::app);
		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		outfile << "light_loss " << light_loss << "\n";
		outfile << "w0_1 " << w0.re << "\n";
		outfile << "w0_2 " << w0.im << "\n";
		outfile << "v_1 " << v.re << "\n";
		outfile << "v_2 " << v.im << "\n";
		outfile.close();
		print_verbose("Done writing MIF<T> parameter info to file " << fname << "\n", verbose, 1);


		if (write_images)
		{
			print_verbose("Writing point images...\n", verbose, 2);
			fname = outfile_prefix + "mif_images" + outfile_type;
			if (!write_array<Complex<T>>(&images[0], 1, images.size(), fname))
			{
				std::cerr << "Error. Unable to write point images to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing point images to file " << fname << "\n", verbose, 1);

			print_verbose("Writing point image magnifications...\n", verbose, 2);
			fname = outfile_prefix + "mif_images_magnifications" + outfile_type;
			if (!write_array<Complex<T>>(&images_mags[0], images.size(), 2, fname))
			{
				std::cerr << "Error. Unable to write point image magnifications to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing point image magnifications to file " << fname << "\n", verbose, 1);
		}

		if (write_image_lines)
		{
			print_verbose("Writing image lines...\n", verbose, 2);
			fname = outfile_prefix + "mif_image_lines" + outfile_type;
			if (!write_ragged_array<Complex<T>>(image_lines, image_lines_lengths, fname))
			{
				std::cerr << "Error. Unable to write image lines to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing image lines to file " << fname << "\n", verbose, 1);

			print_verbose("Writing source lines...\n", verbose, 2);
			fname = outfile_prefix + "mif_source_lines" + outfile_type;
			if (!write_ragged_array<Complex<T>>(source_lines, image_lines_lengths, fname))
			{
				std::cerr << "Error. Unable to write source lines to file " << fname << "\n";
				return false;
			}
			print_verbose("Done writing source lines to file " << fname << "\n", verbose, 1);

			if (write_magnifications)
			{
				print_verbose("Writing image lines magnifications...\n", verbose, 2);
				fname = outfile_prefix + "mif_image_lines_magnifications" + outfile_type;
				if (!write_ragged_array<T>(image_lines_mags, image_lines_lengths, fname))
				{
					std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
					return false;
				}
				print_verbose("Done writing image lines magnifications to file " << fname << "\n", verbose, 1);
			}
		}

		return true;
	}


public:

	/******************************************************************************
	class initializer is empty
	******************************************************************************/
	MIF()
	{

	}

	/******************************************************************************
	class destructor clears memory with no output or error checking
	******************************************************************************/
	~MIF()
	{
		clear_memory(0, false);
	}

	/******************************************************************************
	copy constructor sets this object's dynamic memory pointers to null
	******************************************************************************/
	MIF(const MIF& other) : Microlensing<T>(other)
	{
		image_lines = nullptr;
		source_lines = nullptr;
		image_lines_mags = nullptr;
	}

	/******************************************************************************
	copy assignment sets this object's dynamic memory pointers to null
	******************************************************************************/
	MIF& operator=(const MIF& other)
	{
        if (this == &other) return *this;

		Microlensing<T>::operator=(other);

		image_lines = nullptr;
		source_lines = nullptr;
		image_lines_mags = nullptr;

		return *this;
	}

	bool run(int verbose)
	{
		if (!Microlensing<T>::set_cuda_devices(verbose)) return false;
		if (!clear_memory(verbose)) return false;
		if (!check_input_params(verbose)) return false;
		if (!calculate_derived_params(verbose)) return false;
		if (!Microlensing<T>::allocate_initialize_memory(verbose)) return false;
		if (!Microlensing<T>::populate_star_array(verbose)) return false;
		if (!Microlensing<T>::create_tree(verbose)) return false;
		if (!find_image_line(verbose)) return false;
		if (!find_point_images(verbose)) return false;

		return true;
	}

	bool save(int verbose)
	{
		if (!write_files(verbose)) return false;

		return true;
	}

	Complex<T>* get_images()				{return &images[0];}
	int get_num_images()					{return images.size();}
	Complex<T>* get_images_mags()			{return &images_mags[0];}
	Complex<T>* get_image_lines()			{return image_lines;}
	int get_num_image_lines()				{return image_lines_lengths.size();}
	int* get_image_lines_lengths()			{return &image_lines_lengths[0];}
	int get_total_image_lines_length()		{return std::reduce(image_lines_lengths.begin(), image_lines_lengths.end(), 0);}
	Complex<T>* get_source_lines()			{return source_lines;}
	T* get_image_lines_mags()				{return image_lines_mags;}

};

