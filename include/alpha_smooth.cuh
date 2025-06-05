#pragma once

#include "complex.cuh"

#include <numbers>


/******************************************************************************
Heaviside Step Function

\param x -- number to evaluate

\return 1 if x >= 0, 0 if x < 0
******************************************************************************/
template <typename T>
__host__ __device__ T heaviside(T x)
{
	if (x >= 0)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/******************************************************************************
2-Dimensional Boxcar Function

\param z -- complex number to evalulate
\param corner -- corner of the rectangular region

\return 1 if z lies within or on the border of the rectangle defined by corner,
		0 if it is outside
******************************************************************************/
template <typename T>
__host__ __device__ T boxcar(Complex<T> z, Complex<T> corner)
{
	if (-corner.re <= z.re && z.re <= corner.re && -corner.im <= z.im && z.im <= corner.im)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/******************************************************************************
calculate the deflection angle due to smooth matter

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- corner of the rectangular field of point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth

\return alpha_smooth
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> alpha_smooth(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	Complex<T> a_smooth;

	if (rectangular)
	{
		if (approx)
		{
			Complex<T> s1;
			Complex<T> s2;
			Complex<T> s3;
			Complex<T> s4;

			for (int i = taylor_smooth; i >= 1; i--)
			{
				s1 += Complex<T>(1, 0) / i;
				s2 += Complex<T>(1, 0) / i;
				s3 += Complex<T>(1, 0) / i;
				s4 += Complex<T>(1, 0) / i;

				s1 *= z.conj() / corner;
				s2 *= z.conj() / corner.conj();
				s3 *= z.conj() / -corner;
				s4 *= z.conj() / -corner.conj();
			}

			a_smooth = (corner - z.conj()) * (corner.log() - s1) - (corner.conj() - z.conj()) * (corner.conj().log() - s2)
				+ (-corner - z.conj()) * ((-corner).log() - s3) - (-corner.conj() - z.conj()) * ((-corner).conj().log() - s4);
			a_smooth *= Complex<T>(0, -kappastar * std::numbers::inv_pi_v<T>);
			a_smooth -= kappastar * 2 * (corner.re + z.re);
		}
		else
		{
			Complex<T> c1 = corner - z.conj();
			Complex<T> c2 = corner.conj() - z.conj();
			Complex<T> c3 = -corner - z.conj();
			Complex<T> c4 = -corner.conj() - z.conj();

			a_smooth = c1 * c1.log() - c2 * c2.log() + c3 * c3.log() - c4 * c4.log();
			a_smooth *= Complex<T>(0, -kappastar * std::numbers::inv_pi_v<T>);
			a_smooth -= kappastar * 2 * (corner.re + z.re) * boxcar(z, corner);
			a_smooth -= kappastar * 4 * corner.re * heaviside(corner.im + z.im) * heaviside(corner.im - z.im) * heaviside(z.re - corner.re);
		}
	}
	else
	{
		if (approx)
		{
			a_smooth = -kappastar * z;
		}
		else
		{
			if (z.abs() <= corner.abs())
			{
				a_smooth = -kappastar * z;
			}
			else
			{
				a_smooth = -kappastar * corner.abs() * corner.abs() / z.conj();
			}
		}
	}

	return a_smooth;
}

/******************************************************************************
calculate the derivative of the deflection angle with respect to z due to
smooth matter

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- corner of the rectangular field of point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not

\return d_alpha_smooth_d_z
******************************************************************************/
template <typename T>
__host__ __device__ T d_alpha_smooth_d_z(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx)
{
	T d_a_smooth_d_z;

	if (rectangular)
	{
		if (approx)
		{
			d_a_smooth_d_z = -kappastar;
		}
		else
		{
			d_a_smooth_d_z = -kappastar * boxcar(z, corner);
		}
	}
	else
	{
		if (approx)
		{
			d_a_smooth_d_z = -kappastar;
		}
		else
		{
			d_a_smooth_d_z = -kappastar * heaviside(corner.abs() - z.abs());
		}
	}

	return d_a_smooth_d_z;
}

/******************************************************************************
calculate the derivative of the deflection angle with respect to zbar due to
smooth matter

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- corner of the rectangular field of point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth

\return d_alpha_smooth_d_zbar
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> d_alpha_smooth_d_zbar(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	Complex<T> d_a_smooth_d_zbar;

	if (rectangular)
	{
		if (approx)
		{
			Complex<T> r = z.conj() / corner; //geometric series ratio
			Complex<T> phase = Complex<T>(0, corner.arg());

			/******************************************************************************
			we enforce elsewhere that taylor_smooth is odd, and the highest order term is
			not 0
			******************************************************************************/
			for (int i = taylor_smooth - 1; i >= 2; i -= 2)
			{
				d_a_smooth_d_zbar += (1 - (2 * phase * i).exp()) / i;
				d_a_smooth_d_zbar *= r * r;
			}
			d_a_smooth_d_zbar *= 2;

			d_a_smooth_d_zbar *= Complex<T>(0, -kappastar * std::numbers::inv_pi_v<T>);
			d_a_smooth_d_zbar += kappastar - 4 * kappastar * corner.arg() * std::numbers::inv_pi_v<T>;
		}
		else
		{
			Complex<T> c1 = corner.conj() - z.conj();
			Complex<T> c2 = corner - z.conj();
			Complex<T> c3 = -corner - z.conj();
			Complex<T> c4 = -corner.conj() - z.conj();

			d_a_smooth_d_zbar = c1.log() - c2.log() - c3.log() + c4.log();
			d_a_smooth_d_zbar *= Complex<T>(0, -kappastar * std::numbers::inv_pi_v<T>);
			d_a_smooth_d_zbar -= kappastar * boxcar(z, corner);
		}
	}
	else
	{
		if (approx)
		{
			d_a_smooth_d_zbar = 0;
		}
		else
		{
			d_a_smooth_d_zbar = kappastar * corner.abs() * corner.abs() / z.conj().pow(2);
		}
	}

	return d_a_smooth_d_zbar;
}

/******************************************************************************
calculate the second derivative of the deflection angle with respect to zbar^2
due to smooth matter

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- corner of the rectangular field of point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth

\return d2_alpha_smooth_d_zbar2
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> d2_alpha_smooth_d_zbar2(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	Complex<T> d2_a_smooth_d_zbar2;

	if (rectangular)
	{
		if (approx)
		{
			Complex<T> r = z.conj() / corner; //geometric series ratio
			Complex<T> phase = Complex<T>(0, corner.arg());

			/******************************************************************************
			we enforce elsewhere that taylor_smooth is odd, and the highest order term is
			not 0
			******************************************************************************/
			for (int i = taylor_smooth - 1; i >= 2; i -= 2)
			{
				d2_a_smooth_d_zbar2 += 1 - (2 * phase * i).exp();
				d2_a_smooth_d_zbar2 *= r * r;
			}
			d2_a_smooth_d_zbar2 /= z.conj();
			d2_a_smooth_d_zbar2 *= 2;

			d2_a_smooth_d_zbar2 *= Complex<T>(0, -kappastar * std::numbers::inv_pi_v<T>);
		}
		else
		{
			Complex<T> c1 = corner.conj() - z.conj();
			Complex<T> c2 = corner - z.conj();
			Complex<T> c3 = -corner - z.conj();
			Complex<T> c4 = -corner.conj() - z.conj();

			d2_a_smooth_d_zbar2 = -1 / c1 + 1 / c2 + 1 / c3 - 1 / c4;
			d2_a_smooth_d_zbar2 *= Complex<T>(0, -kappastar * std::numbers::inv_pi_v<T>);
		}
	}
	else
	{
		if (approx)
		{
			d2_a_smooth_d_zbar2 = 0;
		}
		else
		{
			d2_a_smooth_d_zbar2 = -2 * kappastar * corner.abs() * corner.abs() / z.conj().pow(3);
		}
	}

	return d2_a_smooth_d_zbar2;
}

