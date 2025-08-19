#pragma once

#include "complex.cuh"


/******************************************************************************
return the sign of a number

\param x -- number to find the sign of

\return -1, 0, or 1
******************************************************************************/
template <typename T>
__host__ __device__ int sgn(T x)
{
	if (x < -0) return -1;
	if (x > 0) return 1;
	return 0;
}

/******************************************************************************
Heaviside Step Function

\param x -- number to evaluate

\return 1 if x >= 0, 0 if x < 0
******************************************************************************/
template <typename T>
__host__ __device__ int heaviside(T x)
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
__host__ __device__ int boxcar(Complex<T> z, Complex<T> corner)
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

