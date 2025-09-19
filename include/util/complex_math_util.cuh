#pragma once

#include "complex.cuh"


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

