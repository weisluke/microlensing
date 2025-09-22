#pragma once

#include "complex.cuh"
#include "util/math_util.cuh"


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

/******************************************************************************
Kahan sum of products algorithm for a pair of complex numbers multiplied by
scalars

\param c1,a1,c2,a2 -- variables to compute (c1 * a1 + c2 * a2)
******************************************************************************/
template<typename T>
__host__ __device__ Complex<T> Ksop(Complex<T> c1, T a1,
									Complex<T> c2 = Complex<T>(), T a2 = 0)
{
	T re = Ksop(c1.re, a1, c2.re, a2);
	T im = Ksop(c1.im, a1, c2.im, a2);
	return Complex<T>(re, im);
}

/******************************************************************************
Kahan-Babushka-Neumaier sum for complex numbers

\param val -- current value
\param err -- current error
\param to_add -- what to add
\param err_to_add -- error in what to add
******************************************************************************/
template<typename T, typename U>
__host__ __device__ void KBNsum(Complex<T>& val, Complex<T>& err,
								Complex<U> to_add, Complex<U> err_to_add = Complex<U>())
{
	KBNsum(val.re, err.re, to_add.re, err_to_add.re);
	KBNsum(val.im, err.im, to_add.im, err_to_add.im);
}

