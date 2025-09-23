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
find the x and y intersections of a line connecting two points at the
provided x or y values
******************************************************************************/
template <typename T>
__host__ __device__ T get_x_intersection(T y, Complex<T> p1, Complex<T> p2)
{
	T dx = (p2.re - p1.re);
	/******************************************************************************
	if it is a vertical line, return the x coordinate of p1
	******************************************************************************/
	if (dx == 0)
	{
		return p1.re;
	}
	T log_dx = log(fabs(dx));
	T dy = (p2.im - p1.im);
	T log_dy = log(fabs(dy));
	
	/******************************************************************************
	parameter t in parametric equation of a line
	x = x0 + t * dx
	y = y0 + t * dy
	******************************************************************************/
	T log_t = log(fabs(y - p1.im)) - log_dy;
	
	T x = p1.re + sgn(y - p1.im) * sgn(dy) * sgn(dx) * exp(log_t + log_dx);
	return x;
}
template <typename T>
__host__ __device__ T get_y_intersection(T x, Complex<T> p1, Complex<T> p2)
{
	T dy = (p2.im - p1.im);
	/******************************************************************************
	if it is a horizontal line, return the y coordinate of p1
	******************************************************************************/
	if (dy == 0)
	{
		return p1.im;
	}
	T log_dy = log(fabs(dy));
	T dx = (p2.re - p1.re);
	T log_dx = log(fabs(dx));

	/******************************************************************************
	parameter t in parametric equation of a line
	x = x0 + t * dx
	y = y0 + t * dy
	******************************************************************************/
	T log_t = log(fabs(x - p1.re)) - log_dx;
	
	T y = p1.im + sgn(x - p1.re) * sgn(dx) * sgn(dy) * exp(log_t + log_dy);
	return y;
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

