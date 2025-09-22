#pragma once

#include <cmath>


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
Kahan sum of products algorithm
adapted from the Kahan difference of products algorithm, to be agnostic to the
sign of the second term

\param a,b,c,d -- variables to compute (a * b + c * d)
******************************************************************************/
template<typename T>
__host__ __device__ T Ksop(T a, T b, T c = 0, T d = 0)
{
	T ab = a * b;
	T cd = c * d;
	T err, sop;

	/******************************************************************************
	if |ab| >= |cd|, error in ab is larger due to larger jumps between floating
	point values
	a * b = ab + err   -->   a * b - ab = err
	a * b + c * d = (ab + err) + c * d = (c * d + ab) + err
	******************************************************************************/
	if (std::abs(ab) >= std::abs(cd))
	{
		err = std::fma(a, b, -ab);
		sop = std::fma(c, d, ab);
	}
	/******************************************************************************
	flip order otherwise
	******************************************************************************/
	else
	{
		err = std::fma(c, d, -cd);
		sop = std::fma(a, b, cd);
	}
	return sop + err;
}
/******************************************************************************
separate implementation for ints, since there is no fma for ints
******************************************************************************/
__host__ __device__ inline int Ksop(int a, int b, int c = 0, int d = 0)
{
    return a * b + c * d;
}

/******************************************************************************
Kahan-Babushka-Neumaier sum

\param val -- current value
\param err -- current error
\param to_add -- what to add
\param err_to_add -- error in what to add
******************************************************************************/
template<typename T, typename U>
__host__ __device__ void KBNsum(T& val, T& err, U to_add, U err_to_add = 0)
{
	T new_val;

	new_val = val + to_add;
	if (std::abs(val) >= std::abs(to_add))
	{
		err += (val - new_val) + to_add;
	}
	else
	{
		err += (to_add - new_val) + val;
	}
	val = new_val;

	//if not adding additional error, early exit
	if (err_to_add == 0) return;
	//otherwise, add on the error
	KBNsum(val, err, err_to_add);
}

/******************************************************************************
multiply two numbers and retain proper error values
idea taken from the Kahan sum of products algorithm

\param val -- current value
\param err -- current error
\param to_mult -- what to multiply
\param err_to_mult -- error in what to multiply
******************************************************************************/
template<typename T>
__host__ __device__ void Kprod(T& val, T& err, T to_mult, T err_to_mult = 0)
{
	T new_val = 0;
	T new_err = 0;
	T prod; //product of various terms

	/******************************************************************************
	a * b = ab + err  -->  a * b - ab = err
	******************************************************************************/
	prod = val * to_mult;
	KBNsum(new_val, new_err, prod, std::fma(val, to_mult, -prod));

	prod = err * to_mult;
	KBNsum(new_val, new_err, prod, std::fma(err, to_mult, -prod));

	if (err_to_mult != 0)
	{
		prod = val * err_to_mult;
		KBNsum(new_val, new_err, prod, std::fma(val, err_to_mult, -prod));

		prod = err * err_to_mult;
		KBNsum(new_val, new_err, prod, std::fma(err, err_to_mult, -prod));
	}

	val = new_val;
	err = new_err;
}
/******************************************************************************
separate implementation for ints, since there is no fma for ints
******************************************************************************/
__host__ __device__ void Kprod(int& val, int& err, int to_mult, int err_to_mult = 0)
{
	int new_val = 0;
	int new_err = 0;

	KBNsum(new_val, new_err, val * to_mult, err * to_mult);
	if (err_to_mult != 0)
	{
		KBNsum(new_val, new_err, val * err_to_mult, err * err_to_mult);
	}

	val = new_val;
	err = new_err;
}

