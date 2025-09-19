#pragma once


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

