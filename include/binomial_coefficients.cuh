#pragma once


/******************************************************************************
calculate the binomial coefficients up to a maximum order
stored as long long: C(2p-1, p-1) with p up to MAX_EXPANSION_ORDER = 31 exceeds int32
assumes this function will be called once, otherwise the recursion is redundant

\param binom_coeffs -- pointer to array to hold coefficients
\param n -- maximum order to find the coefficients for
******************************************************************************/
void calculate_binomial_coeffs(long long* binom_coeffs, int n)
{
	/******************************************************************************
	recursion limit
	******************************************************************************/
	if (n < 0) //should never hit this unless the maximum order is miscalculated
	{
		return;
	}
	if (n == 0)
	{
		binom_coeffs[0] = 1;
		return;
	}

	calculate_binomial_coeffs(binom_coeffs, n - 1);

	/******************************************************************************
	rows of Pascal's triangle
	******************************************************************************/
	int row_start = n * (n + 1) / 2; //index of row within array is triangular number
	int prev_row_start = (n - 1) * n / 2;
	for (int i = 0; i <= n; i++)
	{
		if (i == 0 || i == n)
		{
			binom_coeffs[row_start + i] = 1;
		}
		else
		{
			binom_coeffs[row_start + i] = binom_coeffs[prev_row_start + i - 1] + binom_coeffs[prev_row_start + i];
		}
	}
}

/******************************************************************************
return the value of the binomial coefficient (n, k)

\param binom_coeffs -- pointer to array holding coefficients
\param n -- maximum order for the coefficients
\param k -- order to find the coefficient of
******************************************************************************/
__host__ __device__ long long get_binomial_coeff(long long* binom_coeffs, int n, int k)
{
	/******************************************************************************
	binomial coefficient only defined if n >= k
	******************************************************************************/
	if (n < k)
	{
		return 0;
	}
	int row_start = n * (n + 1) / 2; //index of row within array is triangular number
	return binom_coeffs[row_start + k];
}

