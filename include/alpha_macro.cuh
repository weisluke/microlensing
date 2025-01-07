#pragma once

#include "complex.cuh"
#include "potential.cuh"


/******************************************************************************
calculate the deflection angle due to the macro-model using a Taylor series
expansion of the potential

\param z -- complex image plane position
\param p -- structure containing macro-model derivatives of the potential

\return alpha_macro
******************************************************************************/
template <typename T>
__device__ Complex<T> alpha_macro(Complex<T> z, potential<T> p)
{
    Complex<T> a_macro;
    
    a_macro += ((p.p11 + p.p22) * z + Complex<T>(p.p11 - p.p22, 2 * p.p12) * z.conj()) / 2;
	a_macro += (Complex<T>(p.p111 + p.p122, -(p.p112 + p.p222)) * z * z
                + 2 * Complex<T>(p.p111 + p.p122, p.p112 + p.p222) * z * z.conj()
                + Complex<T>(p.p111 - 3 * p.p122, 3 * p.p112 - p.p222) * z.conj() * z.conj()
               ) / 8;

    return a_macro;
}

/******************************************************************************
calculate the derivative of the deflection angle with respect to z due to the
macro-model using a Taylor series expansion of the potential

\param z -- complex image plane position
\param p -- structure containing macro-model derivatives of the potential

\return d_alpha_macro_d_z
******************************************************************************/
template <typename T>
__device__ T d_alpha_macro_d_z(Complex<T> z, potential<T> p)
{
    T d_a_macro_d_z;
    
    d_a_macro_d_z += (p.p11 + p.p22) / 2;
	d_a_macro_d_z += (Complex<T>(p.p111 + p.p122, -(p.p112 + p.p222)) * z).re / 2;

    return d_a_macro_d_z;
}

/******************************************************************************
calculate the derivative of the deflection angle with respect to zbar due to
the macro-model using a Taylor series expansion of the potential

\param z -- complex image plane position
\param p -- structure containing macro-model derivatives of the potential

\return d_alpha_macro_d_zbar
******************************************************************************/
template <typename T>
__device__ Complex<T> d_alpha_macro_d_zbar(Complex<T> z, potential<T> p)
{
    Complex<T> d_a_macro_d_zbar;
    
    d_a_macro_d_zbar += Complex<T>(p.p11 - p.p22, 2 * p.p12) / 2
	d_a_macro_d_zbar += (Complex<T>(p.p111 + p.p122, p.p112 + p.p222) * z
                         + Complex<T>(p.p111 - 3 * p.p122, 3 * p.p112 - p.p222) * z.conj()
                        ) / 4;

    return d_a_macro_d_zbar;
}

/******************************************************************************
calculate the second derivative of the deflection angle with respect to z due
to the macro-model using a Taylor series expansion of the potential

\param z -- complex image plane position
\param p -- structure containing macro-model derivatives of the potential

\return d2_alpha_macro_d_z2
******************************************************************************/
template <typename T>
__device__ Complex<T> d2_alpha_macro_d_z2(Complex<T> z, potential<T> p)
{
    Complex<T> d2_a_macro_d_z2;
    
	d2_a_macro_d_z2 += Complex<T>(p.p111 + p.p122, -(p.p112 + p.p222)) / 4;

    return d2_a_macro_d_z2;
}

/******************************************************************************
calculate the second derivative of the deflection angle with respect to z and
zbar due to the macro-model using a Taylor series expansion of the potential

\param z -- complex image plane position
\param p -- structure containing macro-model derivatives of the potential

\return d2_alpha_macro_d_z_d_zbar
******************************************************************************/
template <typename T>
__device__ Complex<T> d2_alpha_macro_d_z_d_zbar(Complex<T> z, potential<T> p)
{
    Complex<T> d2_a_macro_d_z_d_zbar;
    
	d2_a_macro_d_z_d_zbar += Complex<T>(p.p111 + p.p122, p.p112 + p.p222) / 4;

    return d2_a_macro_d_z_d_zbar;
}

/******************************************************************************
calculate the second derivative of the deflection angle with respect to zbar
due to the macro-model using a Taylor series expansion of the potential

\param z -- complex image plane position
\param p -- structure containing macro-model derivatives of the potential

\return d2_alpha_macro_d_zbar2
******************************************************************************/
template <typename T>
__device__ Complex<T> d2_alpha_macro_d_zbar2(Complex<T> z, potential<T> p)
{
    Complex<T> d2_a_macro_d_zbar2;
    
	d2_a_macro_d_zbar2 += Complex<T>(p.p111 - 3 * p.p122, 3 * p.p112 - p.p222) / 4;

    return d2_a_macro_d_zbar2;
}

