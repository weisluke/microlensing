#pragma once

#include "complex.cuh"


namespace microlensing
{

/******************************************************************************
calculate the deflection angle due to the macromodel

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear

\return kappa * z - gamma * z_bar
******************************************************************************/
template <typename T>
__host__ __device__ Complex<T> alpha_macro(Complex<T> z, T kappa, T gamma)
{
    return kappa * z - gamma * z.conj();
}

/******************************************************************************
calculate the derivative of the deflection angle due to the macromodel with
respect to z

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear

\return kappa
******************************************************************************/
template <typename T>
__host__ __device__ T d_alpha_macro_d_z(Complex<T> z, T kappa, T gamma)
{
    return kappa;
}

/******************************************************************************
calculate the derivative of the deflection angle due to the macromodel with
respect to zbar

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear

\return -gamma
******************************************************************************/
template <typename T>
__host__ __device__ T d_alpha_macro_d_zbar(Complex<T> z, T kappa, T gamma)
{
    return -gamma;
}

}

