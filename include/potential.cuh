#pragma once


/******************************************************************************
structure to hold macro-model potential derivatives
******************************************************************************/
template <typename T>
struct potential
{
	T p11 = 0;
    T p12 = 0;
    T p22 = 0;
	T p111 = 0;
    T p112 = 0;
    T p122 = 0;
    T p222 = 0;
};

