#pragma once

#include "util/math_util.cuh"

#include <cmath>
#include <iostream>


/******************************************************************************
template class for handling complex number arithmetic with operator overloads
******************************************************************************/
template <typename T>
class Complex
{
public:
	T re;
	T im;

	/******************************************************************************
	default constructor initializes the complex number to zero
	******************************************************************************/
	__host__ __device__ Complex()
	{
		re = 0;
		im = 0;
	}
	template <typename U> __host__ __device__ Complex(U real)
	{
		re = static_cast<T>(real);
		im = 0;
	}
	template <typename U, typename V> __host__ __device__ Complex(U real, V imag)
	{
		re = static_cast<T>(real);
		im = static_cast<T>(imag);
	}

	/******************************************************************************
	copy constructor
	******************************************************************************/
	template <typename U> __host__ __device__ Complex(const Complex<U>& c1)
	{
		re = static_cast<T>(c1.re);
		im = static_cast<T>(c1.im);
	}

	/******************************************************************************
	assignment operators
	******************************************************************************/
	template <typename U> __host__ __device__ Complex& operator=(const Complex<U>& c1)
	{
		re = static_cast<T>(c1.re);
		im = static_cast<T>(c1.im);
		return *this;
	}
	template <typename U> __host__ __device__ Complex& operator=(const U& num)
	{
		re = static_cast<T>(num);
		im = 0;
		return *this;
	}

	/******************************************************************************
	print a complex number to screen
	******************************************************************************/
	friend std::ostream& operator<< (std::ostream& out, const Complex& c1)
	{
		out << "(" << c1.re << ", " << c1.im << ")";
		return out;
	}

	/******************************************************************************
	complex conjugate of the complex number
	******************************************************************************/
	__host__ __device__ Complex conj()
	{
		return Complex(re, -im);
	}

	/******************************************************************************
	norm of the complex number = sqrt(re*re + im*im)
	******************************************************************************/
	__host__ __device__ T abs()
	{
		return std::hypot(re, im); //avoids under/overflow
	}

	/******************************************************************************
	argument of the complex number in the range [-pi, pi]
	******************************************************************************/
	__host__ __device__ T arg()
	{
		return std::atan2(im, re);
	}

	/******************************************************************************
	exponential of the complex number
	******************************************************************************/
	__host__ __device__ Complex exp()
	{
		return Complex(std::exp(re) * std::cos(im), std::exp(re) * std::sin(im));
	}

	/******************************************************************************
	logarithm of the complex number
	******************************************************************************/
	__host__ __device__ Complex log()
	{
		T abs = this->abs();
		T arg = this->arg();
		
		return Complex(std::log(abs), arg);
	}

	/******************************************************************************
	positive and negative operators
	******************************************************************************/
	__host__ __device__ Complex operator+()
	{
		return Complex(re, im);
	}
	__host__ __device__ Complex operator-()
	{
		return Complex(-re, -im);
	}

	/******************************************************************************
	addition
	******************************************************************************/
	template <typename U> __host__ __device__ Complex& operator+=(Complex<U> c1)
	{
		re += c1.re;
		im += c1.im;
		return *this;
	}
	template <typename U> __host__ __device__ Complex& operator+=(U num)
	{
		re += num;
		return *this;
	}
	template <typename U> __host__ __device__ friend Complex operator+(Complex c1, Complex<U> c2)
	{
		Complex res = c1;
		res += c2;
		return res;
	}
	template <typename U> __host__ __device__ friend Complex operator+(Complex c1, U num)
	{
		Complex res = c1;
		res += num;
		return res;
	}
	template <typename U> __host__ __device__ friend Complex operator+(T num, Complex<U> c1)
	{
		Complex res = num;
		res += c1;
		return res;
	}

	/******************************************************************************
	subtraction
	******************************************************************************/
	template <typename U> __host__ __device__ Complex& operator-=(Complex<U> c1)
	{
		re -= c1.re;
		im -= c1.im;
		return *this;
	}
	template <typename U> __host__ __device__ Complex& operator-=(U num)
	{
		re -= num;
		return *this;
	}
	template <typename U> __host__ __device__ friend Complex operator-(Complex c1, Complex<U> c2)
	{
		Complex res = c1;
		res -= c2;
		return res;
	}
	template <typename U> __host__ __device__ friend Complex operator-(Complex c1, U num)
	{
		Complex res = c1;
		res -= num;
		return res;
	}
	template <typename U> __host__ __device__ friend Complex operator-(T num, Complex<U> c1)
	{
		Complex res = num;
		res -= c1;
		return res;
	}

	/******************************************************************************
	multiplication
	(a + b * i) * (c + d * i)
	= (a * c - b * d) + (a * d  + b * c) * i
	******************************************************************************/
	template <typename U> __host__ __device__ Complex& operator*=(Complex<U> c1)
	{
		T new_re = Ksop(re, c1.re, -im, c1.im);
		T new_im = Ksop(re, c1.im, im, c1.re);
		re = new_re;
		im = new_im;
		return *this;
	}
	template <typename U> __host__ __device__ Complex& operator*=(U num)
	{
		re *= num;
		im *= num;
		return *this;
	}
	template <typename U> __host__ __device__ friend Complex operator*(Complex c1, Complex<U> c2)
	{
		Complex res = c1;
		res *= c2;
		return res;
	}
	template <typename U> __host__ __device__ friend Complex operator*(Complex c1, U num)
	{
		Complex res = c1;
		res *= num;
		return res;
	}
	template <typename U> __host__ __device__ friend Complex operator*(T num, Complex<U> c1)
	{
		Complex res = num;
		res *= c1;
		return res;
	}

	/******************************************************************************
	division
	(a + b * i) / (c + d * i) = (a + b * i) * (c - d * i) / (c^2 + d^2)
	= (a * c + b * d) / (c^2 + d^2) + (b * c - a * d) * i / (c^2 + d^2)
	= (a + b * d / c) / (c + d * d / c) + (b - a * d / c) * i / (c + d * d / c)
	= (a * c / d + b) / (c * c / d + d) + (b * c / d - a) * i / (c * c / d + d)
	******************************************************************************/
	template <typename U> __host__ __device__ Complex& operator/=(Complex<U> c1)
	{
		T f1, f2; //scale factors to avoid under/over flow
		T new_re;
		T new_im;
		
		if (std::abs(c1.im) < std::abs(c1.re))
		{
			f1 = c1.re / c1.re; //purely to cast 1 to type T
			f2 = c1.im / c1.re;
		}
		else
		{
			f1 = c1.re / c1.im;
			f2 = c1.im / c1.im;
		}
		//use Smith's formula
		new_re = Ksop(re, f1, im, f2) / Ksop(c1.re, f1, c1.im, f2);
		new_im = Ksop(im, f1, -re, f2) / Ksop(c1.re, f1, c1.im, f2);

		re = new_re;
		im = new_im;
		return *this;
	}
	template <typename U> __host__ __device__ Complex& operator/=(U num)
	{
		re /= num;
		im /= num;
		return *this;
	}
	template <typename U> __host__ __device__ friend Complex operator/(Complex c1, Complex<U> c2)
	{
		Complex res = c1;
		res /= c2;
		return res;
	}
	template <typename U> __host__ __device__ friend Complex operator/(Complex c1, U num)
	{
		Complex res = c1;
		res /= num;
		return res;
	}
	template <typename U> __host__ __device__ friend Complex operator/(T num, Complex<U> c1)
	{
		Complex res = num;
		res /= c1;
		return res;
	}

	/******************************************************************************
	exponentiation
	******************************************************************************/
	__host__ __device__ Complex pow(int num)
	{
		Complex res = Complex(1, 0);

		if (num > 0)
		{
			for (int i = 0; i < num; i++)
			{
				res *= *this;
			}
		}
		else if (num < 0)
		{
			for (int i = 0; i > num; i--)
			{
				res *= *this;
			}
			res = 1 / res;
		}

		return res;
	}
	template <typename U> __host__ __device__ Complex pow(U num)
	{
		return (num * this->log()).exp();
	}

};

