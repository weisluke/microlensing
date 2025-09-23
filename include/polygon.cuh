#pragma once

#include "complex.cuh"
#include "util/math_util.cuh"


namespace polygon
{
	/******************************************************************************
	the intersection polygons will contain at most 7 points
	******************************************************************************/
	const int MAX_NUM_SIDES = 7;
}

/******************************************************************************
template class for handling the intersection of a triangle with square pixels
******************************************************************************/
template <typename T>
class Polygon
{
public:
	Complex<T> points[polygon::MAX_NUM_SIDES];
	int numsides;

	/******************************************************************************
	default constructor
	******************************************************************************/
	__host__ __device__ Polygon(Complex<T>* pts = nullptr, int len = 0)
	{
		if (len > polygon::MAX_NUM_SIDES)
		{
			len = 0;
		}
		for (int i = 0; i < len; i++)
		{
			points[i] = pts[i];
		}
		for (int i = len; i < polygon::MAX_NUM_SIDES; i++)
		{
			points[i] = Complex<T>();
		}
		numsides = len;
	}

	/******************************************************************************
	copy constructor
	******************************************************************************/
	template <typename U> __host__ __device__ Polygon(const Polygon<U>& poly)
	{
		for (int i = 0; i < poly.numsides; i++)
		{
			points[i] = poly.points[i];
		}
		for (int i = poly.numsides; i < polygon::MAX_NUM_SIDES; i++)
		{
			points[i] = Complex<T>();
		}
		numsides = poly.numsides;
	}

	/******************************************************************************
	assignment operator
	******************************************************************************/
	template <typename U> __host__ __device__ Polygon& operator=(const Polygon<U>& poly)
	{
		for (int i = 0; i < poly.numsides; i++)
		{
			points[i] = poly.points[i];
		}
		for (int i = poly.numsides; i < polygon::MAX_NUM_SIDES; i++)
		{
			points[i] = Complex<T>();
		}
		numsides = poly.numsides;
		return *this;
	}

	/******************************************************************************
	add a new point to the polygon
	******************************************************************************/
	template <typename U> __host__ __device__ void add_point(Complex<U> pt)
	{
		/******************************************************************************
		if polygon is full, set numsides to 0. this effectively removes the polygon
		as it is in error
		******************************************************************************/
		if (numsides >= polygon::MAX_NUM_SIDES)
		{
			for (int i = 0; i < polygon::MAX_NUM_SIDES; i++)
			{
				points[i] = Complex<T>();
			}
			numsides = 0;
			return;
		}
		points[numsides++] = pt;
	}

	/******************************************************************************
	calculate the area of the polygon using the shoelace formula
	******************************************************************************/
	__host__ __device__ T area()
	{
		/******************************************************************************
		a polygon must have at least 3 sides to have an area
		******************************************************************************/
		if (numsides < 3)
		{
			return 0;
		}

		T result = 0;

		/******************************************************************************
		the shoelace formula adds x_i * y_i+1 - x_i+1 * y_i to the area
		******************************************************************************/
		for (int i = 0; i < numsides; i++)
		{
			result += Ksop(points[i].re, points[(i + 1) % numsides].im,
						   -points[(i + 1) % numsides].re, points[i].im);
		}

		return result / 2;
	}

	/******************************************************************************
	find the x and y intersections of a line connecting two points at the
	provided x or y values
	******************************************************************************/
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
	calculate the minimum and maximum x and y values of the polygon coordinates
	******************************************************************************/
	__host__ __device__ T get_min_x()
	{
		T min = points[0].re;
		for (int i = 1; i < numsides; i++)
		{
			if (points[i].re < min)
			{
				min = points[i].re;
			}
		}
		return min;
	}
	__host__ __device__ T get_max_x()
	{
		T max = points[0].re;
		for (int i = 1; i < numsides; i++)
		{
			if (points[i].re > max)
			{
				max = points[i].re;
			}
		}
		return max;
	}
	__host__ __device__ T get_min_y()
	{
		T min = points[0].im;
		for (int i = 1; i < numsides; i++)
		{
			if (points[i].im < min)
			{
				min = points[i].im;
			}
		}
		return min;
	}
	__host__ __device__ T get_max_y()
	{
		T max = points[0].im;
		for (int i = 1; i < numsides; i++)
		{
			if (points[i].im > max)
			{
				max = points[i].im;
			}
		}
		return max;
	}

	/******************************************************************************
	clip a polygon along a vertical or horizontal line
	******************************************************************************/
	__host__ __device__ Polygon clip_at_x(T x, bool clip_left)
	{
		Polygon left_poly;
		Polygon right_poly;

		for (int i = 0; i < numsides; i++)
		{
			if (points[i].re < x)
			{
				left_poly.add_point(points[i]);
				if (points[(i + 1) % numsides].re > x)
				{
					T y = this->get_y_intersection(x, points[i], points[(i + 1) % numsides]);

					left_poly.add_point(Complex<T>(x, y));
					right_poly.add_point(Complex<T>(x, y));
				}
			}
			else if (points[i].re > x)
			{
				right_poly.add_point(points[i]);
				if (points[(i + 1) % numsides].re < x)
				{
					T y = this->get_y_intersection(x, points[i], points[(i + 1) % numsides]);

					left_poly.add_point(Complex<T>(x, y));
					right_poly.add_point(Complex<T>(x, y));
				}
			}
			else
			{
				left_poly.add_point(points[i]);
				right_poly.add_point(points[i]);
			}
		}

		if (clip_left)
		{
			*this = right_poly;
			return left_poly;
		}
		else
		{
			*this = left_poly;
			return right_poly;
		}
	}
	__host__ __device__ Polygon clip_at_y(T y, bool clip_bottom)
	{
		Polygon bottom_poly;
		Polygon top_poly;

		for (int i = 0; i < numsides; i++)
		{
			if (points[i].im < y)
			{
				bottom_poly.add_point(points[i]);
				if (points[(i + 1) % numsides].im > y)
				{
					T x = this->get_x_intersection(y, points[i], points[(i + 1) % numsides]);

					bottom_poly.add_point(Complex<T>(x, y));
					top_poly.add_point(Complex<T>(x, y));
				}
			}
			else if (points[i].im > y)
			{
				top_poly.add_point(points[i]);
				if (points[(i + 1) % numsides].im < y)
				{
					T x = this->get_x_intersection(y, points[i], points[(i + 1) % numsides]);

					bottom_poly.add_point(Complex<T>(x, y));
					top_poly.add_point(Complex<T>(x, y));
				}
			}
			else
			{
				bottom_poly.add_point(points[i]);
				top_poly.add_point(points[i]);
			}
		}

		if (clip_bottom)
		{
			*this = top_poly;
			return bottom_poly;
		}
		else
		{
			*this = bottom_poly;
			return top_poly;
		}
	}

	__device__ bool allocate_area_among_pixels(T factor, T* pixels, Complex<int> npixels)
	{
		if (numsides < 3)
		{
			return false;
		}

		/******************************************************************************
		if the point is entirely outside the pixel region, return
		******************************************************************************/
		if (this->get_min_x() >= npixels.re || this->get_max_x() <= 0
			|| this->get_min_y() >= npixels.im || this->get_max_y() <= 0)
		{
			return false;
		}

		/******************************************************************************
		get the initial polygon area for use later
		******************************************************************************/
		T whole_area = fabs(this->area());
		if (whole_area == 0)
		{
			return false;
		}

		/******************************************************************************
		if any region of the polygon extends outside the pixel region, clip it
		******************************************************************************/
		if (this->get_min_x() < 0)
		{
			this->clip_at_x(0, true);
		}
		if (this->get_max_x() > npixels.re)
		{
			this->clip_at_x(npixels.re, false);
		}
		if (this->get_min_y() < 0)
		{
			this->clip_at_y(0, true);
		}
		if (this->get_max_y() > npixels.im)
		{
			this->clip_at_y(npixels.im, false);
		}

		/******************************************************************************
		if it's not actually a polygon after clipping, return
		******************************************************************************/
		if (numsides < 3)
		{
			return false;
		}


		int xmin = static_cast<int>(this->get_min_x());
		int xmax = static_cast<int>(this->get_max_x());
		if (xmax == npixels.re)
		{
			xmax--;
		}

		for (int x = xmin; x < xmax; x++)
		{
			Polygon top_poly = this->clip_at_x(x + 1, true);
			if (top_poly.numsides < 3)
			{
				continue;
			}

			int ymin = static_cast<int>(top_poly.get_min_y());
			int ymax = static_cast<int>(top_poly.get_max_y());
			if (ymax == npixels.im)
			{
				ymax--;
			}

			for (int y = ymin; y < ymax; y++)
			{
				Polygon bottom_poly = top_poly.clip_at_y(y + 1, true);
				if (bottom_poly.numsides < 3)
				{
					continue;
				}
				T to_add = factor * fabs(bottom_poly.area()) / whole_area;
				atomicAdd(&pixels[npixels.re * y + x], to_add);
			}
			T to_add = factor * fabs(top_poly.area()) / whole_area;
			atomicAdd(&pixels[npixels.re * ymax + x], to_add);
		}

		if (numsides < 3)
		{
			return true;
		}

		int ymin = static_cast<int>(this->get_min_y());
		int ymax = static_cast<int>(this->get_max_y());
		if (ymax == npixels.im)
		{
			ymax--;
		}

		for (int y = ymin; y < ymax; y++)
		{
			Polygon bottom_poly = this->clip_at_y(y + 1, true);
			if (bottom_poly.numsides < 3)
			{
				continue;
			}
			T to_add = factor * fabs(bottom_poly.area()) / whole_area;
			atomicAdd(&pixels[npixels.re * y + xmax], to_add);
		}

		T to_add = factor * fabs(this->area()) / whole_area;
		atomicAdd(&pixels[npixels.re * ymax + xmax], to_add);

		return true;

	}

};

