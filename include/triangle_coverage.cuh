#pragma once

#include "complex.cuh"

#include <cmath>

/******************************************************************************
Exact allocation of a triangle's area among the pixels it overlaps, without
building clipped polygons.

For a counter-clockwise polygon P and the unit pixel [i,i+1]x[j,j+1],
Green's theorem gives

    area(P ∩ pixel) = sum over edges e of
        sign(dy_e) * ∫_{y in [j,j+1] ∩ y-range(e)} ( clamp(x_e(y), i, i+1) - i ) dy

x_e(y) is linear, so the integral of the clamped linear function has a closed
form built from two "integral of max(u,0)" terms. Everything is straight-line
arithmetic on a handful of registers: no dynamic indexing, no modulo, no
polygon copies, no log/exp.

Two precisions are involved: T is the precision of the vertices (the lens
mapping), P the precision used for the per-pixel arithmetic. Before the
per-pixel work the vertices are shifted so that the lower-left pixel of the
(clamped) bounding box is the origin; the coordinates that remain are a few
pixels in size, so P = float would be viable even inside the double build and
even for maps with 10^5 pixels per side. The double build nevertheless uses
P = T: single-precision pixel arithmetic changes its maps at the 1e-7 (mean)
to 1e-5 (p99.9) relative level, far above its 1e-13 rounding floor.

The original code adds factor*|piece|/|whole| per pixel; summing the signed
coverage and scaling by factor/area (signed) is identical.
******************************************************************************/
namespace tri_cov
{
	/******************************************************************************
	∫_0^L max(u(t), 0) dt for u linear from a to b over an interval of length L
	= L * (max(a,0) + max(b,0))^2 / (2 * (|a| + |b|))
	(exact for same-sign and mixed-sign endpoints; 0/0 guarded)
	******************************************************************************/
	template <typename P>
	__host__ __device__ __forceinline__ P int_max0(P a, P b, P L)
	{
		P pa = a > 0 ? a : P(0);
		P pb = b > 0 ? b : P(0);
		P num = pa + pb;
		P den = fabs(a) + fabs(b);
		if constexpr (sizeof(P) == 8)
		{
			//a zero numerator skips the division: the FP64 division routes it through its slow path
			P t = L * num * num;
			return t > 0 ? t / (2 * den) : P(0);
		}
		else
		{
			return den > 0 ? L * num * num / (2 * den) : P(0);
		}
	}

	/******************************************************************************
	∫_0^L clamp(u(t), 0, 1) dt
	******************************************************************************/
	template <typename P>
	__host__ __device__ __forceinline__ P int_clamp01(P a, P b, P L)
	{
		return int_max0(a, b, L) - int_max0(a - 1, b - 1, L);
	}

	template <typename P>
	struct Edge
	{
		P xa, ya, xb, yb; //oriented so that ya <= yb
		P slope;          //dx/dy
		P sgn;            //+1 if the original edge went upward, -1 if downward, 0 if horizontal
	};

	template <typename P>
	__host__ __device__ __forceinline__ Edge<P> make_edge(P px, P py, P qx, P qy)
	{
		Edge<P> e;
		if (qy > py)
		{
			e.xa = px; e.ya = py; e.xb = qx; e.yb = qy; e.sgn = 1;
		}
		else
		{
			e.xa = qx; e.ya = qy; e.xb = px; e.yb = py; e.sgn = -1;
		}
		P dy = e.yb - e.ya;
		e.slope = dy > 0 ? (e.xb - e.xa) / dy : P(0);
		if (!(dy > 0)) e.sgn = 0;
		return e;
	}

	/******************************************************************************
	allocate the area of triangle (p0,p1,p2) among the pixels

	\param p0,p1,p2 -- vertices in pixel units (precision T)
	\param whole_area -- signed shoelace area of the triangle in precision T
	                     (already computed by the caller for the parity test)
	\param factor -- total amount to distribute among the pixels
	\param npix -- number of pixels in each direction
	\param add_one -- callable add_one(long long pixel_index, P value), used when
	                  the whole triangle lies inside one pixel (the hot path in
	                  magnified regions; the caller may aggregate these)
	\param add_many -- callable add_many(long long pixel_index, P value), used for
	                   the general multi-pixel case
	******************************************************************************/
	template <typename P, typename T, typename AddOne, typename AddMany>
	__host__ __device__ __forceinline__ void allocate_triangle(Complex<T> p0, Complex<T> p1, Complex<T> p2,
		T whole_area, P factor, Complex<int> npix, AddOne&& add_one, AddMany&& add_many)
	{
		if (whole_area == 0) return;

		T xmin = fmin(p0.re, fmin(p1.re, p2.re));
		T xmax = fmax(p0.re, fmax(p1.re, p2.re));
		T ymin = fmin(p0.im, fmin(p1.im, p2.im));
		T ymax = fmax(p0.im, fmax(p1.im, p2.im));

		/******************************************************************************
		entirely outside the pixel region
		******************************************************************************/
		if (xmin >= npix.re || xmax <= 0 || ymin >= npix.im || ymax <= 0) return;

		int ix0 = static_cast<int>(floor(xmin));
		int ix1 = static_cast<int>(floor(xmax));
		int iy0 = static_cast<int>(floor(ymin));
		int iy1 = static_cast<int>(floor(ymax));

		/******************************************************************************
		a vertex exactly on the upper/right boundary of the last pixel contributes
		zero area to the next pixel; drop that column/row
		******************************************************************************/
		if (ix1 > ix0 && static_cast<T>(ix1) == xmax) ix1--;
		if (iy1 > iy0 && static_cast<T>(iy1) == ymax) iy1--;

		/******************************************************************************
		fast path: the whole triangle lies inside one pixel (the common case in
		magnified regions). same behaviour as the original for zero-width slivers.
		******************************************************************************/
		if (ix0 == ix1 && iy0 == iy1)
		{
			add_one(1LL * npix.re * iy0 + ix0, factor);
			return;
		}

		ix0 = ix0 < 0 ? 0 : ix0;
		iy0 = iy0 < 0 ? 0 : iy0;
		ix1 = ix1 > npix.re - 1 ? npix.re - 1 : ix1;
		iy1 = iy1 > npix.im - 1 ? npix.im - 1 : iy1;

		/******************************************************************************
		shift the origin to the lower-left pixel of the clamped bounding box and
		continue in precision P
		******************************************************************************/
		const P x0 = static_cast<P>(p0.re - ix0), y0 = static_cast<P>(p0.im - iy0);
		const P x1 = static_cast<P>(p1.re - ix0), y1 = static_cast<P>(p1.im - iy0);
		const P x2 = static_cast<P>(p2.re - ix0), y2 = static_cast<P>(p2.im - iy0);

		/******************************************************************************
		area in precision P from the shifted vertices so that the pieces sum to
		factor exactly up to P rounding; the sign is that of whole_area
		******************************************************************************/
		const P area = P(0.5) * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
		if (area == 0) return;
		const P scale = factor / area; //signed: coverage has the sign of area

		Edge<P> e0 = make_edge(x0, y0, x1, y1);
		Edge<P> e1 = make_edge(x1, y1, x2, y2);
		Edge<P> e2 = make_edge(x2, y2, x0, y0);

		for (int iy = iy0; iy <= iy1; iy++)
		{
			P row_lo = static_cast<P>(iy - iy0);
			P row_hi = row_lo + 1;

			/******************************************************************************
			portion of each edge within this row, evaluated once per row
			******************************************************************************/
			P ra0 = fmax(e0.ya, row_lo), rb0 = fmin(e0.yb, row_hi), L0 = rb0 - ra0;
			P ra1 = fmax(e1.ya, row_lo), rb1 = fmin(e1.yb, row_hi), L1 = rb1 - ra1;
			P ra2 = fmax(e2.ya, row_lo), rb2 = fmin(e2.yb, row_hi), L2 = rb2 - ra2;
			L0 = L0 > 0 ? L0 : P(0);
			L1 = L1 > 0 ? L1 : P(0);
			L2 = L2 > 0 ? L2 : P(0);
			P xra0 = e0.xa + (ra0 - e0.ya) * e0.slope, xrb0 = xra0 + L0 * e0.slope;
			P xra1 = e1.xa + (ra1 - e1.ya) * e1.slope, xrb1 = xra1 + L1 * e1.slope;
			P xra2 = e2.xa + (ra2 - e2.ya) * e2.slope, xrb2 = xra2 + L2 * e2.slope;

			/******************************************************************************
			columns outside the span of the edges that cross this row receive zero
			coverage (the edge integrals cancel), so visit only the span. without this
			a sliver triangle with a tiny area but a huge bounding box costs as much
			as the whole map, and the cancellation left rounding noise of up to
			factor per pixel in thin slivers
			******************************************************************************/
			P lo = static_cast<P>(ix1 - ix0 + 1), hi = P(-1);
			if (L0 > 0) { lo = fmin(lo, fmin(xra0, xrb0)); hi = fmax(hi, fmax(xra0, xrb0)); }
			if (L1 > 0) { lo = fmin(lo, fmin(xra1, xrb1)); hi = fmax(hi, fmax(xra1, xrb1)); }
			if (L2 > 0) { lo = fmin(lo, fmin(xra2, xrb2)); hi = fmax(hi, fmax(xra2, xrb2)); }
			if (!(hi >= lo)) continue;
			const int ca = lo > 0 ? static_cast<int>(floor(lo)) : 0;
			const int cb = hi < static_cast<P>(ix1 - ix0) ? static_cast<int>(floor(hi)) : ix1 - ix0;

			for (int ix = ix0 + ca; ix <= ix0 + cb; ix++)
			{
				P col = static_cast<P>(ix - ix0);
				P cov = e0.sgn * int_clamp01(xra0 - col, xrb0 - col, L0)
					  + e1.sgn * int_clamp01(xra1 - col, xrb1 - col, L1)
					  + e2.sgn * int_clamp01(xra2 - col, xrb2 - col, L2);
				P to_add = cov * scale;
				/******************************************************************************
				guard against rounding (and degenerate slivers) producing values outside
				the physically meaningful range [0, factor]
				******************************************************************************/
				to_add = to_add < 0 ? P(0) : (to_add > factor ? factor : to_add);
				add_many(1LL * npix.re * iy + ix, to_add);
			}
		}
	}

	/******************************************************************************
	convenience overload with a single callback
	******************************************************************************/
	template <typename P, typename T, typename AddFn>
	__host__ __device__ __forceinline__ void allocate_triangle(Complex<T> p0, Complex<T> p1, Complex<T> p2,
		T whole_area, P factor, Complex<int> npix, AddFn&& add)
	{
		allocate_triangle<P, T>(p0, p1, p2, whole_area, factor, npix, add, add);
	}
}
