#pragma once

#include "complex.cuh"

#include <cfloat>
#include <cmath>

/******************************************************************************
Exact allocation of a triangle's area among the pixels it overlaps, without
building clipped polygons.

For a counter-clockwise polygon P and the unit pixel [i,i+1]x[j,j+1],
Green's theorem gives

    area(P ∩ pixel) = sum over edges e of
        sign(dy_e) * ∫_{y in [j,j+1] ∩ y-range(e)} ( clamp(x_e(y), i, i+1) - i ) dy

x_e(y) is linear, so the integral of the clamped linear function has a closed
form built from two "integral of max(u,0)" terms, each evaluated without a
division (see int_max0; the reciprocal slope is computed once per edge). Everything is straight-line
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

Triangles with a large bounding box can instead be deposited along their
perimeter, as values on the edges plus steps whose row-wise prefix sum fills
the interior (see allocate_triangle).
******************************************************************************/
namespace tri_cov
{
	/******************************************************************************
	∫_0^L max(u(t), 0) dt for u linear from a to b over an interval of length L
	= L * (max(a,0) + max(b,0))^2 / (2 * (|a| + |b|))

	the division is avoided. if a and b are both positive the integral is the
	trapezoid 0.5 * L * (a + b). otherwise (one endpoint <= 0) we have
	|a| + |b| = |a - b| = L * |slope|, so the integral equals
	(max(a,0) + max(b,0))^2 / (2 * |slope|), and 1 / (2 |slope|) is precomputed
	once per edge (inv2s, 0 for a vertical or horizontal edge). when both
	endpoints are <= 0 the numerator is 0, so the result is 0 as long as inv2s
	is finite. make_edge keeps it finite: a subnormal slope, whose reciprocal
	can overflow, counts as 0 (its integral is below the smallest normal number)

	\param inv2s -- 1 / (2 |slope|) of the edge, or 0 when |slope| is 0 or subnormal
	******************************************************************************/
	template <typename P>
	__host__ __device__ __forceinline__ P int_max0(P a, P b, P L, P inv2s)
	{
		P pa = a > 0 ? a : P(0);
		P pb = b > 0 ? b : P(0);
		P sum = pa + pb;
		return (a > 0 && b > 0) ? P(0.5) * L * sum : sum * sum * inv2s;
	}

	/******************************************************************************
	∫_0^L clamp(u(t), 0, 1) dt
	******************************************************************************/
	template <typename P>
	__host__ __device__ __forceinline__ P int_clamp01(P a, P b, P L, P inv2s)
	{
		return int_max0(a, b, L, inv2s) - int_max0(a - 1, b - 1, L, inv2s);
	}

	template <typename P>
	struct Edge
	{
		P xa, ya, xb, yb; //oriented so that ya <= yb
		P slope;          //dx/dy
		P sgn;            //+1 if the original edge went upward, -1 if downward, 0 if horizontal
		P inv2s;          //1 / (2 |slope|), 0 for a vertical or horizontal edge (or a subnormal slope)
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
		//the smallest normal number: below it 1 / (2 |slope|) can overflow (float and double alike)
		const P min_normal = sizeof(P) == sizeof(float) ? P(FLT_MIN) : P(DBL_MIN);
		e.inv2s = fabs(e.slope) >= min_normal ? P(1) / (2 * fabs(e.slope)) : P(0);
		return e;
	}

	/******************************************************************************
	allocate the area of triangle (p0,p1,p2) among the pixels

	\param p0,p1,p2 -- vertices in pixel units (precision T)
	\param whole_area -- signed shoelace area of the triangle in precision T
	                     (already computed by the caller for the parity test)
	\param factor -- total amount to distribute among the pixels
	\param npix -- number of pixels in each direction
	\param edge_min_bbox -- a triangle whose clamped bounding box has more pixels
	                        than this takes the perimeter path (LLONG_MAX: never;
	                        a triangle inside one pixel always goes to add_one)
	\param add_one -- callable add_one(long long pixel_index, P value), used when
	                  the whole triangle lies inside one pixel (the hot path in
	                  magnified regions; the caller may aggregate these)
	\param add_many -- callable add_many(long long pixel_index, P value), used for
	                   the general multi-pixel case. on the direct path the
	                   values are clamped to [0, factor]; on the perimeter path
	                   they are partial values, possibly negative and not
	                   clamped, that are exact only once the steps are summed
	\param add_step -- callable add_step(long long pixel_index, P value), used only
	                   by the perimeter path. the values go to a step buffer of
	                   the map's size whose row-wise inclusive prefix sum the
	                   caller adds to the map once all triangles are deposited
	******************************************************************************/
	template <typename P, typename T, typename AddOne, typename AddMany, typename AddStep>
	__host__ __device__ __forceinline__ void allocate_triangle(Complex<T> p0, Complex<T> p1, Complex<T> p2,
		T whole_area, P factor, Complex<int> npix, long long edge_min_bbox,
		AddOne&& add_one, AddMany&& add_many, AddStep&& add_step)
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

		/******************************************************************************
		perimeter path. for one edge and one pixel row, the edge's coverage of column
		i is sgn*L for the columns left of the edge's span in that row, the clamped
		integral on the span, and 0 to its right. the sgn*L terms of the three edges
		cancel, so each edge-row deposits sgn*part on its span (add_many) and a step
		of -sgn*L at the first span column (add_step); the row-wise prefix sum of the
		steps restores the columns between the edges. the cost is proportional to
		the perimeter instead of the area. exact up to rounding. there is no clamp
		to [0, factor] as in the direct path: the values deposited here are partial
		(a span value can be negative until the steps are summed).

		rows are in the frame of the clamped bounding box (rows 0..iy1-iy0); columns
		are clamped to the map, not to the bounding box, because a step can fall on
		the column just right of it. a span that starts left of the map has its step
		on column 0; one that starts right of the last column has no step at all
		******************************************************************************/
		if (1LL * (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > edge_min_bbox)
		{
			const int ncol = npix.re;
#pragma unroll 1
			for (int q = 0; q < 3; q++)
			{
				//selected, not indexed from an array, so that nothing goes to local memory
				const Edge<P> e = q == 0 ? e0 : (q == 1 ? e1 : e2);
				if (e.sgn == 0) continue;
				int r0 = static_cast<int>(floor(e.ya)), r1 = static_cast<int>(ceil(e.yb)) - 1;
				r0 = r0 < 0 ? 0 : r0;
				r1 = r1 > iy1 - iy0 ? iy1 - iy0 : r1;
				for (int r = r0; r <= r1; r++)
				{
					const P row_lo = static_cast<P>(r), row_hi = row_lo + 1;
					const P ra = fmax(e.ya, row_lo), rb = fmin(e.yb, row_hi);
					const P L = rb - ra;
					if (!(L > 0)) continue;
					const P xra = e.xa + (ra - e.ya) * e.slope, xrb = xra + L * e.slope;
					const P lo = fmin(xra, xrb), hi = fmax(xra, xrb);
					const int gs = ix0 + static_cast<int>(floor(lo)); //first span column, map frame
					const int ge = ix0 + static_cast<int>(floor(hi)); //last span column, map frame
					const long long rowbase = 1LL * ncol * (iy0 + r);
					const int cs = gs < 0 ? 0 : gs;
					const int ce = ge > ncol - 1 ? ncol - 1 : ge;
					if (gs < ncol) add_step(rowbase + cs, -e.sgn * L * scale);
					for (int gi = cs; gi <= ce; gi++)
					{
						const P col = static_cast<P>(gi - ix0);
						const P part = int_clamp01(xra - col, xrb - col, L, e.inv2s);
						add_many(rowbase + gi, e.sgn * part * scale);
					}
				}
			}
			return;
		}

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
				P cov = e0.sgn * int_clamp01(xra0 - col, xrb0 - col, L0, e0.inv2s)
					  + e1.sgn * int_clamp01(xra1 - col, xrb1 - col, L1, e1.inv2s)
					  + e2.sgn * int_clamp01(xra2 - col, xrb2 - col, L2, e2.inv2s);
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
}
