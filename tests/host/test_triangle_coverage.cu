/******************************************************************************
Host-only test of tri_cov (current) against tri_cov_ref (frozen copy of the
coverage code at e3168bc), and of the perimeter path against the direct path.
One PASS/FAIL line per case (INFO lines give the worst error/tolerance ratio
per triangle category); exit 0 only if all pass.
******************************************************************************/
#include "triangle_coverage.cuh"
#include "tri_cov_ref.cuh"

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

static void report(const std::string& name, bool ok, const std::string& detail)
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
}

template <typename P> P eps_tol();
template <> double eps_tol<double>() { return 1e-12; }
template <> float eps_tol<float>() { return 1e-5f; }

template <typename P>
static P inv2s_of(P slope)
{
	return slope != 0 ? P(1) / (2 * std::fabs(slope)) : P(0);
}

/******************************************************************************
1D integrals
******************************************************************************/
template <typename P, bool CLAMP>
static void test_integral(const char* name, const char* ptag)
{
	std::mt19937_64 gen(12345);
	std::uniform_real_distribution<double> da(-3, 3), dL(1e-9, 1), ds(-50, 50);
	double worst = 0; long bad = 0; std::string first;
	for (int i = 0; i < 1000000; i++)
	{
		P a = static_cast<P>(da(gen));
		P L = static_cast<P>(dL(gen));
		double sd = ds(gen);
		int k = i % 10;
		if (k == 0) sd = 0;
		else if (k == 1) sd = 1e-6;
		else if (k == 2) sd = -1e-6;
		P slope = static_cast<P>(sd);
		P b = a + L * slope;
		P ref, got;
		if (CLAMP)
		{
			ref = tri_cov_ref::int_clamp01<P>(a, b, L);
			got = tri_cov::int_clamp01<P>(a, b, L, inv2s_of(slope));
		}
		else
		{
			ref = tri_cov_ref::int_max0<P>(a, b, L);
			got = tri_cov::int_max0<P>(a, b, L, inv2s_of(slope));
		}
		double scale = std::fmax(1.0, std::fabs((double)ref));
		double err = std::fabs((double)got - (double)ref) / scale;
		if (!(err <= eps_tol<P>()))
		{
			if (bad == 0)
			{
				char buf[200];
				std::snprintf(buf, sizeof buf, "a=%.9g b=%.9g L=%.9g slope=%.9g ref=%.9g got=%.9g", (double)a, (double)b, (double)L, (double)slope, (double)ref, (double)got);
				first = buf;
			}
			bad++;
		}
		if (!(err <= worst)) worst = err;
	}
	char buf[400];
	std::snprintf(buf, sizeof buf, "%ld of 1e6 outside tolerance (worst %.3g); first: %s", bad, worst, first.c_str());
	report(std::string(name) + "_" + ptag, bad == 0, buf);
}

/******************************************************************************
an edge with a subnormal slope: 1 / (2 |slope|) overflows on the host, so
inv2s must still be finite and the integrals must not become NaN
******************************************************************************/
template <typename P>
static void test_subnormal_slope(const char* ptag)
{
	const P tiny = std::numeric_limits<P>::denorm_min();
	const tri_cov::Edge<P> e = tri_cov::make_edge<P>(P(0), P(0), tiny, P(1));
	const P both_nonpos = tri_cov::int_max0<P>(P(-0.25), P(-0.25) + e.slope, P(1), e.inv2s);
	const P mixed = tri_cov::int_max0<P>(P(0), e.slope, P(1), e.inv2s);
	const P clamped = tri_cov::int_clamp01<P>(P(0.5), P(0.5) + e.slope, P(1), e.inv2s);
	const bool ok = e.slope > 0 && std::isfinite(e.inv2s) && both_nonpos == 0
		&& std::isfinite(mixed) && std::fabs(mixed) <= tiny && clamped == P(0.5);
	char buf[300];
	std::snprintf(buf, sizeof buf, "slope=%.3g inv2s=%.3g int_max0(both<=0)=%.3g int_max0(mixed)=%.3g int_clamp01=%.9g",
		(double)e.slope, (double)e.inv2s, (double)both_nonpos, (double)mixed, (double)clamped);
	report(std::string("inv2s_finite_for_subnormal_slope_") + ptag, ok, buf);
}

/******************************************************************************
triangle sets
******************************************************************************/
enum { CAT_RANDOM, CAT_MODERATE, CAT_SLIVER, CAT_AXIS, CAT_CROSSING, CAT_COVERING, NCAT };
static const char* const cat_names[NCAT] = {"random", "moderate", "sliver", "axis-aligned", "crossing-edge", "covering"};

template <typename P>
struct Tri { Complex<P> p0, p1, p2; int cat; };

/******************************************************************************
triangles made for a 64 x 48 map, scaled by (sx, sy) to other maps; both
orientations of each
******************************************************************************/
template <typename P>
static std::vector<Tri<P>> triangle_set(double sx, double sy)
{
	std::mt19937_64 gen(777);
	std::uniform_real_distribution<double> dx(-10, 74), dy(-10, 58), d01(0, 1), dsm(-1, 1);
	std::vector<Tri<P>> tris;
	int cat = CAT_RANDOM;
	auto push = [&](double x0, double y0, double x1, double y1, double x2, double y2)
	{
		tris.push_back({Complex<P>(P(sx * x0), P(sy * y0)), Complex<P>(P(sx * x1), P(sy * y1)), Complex<P>(P(sx * x2), P(sy * y2)), cat});
	};
	//random
	cat = CAT_RANDOM;
	for (int i = 0; i < 3000; i++) push(dx(gen), dy(gen), dx(gen), dy(gen), dx(gen), dy(gen));
	//random moderate-size triangles (a few pixels)
	cat = CAT_MODERATE;
	for (int i = 0; i < 3000; i++)
	{
		double cx = 64 * d01(gen), cy = 48 * d01(gen);
		push(cx + 3 * dsm(gen), cy + 3 * dsm(gen), cx + 3 * dsm(gen), cy + 3 * dsm(gen), cx + 3 * dsm(gen), cy + 3 * dsm(gen));
	}
	//slivers, area ~1e-6
	cat = CAT_SLIVER;
	for (int i = 0; i < 500; i++)
	{
		double x0 = 64 * d01(gen), y0 = 48 * d01(gen), x1 = 64 * d01(gen), y1 = 48 * d01(gen);
		double len = std::hypot(x1 - x0, y1 - y0);
		double t = d01(gen);
		double h = 2e-6 / len; //area = 0.5 * len * h = 1e-6
		double nx = -(y1 - y0) / len, ny = (x1 - x0) / len;
		push(x0, y0, x1, y1, x0 + t * (x1 - x0) + h * nx, y0 + t * (y1 - y0) + h * ny);
	}
	//axis-aligned right triangles, integer and half-integer vertices
	cat = CAT_AXIS;
	for (int i = 0; i < 1000; i++)
	{
		double x = std::floor(2 * 64 * d01(gen)) / 2, y = std::floor(2 * 48 * d01(gen)) / 2;
		double w = (1 + std::floor(2 * 20 * d01(gen))) / 2, h = (1 + std::floor(2 * 20 * d01(gen))) / 2;
		if (i % 4 == 0) push(x, y, x + w, y, x, y + h);
		else if (i % 4 == 1) push(x, y, x + w, y, x + w, y + h);
		else if (i % 4 == 2) push(x, y + h, x + w, y, x + w, y + h);
		else push(x, y, x + w, y + h, x, y + h);
	}
	//crossing each map edge
	cat = CAT_CROSSING;
	push(-5, 10, 5, 5, 3, 20);
	push(60, 10, 70, 15, 62, 25);
	push(10, -5, 20, 3, 30, -2);
	push(10, 40, 20, 52, 30, 45);
	push(-5, -5, 5, 2, 2, 5);
	push(60, 44, 70, 55, 66, 40);
	push(-5, 20, 70, 25, 30, 30);
	//covering the whole map
	cat = CAT_COVERING;
	push(-100, -100, 300, -100, -100, 300);
	push(-1, -1, 200, 0, 0, 200);
	//both orientations
	const size_t n0 = tris.size();
	for (size_t i = 0; i < n0; i++) tris.push_back({tris[i].p0, tris[i].p2, tris[i].p1, tris[i].cat});
	return tris;
}

template <typename P>
static P shoelace(const Tri<P>& tr)
{
	return P(0.5) * ((tr.p1.re - tr.p0.re) * (tr.p2.im - tr.p0.im) - (tr.p2.re - tr.p0.re) * (tr.p1.im - tr.p0.im));
}

//width + height of the triangle's (unclamped) bounding box, in pixels
template <typename P>
static double bbox_wh(const Tri<P>& tr)
{
	double xmin = std::fmin(tr.p0.re, std::fmin(tr.p1.re, tr.p2.re)), xmax = std::fmax(tr.p0.re, std::fmax(tr.p1.re, tr.p2.re));
	double ymin = std::fmin(tr.p0.im, std::fmin(tr.p1.im, tr.p2.im)), ymax = std::fmax(tr.p0.im, std::fmax(tr.p1.im, tr.p2.im));
	return (xmax - xmin) + (ymax - ymin);
}

/******************************************************************************
conditioning floor: per pixel the coverage is a sum of O(1) edge integrals that
cancel down to ~area, then scaled by factor / area, so any change of rounding
path moves a pixel by up to ~eps * factor * (bbox width + bbox height) / area.
this exceeds the plan's thresholds only for slivers (area ~1e-6), thin
triangles and, in float, triangles of less than half a pixel; for
well-conditioned triangles it is below them.
******************************************************************************/
template <typename P>
static double conditioning_floor(const Tri<P>& tr, double factor)
{
	long double ex = (long double)tr.p1.re - tr.p0.re, ey = (long double)tr.p1.im - tr.p0.im;
	long double fx = (long double)tr.p2.re - tr.p0.re, fy = (long double)tr.p2.im - tr.p0.im;
	double ar = (double)std::fabs(0.5L * (ex * fy - fx * ey));
	double eps = sizeof(P) == 8 ? DBL_EPSILON : (double)FLT_EPSILON;
	return 4.0 * eps * factor * bbox_wh(tr) / ar;
}

/******************************************************************************
direct path
******************************************************************************/
template <typename P>
static void test_direct(const char* ptag)
{
	const Complex<int> npix(64, 48);
	const std::vector<Tri<P>> tris = triangle_set<P>(1, 1);

	const P factor = P(1.7);
	const size_t N = size_t(npix.re) * npix.im;
	double worst = 0; long bad = 0, n_steps = 0; std::string first;
	std::vector<P> got(N), ref(N);
	for (size_t t = 0; t < tris.size(); t++)
	{
		std::fill(got.begin(), got.end(), P(0));
		std::fill(ref.begin(), ref.end(), P(0));
		const Tri<P>& tr = tris[t];
		P area = shoelace(tr);
		auto add_g = [&](long long i, P v) { got[i] += v; };
		auto add_r = [&](long long i, P v) { ref[i] += v; };
		auto no_step = [&](long long, P) { n_steps++; };
		tri_cov::allocate_triangle<P, P>(tr.p0, tr.p1, tr.p2, area, factor, npix, LLONG_MAX, add_g, add_g, no_step);
		tri_cov_ref::allocate_triangle<P, P>(tr.p0, tr.p1, tr.p2, area, factor, npix, add_r, add_r);
		double tol;
		if (sizeof(P) == 8) tol = 1e-12 * factor;
		else tol = 8.0 * FLT_EPSILON * factor * bbox_wh(tr);
		/******************************************************************************
		measured worst ratio of the observed difference to the conditioning floor is ~1
		for both precisions
		******************************************************************************/
		tol = std::fmax(tol, conditioning_floor(tr, factor));
		for (size_t i = 0; i < N; i++)
		{
			double err = std::fabs((double)got[i] - (double)ref[i]);
			if (!(err <= tol))
			{
				if (bad == 0)
				{
					char buf[300];
					std::snprintf(buf, sizeof buf, "tri %zu pixel %zu got=%.9g ref=%.9g tol=%.3g", t, i, (double)got[i], (double)ref[i], tol);
					first = buf;
				}
				bad++;
			}
			if (!(err / tol <= worst)) worst = err / tol;
		}
	}
	char buf[400];
	std::snprintf(buf, sizeof buf, "%ld pixels outside tolerance over %zu triangles (worst err/tol %.3g), %ld add_step calls; first: %s",
		bad, tris.size(), worst, n_steps, first.c_str());
	report(std::string("direct_path_matches_reference_") + ptag, bad == 0 && n_steps == 0, buf);
}

/******************************************************************************
perimeter path: deposit into map E and step buffer S, then add the per-row
inclusive prefix sum of S into E
******************************************************************************/
template <typename P>
static void deposit_perimeter(const Tri<P>& tr, P factor, Complex<int> npix, long long edge_min_bbox,
	std::vector<P>& E, std::vector<P>& S, long& n_bad_index, long& n_steps)
{
	const long long N = 1LL * npix.re * npix.im;
	auto add = [&](long long i, P v) { if (i < 0 || i >= N) n_bad_index++; else E[i] += v; };
	auto step = [&](long long i, P v) { n_steps++; if (i < 0 || i >= N) n_bad_index++; else S[i] += v; };
	tri_cov::allocate_triangle<P, P>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, edge_min_bbox, add, add, step);
}

template <typename P>
static void add_row_prefix(std::vector<P>& E, const std::vector<P>& S, Complex<int> npix)
{
	for (int iy = 0; iy < npix.im; iy++)
	{
		P run = 0;
		for (int ix = 0; ix < npix.re; ix++)
		{
			const size_t k = size_t(npix.re) * iy + ix;
			run += S[k];
			E[k] += run;
		}
	}
}

template <typename P>
static void deposit_direct(const Tri<P>& tr, P factor, Complex<int> npix, std::vector<P>& D, long& n_steps)
{
	auto add = [&](long long i, P v) { D[i] += v; };
	auto step = [&](long long, P) { n_steps++; };
	tri_cov::allocate_triangle<P, P>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, LLONG_MAX, add, add, step);
}

/******************************************************************************
tolerances of the perimeter-vs-direct comparison. the plan's thresholds, raised
to rounding floors that scale with factor / area:

- per pixel: the conditioning floor (as for the direct path). it is binding
  only for ill-conditioned triangles, those whose floor exceeds the plan's
  per-pixel threshold (slivers, thin triangles and, in float, triangles of less
  than half a pixel)
- per triangle sum: in a row without a vertex the two edges have L = 1 and
  their steps cancel exactly; in one of the (at most 3) rows that hold a
  vertex they cancel only up to the rounding of L * scale, and that residue
  (~eps * factor / area) lands on every column right of the triangle. so
  4 * eps * factor / area per map pixel right of the triangle's bounding box
  in 3 rows (fewer if it overlaps fewer). for an ill-conditioned triangle, also
  the per-pixel floor summed over all the rows it overlaps

all failures of the plan's thresholds are explained by these floors; the
measured worst ratios are printed per category (INFO lines)
******************************************************************************/
template <typename P>
struct PerimTol { double plan_px, tol_px, plan_sum, tol_sum; };

template <typename P>
static PerimTol<P> perimeter_tolerances(const Tri<P>& tr, P factor, Complex<int> npix)
{
	const double eps = sizeof(P) == 8 ? DBL_EPSILON : (double)FLT_EPSILON;
	PerimTol<P> t;
	t.plan_px = sizeof(P) == 8 ? 1e-12 * factor : 8.0 * FLT_EPSILON * factor * bbox_wh(tr);
	const double floor_px = conditioning_floor(tr, factor);
	t.tol_px = std::fmax(t.plan_px, floor_px);
	t.plan_sum = (sizeof(P) == 8 ? 1e-12 : 1e-6) * factor;

	long double ex = (long double)tr.p1.re - tr.p0.re, ey = (long double)tr.p1.im - tr.p0.im;
	long double fx = (long double)tr.p2.re - tr.p0.re, fy = (long double)tr.p2.im - tr.p0.im;
	const double area = (double)std::fabs(0.5L * (ex * fy - fx * ey));
	const double xmax = std::fmax(tr.p0.re, std::fmax(tr.p1.re, tr.p2.re));
	const double ymin = std::fmin(tr.p0.im, std::fmin(tr.p1.im, tr.p2.im)), ymax = std::fmax(tr.p0.im, std::fmax(tr.p1.im, tr.p2.im));
	const double r0 = std::fmax(std::floor(ymin), 0.0), r1 = std::fmin(std::ceil(ymax) - 1, npix.im - 1.0);
	const double rows = r1 >= r0 ? r1 - r0 + 1 : 0;
	const double tail_cols = std::fmin(std::fmax(npix.re - 1 - std::floor(xmax), 0.0), (double)npix.re);
	double tol_sum = std::fmax(t.plan_sum, 4.0 * eps * factor / area * std::fmin(rows, 3.0) * tail_cols);
	if (floor_px > t.plan_px) tol_sum = std::fmax(tol_sum, floor_px * rows * npix.re);
	t.tol_sum = tol_sum;
	return t;
}

template <typename P>
static void test_perimeter(const char* ptag)
{
	const Complex<int> grids[3] = {Complex<int>(64, 48), Complex<int>(7, 3), Complex<int>(1, 1)};
	const P factor = P(1.7);
	//worst ratios per category: err / plan threshold, err / tolerance used
	double wpx_plan[NCAT] = {}, wpx_tol[NCAT] = {}, wsum_plan[NCAT] = {}, wsum_tol[NCAT] = {};
	//triangles per category, and those where a floor is above the plan threshold
	long ncat[NCAT] = {}, nfl_px[NCAT] = {}, nfl_sum[NCAT] = {};
	long bad_px = 0, bad_sum = 0, n_bad_index = 0, n_direct_steps = 0, n_perim_steps = 0, ntri = 0;
	std::string first_px, first_sum;
	for (const Complex<int>& g : grids)
	{
		const std::vector<Tri<P>> tris = triangle_set<P>(g.re / 64.0, g.im / 48.0);
		const size_t N = size_t(g.re) * g.im;
		std::vector<P> E(N), S(N), D(N);
		for (size_t t = 0; t < tris.size(); t++, ntri++)
		{
			const Tri<P>& tr = tris[t];
			const int c = tr.cat;
			std::fill(E.begin(), E.end(), P(0));
			std::fill(S.begin(), S.end(), P(0));
			std::fill(D.begin(), D.end(), P(0));
			deposit_perimeter(tr, factor, g, 0, E, S, n_bad_index, n_perim_steps);
			add_row_prefix(E, S, g);
			deposit_direct(tr, factor, g, D, n_direct_steps);

			const PerimTol<P> tol = perimeter_tolerances(tr, factor, g);
			ncat[c]++;
			if (tol.tol_px > tol.plan_px) nfl_px[c]++;
			if (tol.tol_sum > tol.plan_sum) nfl_sum[c]++;
			long double sE = 0, sD = 0;
			for (size_t i = 0; i < N; i++)
			{
				sE += E[i];
				sD += D[i];
				const double err = std::fabs((double)E[i] - (double)D[i]);
				if (!(err <= tol.tol_px))
				{
					if (bad_px == 0)
					{
						char buf[300];
						std::snprintf(buf, sizeof buf, "grid %dx%d %s tri %zu pixel %zu perimeter=%.9g direct=%.9g tol=%.3g",
							g.re, g.im, cat_names[c], t, i, (double)E[i], (double)D[i], tol.tol_px);
						first_px = buf;
					}
					bad_px++;
				}
				if (!(err / tol.plan_px <= wpx_plan[c])) wpx_plan[c] = err / tol.plan_px;
				if (!(err / tol.tol_px <= wpx_tol[c])) wpx_tol[c] = err / tol.tol_px;
			}
			const double err_sum = (double)std::fabs(sE - sD);
			if (!(err_sum <= tol.tol_sum))
			{
				if (bad_sum == 0)
				{
					char buf[300];
					std::snprintf(buf, sizeof buf, "grid %dx%d %s tri %zu sum perimeter=%.12g direct=%.12g tol=%.3g",
						g.re, g.im, cat_names[c], t, (double)sE, (double)sD, tol.tol_sum);
					first_sum = buf;
				}
				bad_sum++;
			}
			if (!(err_sum / tol.plan_sum <= wsum_plan[c])) wsum_plan[c] = err_sum / tol.plan_sum;
			if (!(err_sum / tol.tol_sum <= wsum_tol[c])) wsum_tol[c] = err_sum / tol.tol_sum;
		}
	}
	for (int c = 0; c < NCAT; c++)
	{
		std::printf("INFO perimeter_vs_direct_%s %-13s %5ld triangles | pixel: worst err/plan %.3g, err/tol %.3g, floor above plan for %ld"
			" | sum: worst err/plan %.3g, err/tol %.3g, floor above plan for %ld\n",
			ptag, cat_names[c], ncat[c], wpx_plan[c], wpx_tol[c], nfl_px[c], wsum_plan[c], wsum_tol[c], nfl_sum[c]);
	}
	char buf[500];
	std::snprintf(buf, sizeof buf, "%ld pixels outside tolerance over %ld triangles, %ld out-of-map indices, %ld add_step calls on the direct path; first: %s",
		bad_px, ntri, n_bad_index, n_direct_steps, first_px.c_str());
	report(std::string("perimeter_path_matches_direct_") + ptag, bad_px == 0 && n_bad_index == 0 && n_direct_steps == 0 && n_perim_steps > 0, buf);
	std::snprintf(buf, sizeof buf, "%ld of %ld triangles outside tolerance; first: %s", bad_sum, ntri, first_sum.c_str());
	report(std::string("perimeter_path_conserves_area_") + ptag, bad_sum == 0, buf);
}

/******************************************************************************
threshold: the clamped bounding box of this triangle is 9 x 9 = 81 pixels
(columns 10..18, rows 20..28)
******************************************************************************/
template <typename P>
static void test_threshold(const char* ptag)
{
	const Complex<int> npix(64, 48);
	const P factor = P(1.7);
	const Tri<P> tris[2] = {
		{Complex<P>(P(10.25), P(20.25)), Complex<P>(P(18.75), P(20.5)), Complex<P>(P(12.5), P(28.75)), 0},
		{Complex<P>(P(10.25), P(20.25)), Complex<P>(P(12.5), P(28.75)), Complex<P>(P(18.75), P(20.5)), 0}};
	const size_t N = size_t(npix.re) * npix.im;
	std::vector<P> E(N), S(N);
	long n_bad_index = 0, steps81 = 0, steps80 = 0;
	for (const Tri<P>& tr : tris)
	{
		deposit_perimeter(tr, factor, npix, 81, E, S, n_bad_index, steps81);
		deposit_perimeter(tr, factor, npix, 80, E, S, n_bad_index, steps80);
	}
	char buf[200];
	std::snprintf(buf, sizeof buf, "add_step calls: %ld with edge_min_bbox = 81 (want 0), %ld with 80 (want >= 2)", steps81, steps80);
	report(std::string("threshold_selects_path_") + ptag, steps81 == 0 && steps80 >= 2 && n_bad_index == 0, buf);
}

/******************************************************************************
steps outside the map
******************************************************************************/
template <typename P>
static void test_outside(const char* ptag)
{
	const Complex<int> npix(64, 48);
	const P factor = P(1.7);
	const size_t N = size_t(npix.re) * npix.im;
	bool ok = true;
	std::string detail;
	char buf[300];

	//entirely left of column 0 in x, overlapping the map in y: nothing at all
	{
		const Tri<P> tr = {Complex<P>(P(-10.5), P(5.25)), Complex<P>(P(-0.5), P(9.75)), Complex<P>(P(-6.25), P(30.5)), 0};
		long calls = 0;
		auto count = [&](long long, P) { calls++; };
		tri_cov::allocate_triangle<P, P>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, 0, count, count, count);
		if (calls != 0) { ok = false; std::snprintf(buf, sizeof buf, "left of the map: %ld callback calls (want 0); ", calls); detail += buf; }
	}

	/******************************************************************************
	the left edge, from (-5.5, 10.25) to (3.25, 30.75), is at x < 0 for y < 23.14,
	so in rows 10..23 an edge-row span starts at a negative column and its step
	must land on column 0 of that row. a second triangle hangs over the right
	edge of the map; its edge from (80.25, 12.5) to (70.25, 30.75) lies entirely
	right of the map and gives no step, and no step may wrap to column 0 of the
	next row (all its steps are at columns >= 60)
	******************************************************************************/
	const Tri<P> left = {Complex<P>(P(-5.5), P(10.25)), Complex<P>(P(20.25), P(12.5)), Complex<P>(P(3.25), P(30.75)), 0};
	const Tri<P> right = {Complex<P>(P(60.5), P(10.25)), Complex<P>(P(80.25), P(12.5)), Complex<P>(P(70.25), P(30.75)), 0};
	for (int side = 0; side < 2; side++)
	{
		for (int orient = 0; orient < 2; orient++)
		{
			Tri<P> tr = side == 0 ? left : right;
			if (orient) std::swap(tr.p1, tr.p2);
			std::vector<long long> step_idx;
			std::vector<P> E(N, P(0)), S(N, P(0)), D(N, P(0));
			long n_bad_index = 0, n_steps = 0, n_direct_steps = 0;
			auto add = [&](long long i, P v) { if (i < 0 || i >= (long long)N) n_bad_index++; else E[i] += v; };
			auto step = [&](long long i, P v) { n_steps++; step_idx.push_back(i); if (i < 0 || i >= (long long)N) n_bad_index++; else S[i] += v; };
			tri_cov::allocate_triangle<P, P>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, 0, add, add, step);
			add_row_prefix(E, S, npix);
			deposit_direct(tr, factor, npix, D, n_direct_steps);

			if (n_bad_index != 0) { ok = false; std::snprintf(buf, sizeof buf, "side %d orient %d: %ld out-of-map indices; ", side, orient, n_bad_index); detail += buf; }
			if (side == 0)
			{
				for (int r = 10; r <= 23; r++)
				{
					bool found = false;
					for (long long i : step_idx) found = found || i == 1LL * npix.re * r;
					if (!found) { ok = false; std::snprintf(buf, sizeof buf, "left orient %d: no step on column 0 of row %d; ", orient, r); detail += buf; }
				}
			}
			else
			{
				for (long long i : step_idx)
				{
					if (i % npix.re < 60 || i / npix.re < 10 || i / npix.re > 30)
					{
						ok = false;
						std::snprintf(buf, sizeof buf, "right orient %d: step at row %lld column %lld; ", orient, i / npix.re, i % npix.re);
						detail += buf;
					}
				}
			}
			double worst = 0;
			for (size_t i = 0; i < N; i++) worst = std::fmax(worst, std::fabs((double)E[i] - (double)D[i]));
			const double tol = sizeof(P) == 8 ? 1e-12 * factor : 8.0 * FLT_EPSILON * factor * bbox_wh(tr);
			if (!(worst <= tol)) { ok = false; std::snprintf(buf, sizeof buf, "side %d orient %d: worst |perimeter - direct| %.3g > %.3g; ", side, orient, worst, tol); detail += buf; }
		}
	}
	report(std::string("steps_outside_map_are_dropped_") + ptag, ok, detail);
}

int main()
{
	test_integral<double, false>("int_max0_matches_reference", "double");
	test_integral<float, false>("int_max0_matches_reference", "float");
	test_integral<double, true>("int_clamp01_matches_reference", "double");
	test_integral<float, true>("int_clamp01_matches_reference", "float");
	test_subnormal_slope<double>("double");
	test_subnormal_slope<float>("float");
	test_direct<double>("double");
	test_direct<float>("float");
	test_perimeter<double>("double");
	test_perimeter<float>("float");
	test_threshold<double>("double");
	test_threshold<float>("float");
	test_outside<double>("double");
	test_outside<float>("float");
	return n_fail == 0 ? 0 : 1;
}
