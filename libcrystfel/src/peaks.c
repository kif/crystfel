/*
 * peaks.c
 *
 * Peak search and other image analysis
 *
 * Copyright © 2012 Deutsches Elektronen-Synchrotron DESY,
 *                  a research centre of the Helmholtz Association.
 * Copyright © 2012 Richard Kirian
 *
 * Authors:
 *   2010-2012 Thomas White <taw@physics.org>
 *   2012      Kenneth Beyerlein <kenneth.beyerlein@desy.de>
 *   2011      Andrew Martin <andrew.martin@desy.de>
 *   2011      Richard Kirian
 *
 * This file is part of CrystFEL.
 *
 * CrystFEL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CrystFEL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CrystFEL.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <gsl/gsl_statistics_int.h>
#include <pthread.h>
#include <fenv.h>

#include "image.h"
#include "utils.h"
#include "peaks.h"
#include "detector.h"
#include "filters.h"
#include "reflist-utils.h"
#include "beam-parameters.h"
#include "cell-utils.h"


/* Degree of polarisation of X-ray beam */
#define POL (1.0)

static int cull_peaks_in_panel(struct image *image, struct panel *p)
{
	int i, n;
	int nelim = 0;

	n = image_feature_count(image->features);

	for ( i=0; i<n; i++ ) {

		struct imagefeature *f;
		int j, ncol;

		f = image_get_feature(image->features, i);
		if ( f == NULL ) continue;

		if ( f->fs < p->min_fs ) continue;
		if ( f->fs > p->max_fs ) continue;
		if ( f->ss < p->min_ss ) continue;
		if ( f->ss > p->max_ss ) continue;

		/* How many peaks are in the same column? */
		ncol = 0;
		for ( j=0; j<n; j++ ) {

			struct imagefeature *g;

			if ( i==j ) continue;

			g = image_get_feature(image->features, j);
			if ( g == NULL ) continue;

			if ( p->badrow == 'f' ) {
				if ( fabs(f->ss - g->ss) < 2.0 ) ncol++;
			} else if ( p->badrow == 's' ) {
				if ( fabs(f->fs - g->fs) < 2.0 ) ncol++;
			} /* else do nothing */

		}

		/* More than three? */
		if ( ncol <= 3 ) continue;

		/* Yes?  Delete them all... */
		for ( j=0; j<n; j++ ) {
			struct imagefeature *g;
			g = image_get_feature(image->features, j);
			if ( g == NULL ) continue;
			if ( p->badrow == 'f' ) {
				if ( fabs(f->ss - g->ss) < 2.0 ) {
					image_remove_feature(image->features,
					                     j);
					nelim++;
				}
			} else if ( p->badrow == 's' ) {
				if ( fabs(f->fs - g->ss) < 2.0 ) {
					image_remove_feature(image->features,
					                     j);
					nelim++;
				}
			} else {
				ERROR("Invalid badrow direction.\n");
				abort();
			}

		}

	}

	return nelim;
}


/* Post-processing of the peak list to remove noise */
static int cull_peaks(struct image *image)
{
	int nelim = 0;
	struct panel *p;
	int i;

	for ( i=0; i<image->det->n_panels; i++ ) {
		p = &image->det->panels[i];
		if ( p->badrow != '-' ) {
			nelim += cull_peaks_in_panel(image, p);
		}
	}

	return nelim;
}


/* cfs, css relative to panel origin */
static int *make_BgMask(struct image *image, struct panel *p,
                        double ir_out, double ir_inn)
{
	Reflection *refl;
	RefListIterator *iter;
	int *mask;
	int w, h;

	w = p->max_fs - p->min_fs + 1;
	h = p->max_ss - p->min_ss + 1;
	mask = calloc(w*h, sizeof(int));
	if ( mask == NULL ) return NULL;

	if ( image->reflections == NULL ) return mask;

	/* Loop over all reflections */
	for ( refl = first_refl(image->reflections, &iter);
	      refl != NULL;
	      refl = next_refl(refl, iter) )
	{
		struct panel *p2;
		double pk2_fs, pk2_ss;
		signed int dfs, dss;
		double pk2_cfs, pk2_css;

		get_detector_pos(refl, &pk2_fs, &pk2_ss);

		/* Determine if reflection is in the same panel */
		p2 = find_panel(image->det, pk2_fs, pk2_ss);
		if ( p2 != p ) continue;

		pk2_cfs = pk2_fs - p->min_fs;
		pk2_css = pk2_ss - p->min_ss;

		for ( dfs=-ir_inn; dfs<=ir_inn; dfs++ ) {
		for ( dss=-ir_inn; dss<=ir_inn; dss++ ) {

			signed int fs, ss;

			/* In peak region for this peak? */
			if ( dfs*dfs + dss*dss > ir_inn*ir_inn ) continue;

			fs = pk2_cfs + dfs;
			ss = pk2_css + dss;

			/* On panel? */
			if ( fs >= w ) continue;
			if ( ss >= h ) continue;
			if ( fs < 0 ) continue;
			if ( ss < 0 ) continue;

			mask[fs + ss*w] = 1;

		}
		}

	}

	return mask;
}


/* Returns non-zero if peak has been vetoed.
 * i.e. don't use result if return value is not zero. */
static int integrate_peak(struct image *image, int cfs, int css,
                          double *pfs, double *pss,
                          double *intensity, double *sigma,
                          double ir_inn, double ir_mid, double ir_out,
                          int use_max_adu, int *bgPkMask, int *saturated)
{
	signed int dfs, dss;
	double lim_sq, out_lim_sq, mid_lim_sq;
	double pk_total;
	int pk_counts;
	double fsct, ssct;
	double bg_tot = 0.0;
	int bg_counts = 0;
	struct panel *p;
	double bg_mean, bg_var;
	double bg_tot_sq = 0.0;
	double var;
	double aduph;
	int p_cfs, p_css, p_w, p_h;

	p = find_panel(image->det, cfs, css);
	if ( p == NULL ) return 1;
	if ( p->no_index ) return 1;

	/* Determine regions where there is expected to be a peak */
	p_cfs = cfs - p->min_fs;
	p_css = css - p->min_ss;  /* Panel-relative coordinates */
	p_w = p->max_fs - p->min_fs + 1;
	p_h = p->max_ss - p->min_ss + 1;

	aduph = p->adu_per_eV * ph_lambda_to_eV(image->lambda);

	lim_sq = pow(ir_inn, 2.0);
	mid_lim_sq = pow(ir_mid, 2.0);
	out_lim_sq = pow(ir_out, 2.0);

	/* Estimate the background */
	for ( dfs=-ir_out; dfs<=+ir_out; dfs++ ) {
	for ( dss=-ir_out; dss<=+ir_out; dss++ ) {

		double val;
		uint16_t flags;
		int idx;

		/* Restrict to annulus */
		if ( dfs*dfs + dss*dss > out_lim_sq ) continue;
		if ( dfs*dfs + dss*dss < mid_lim_sq ) continue;

		/* Strayed off one panel? */
		if ( (p_cfs+dfs >= p_w) || (p_css+dss >= p_h)
		  || (p_cfs+dfs < 0 ) || (p_css+dss < 0) ) return 1;

		/* Wandered into a bad region? */
		if ( in_bad_region(image->det, p->min_fs+p_cfs+dfs,
		                               p->min_ss+p_css+dss) )
		{
			return 1;
		}

		/* Check if there is a peak in the background region */
		if ( (bgPkMask != NULL)
		  && bgPkMask[(p_cfs+dfs) + p_w*(p_css+dss)] ) continue;

		idx = dfs+cfs+image->width*(dss+css);

		/* Veto this peak if we tried to integrate in a bad region */
		if ( image->flags != NULL ) {

			flags = image->flags[idx];

			/* It must have all the "good" bits to be valid */
			if ( !((flags & image->det->mask_good)
			                   == image->det->mask_good) ) return 1;

			/* If it has any of the "bad" bits, reject */
			if ( flags & image->det->mask_bad ) return 1;

		}

		val = image->data[idx];

		/* Check if peak contains saturation in bg region */
		if ( use_max_adu && (val > p->max_adu) ) {
			if ( saturated != NULL ) *saturated = 1;
		}

		bg_tot += val;
		bg_tot_sq += pow(val, 2.0);
		bg_counts++;

	}
	}

	if ( bg_counts == 0 ) return 1;
	bg_mean = bg_tot / bg_counts;
	bg_var = (bg_tot_sq/bg_counts) - pow(bg_mean, 2.0);

	/* Measure the peak */
	pk_total = 0.0;
	pk_counts = 0;
	fsct = 0.0;  ssct = 0.0;
	for ( dfs=-ir_inn; dfs<=+ir_inn; dfs++ ) {
	for ( dss=-ir_inn; dss<=+ir_inn; dss++ ) {

		double val;
		uint16_t flags;
		int idx;

		/* Inner mask radius */
		if ( dfs*dfs + dss*dss > lim_sq ) continue;

		/* Strayed off one panel? */
		if ( (p_cfs+dfs >= p_w) || (p_css+dss >= p_h)
		  || (p_cfs+dfs < 0 ) || (p_css+dss < 0) ) return 1;

		/* Wandered into a bad region? */
		if ( in_bad_region(image->det, p->min_fs+p_cfs+dfs,
		                               p->min_ss+p_css+dss) )
		{
			return 1;
		}

		idx = dfs+cfs+image->width*(dss+css);

		/* Veto this peak if we tried to integrate in a bad region */
		if ( image->flags != NULL ) {

			flags = image->flags[idx];

			/* It must have all the "good" bits to be valid */
			if ( !((flags & image->det->mask_good)
			                   == image->det->mask_good) ) return 1;

			/* If it has any of the "bad" bits, reject */
			if ( flags & image->det->mask_bad ) return 1;

		}

		val = image->data[idx] - bg_mean;

		/* Check if peak contains saturation */
		if ( use_max_adu && (val > p->max_adu) ) {
			if ( saturated != NULL ) *saturated = 1;
		}

		pk_counts++;
		pk_total += val;

		fsct += val*(cfs+dfs);
		ssct += val*(css+dss);

	}
	}

	if ( pk_counts == 0 ) return 1;

	*pfs = ((double)fsct / pk_total) + 0.5;
	*pss = ((double)ssct / pk_total) + 0.5;

	var = pk_counts * bg_var;
	var += aduph * pk_total;
	if ( var < 0.0 ) return 1;

	if ( intensity != NULL ) *intensity = pk_total;
	if ( sigma != NULL ) *sigma = sqrt(var);

	return 0;
}


static void search_peaks_in_panel(struct image *image, float threshold,
                                  float min_gradient, float min_snr,
                                  struct panel *p,
                                  double ir_inn, double ir_mid, double ir_out)
{
	int fs, ss, stride;
	float *data;
	double d;
	int idx;
	double f_fs = 0.0;
	double f_ss = 0.0;
	double intensity = 0.0;
	double sigma = 0.0;
	int nrej_dis = 0;
	int nrej_pro = 0;
	int nrej_fra = 0;
	int nrej_bad = 0;
	int nrej_snr = 0;
	int nacc = 0;
	int ncull;

	data = image->data;
	stride = image->width;

	for ( fs = p->min_fs+1; fs <= p->max_fs-1; fs++ ) {
	for ( ss = p->min_ss+1; ss <= p->max_ss-1; ss++ ) {

		double dx1, dx2, dy1, dy2;
		double dxs, dys;
		double grad;
		int mask_fs, mask_ss;
		int s_fs, s_ss;
		double max;
		unsigned int did_something;
		int r;

		/* Overall threshold */
		if ( data[fs+stride*ss] < threshold ) continue;

		/* Immediate rejection of pixels above max_adu */
		if ( data[fs+stride*ss] > p->max_adu ) continue;

		/* Get gradients */
		dx1 = data[fs+stride*ss] - data[(fs+1)+stride*ss];
		dx2 = data[(fs-1)+stride*ss] - data[fs+stride*ss];
		dy1 = data[fs+stride*ss] - data[(fs+1)+stride*(ss+1)];
		dy2 = data[fs+stride*(ss-1)] - data[fs+stride*ss];

		/* Average gradient measurements from both sides */
		dxs = ((dx1*dx1) + (dx2*dx2)) / 2;
		dys = ((dy1*dy1) + (dy2*dy2)) / 2;

		/* Calculate overall gradient */
		grad = dxs + dys;

		if ( grad < min_gradient ) continue;

		mask_fs = fs;
		mask_ss = ss;

		do {

			max = data[mask_fs+stride*mask_ss];
			did_something = 0;

			for ( s_ss=biggest(mask_ss-ir_inn, p->min_ss);
			      s_ss<=smallest(mask_ss+ir_inn, p->max_ss);
			      s_ss++ )
			{
			for ( s_fs=biggest(mask_fs-ir_inn, p->min_fs);
			      s_fs<=smallest(mask_fs+ir_inn, p->max_fs);
			      s_fs++ )
			{

				if ( data[s_fs+stride*s_ss] > max ) {
					max = data[s_fs+stride*s_ss];
					mask_fs = s_fs;
					mask_ss = s_ss;
					did_something = 1;
				}

			}
			}

			/* Abort if drifted too far from the foot point */
			if ( distance(mask_fs, mask_ss, fs, ss) > ir_inn )
			{
				break;
			}

		} while ( did_something );

		/* Too far from foot point? */
		if ( distance(mask_fs, mask_ss, fs, ss) > ir_inn ) {
			nrej_dis++;
			continue;
		}

		/* Should be enforced by bounds used above.  Muppet check. */
		assert(mask_fs <= p->max_fs);
		assert(mask_ss <= p->max_ss);
		assert(mask_fs >= p->min_fs);
		assert(mask_ss >= p->min_ss);

		/* Centroid peak and get better coordinates. */
		r = integrate_peak(image, mask_fs, mask_ss,
		                   &f_fs, &f_ss, &intensity, &sigma,
		                   ir_inn, ir_mid, ir_out, 0, NULL, NULL);

		if ( r ) {
			/* Bad region - don't detect peak */
			nrej_bad++;
			continue;
		}

		/* It is possible for the centroid to fall outside the image */
		if ( (f_fs < p->min_fs) || (f_fs > p->max_fs)
		  || (f_ss < p->min_ss) || (f_ss > p->max_ss) ) {
			nrej_fra++;
			continue;
		}

		if ( fabs(intensity)/sigma < min_snr ) {
			nrej_snr++;
			continue;
		}

		/* Check for a nearby feature */
		image_feature_closest(image->features, f_fs, f_ss, &d, &idx);
		if ( d < 2.0*ir_inn ) {
			nrej_pro++;
			continue;
		}

		/* Add using "better" coordinates */
		image_add_feature(image->features, f_fs, f_ss, image, intensity,
		                  NULL);
		nacc++;

	}
	}

	if ( image->det != NULL ) {
		ncull = cull_peaks(image);
		nacc -= ncull;
	} else {
		STATUS("Not culling peaks because I don't have a "
		       "detector geometry file.\n");
		ncull = 0;
	}

//	STATUS("%i accepted, %i box, %i proximity, %i outside panel, "
//	       "%i in bad regions, %i with SNR < %g, %i badrow culled.\n",
//	       nacc, nrej_dis, nrej_pro, nrej_fra, nrej_bad,
//	       nrej_snr, min_snr, ncull);

	if ( ncull != 0 ) {
		STATUS("WARNING: %i peaks were badrow culled.  This feature"
		       " should not usually be used.\nConsider setting"
		       " badrow=- in the geometry file.\n", ncull);
	}
}


void search_peaks(struct image *image, float threshold, float min_gradient,
                  float min_snr, double ir_inn, double ir_mid, double ir_out)
{
	int i;

	if ( image->features != NULL ) {
		image_feature_list_free(image->features);
	}
	image->features = image_feature_list_new();

	for ( i=0; i<image->det->n_panels; i++ ) {

		struct panel *p = &image->det->panels[i];

		if ( p->no_index ) continue;
		search_peaks_in_panel(image, threshold, min_gradient,
		                      min_snr, p, ir_inn, ir_mid, ir_out);

	}
}


double peak_lattice_agreement(struct image *image, UnitCell *cell, double *pst)
{
	int i;
	int n_feat = 0;
	int n_sane = 0;
	double ax, ay, az;
	double bx, by, bz;
	double cx, cy, cz;
	double min_dist = 0.25;
	double stot = 0.0;

	/* Round towards nearest */
	fesetround(1);

	/* Cell basis vectors for this image */
	cell_get_cartesian(cell, &ax, &ay, &az, &bx, &by, &bz, &cx, &cy, &cz);

	/* Loop over peaks, checking proximity to nearest reflection */
	for ( i=0; i<image_feature_count(image->features); i++ ) {

		struct imagefeature *f;
		struct rvec q;
		double h,k,l,hd,kd,ld;

		/* Assume all image "features" are genuine peaks */
		f = image_get_feature(image->features, i);
		if ( f == NULL ) continue;
		n_feat++;

		/* Reciprocal space position of found peak */
		q = get_q(image, f->fs, f->ss, NULL, 1.0/image->lambda);

		/* Decimal and fractional Miller indices of nearest
		 * reciprocal lattice point */
		hd = q.u * ax + q.v * ay + q.w * az;
		kd = q.u * bx + q.v * by + q.w * bz;
		ld = q.u * cx + q.v * cy + q.w * cz;
		h = lrint(hd);
		k = lrint(kd);
		l = lrint(ld);

		/* Check distance */
		if ( (fabs(h - hd) < min_dist) && (fabs(k - kd) < min_dist)
		  && (fabs(l - ld) < min_dist) )
		{
			double sval;
			n_sane++;
			sval = pow(h-hd, 2.0) + pow(k-kd, 2.0) + pow(l-ld, 2.0);
			stot += 1.0 - sval;
			continue;
		}

	}

	*pst = stot;
	return (double)n_sane / (float)n_feat;
}


int peak_sanity_check(struct image *image)
{
	double stot;
	/* 0 means failed test, 1 means passed test */
	return peak_lattice_agreement(image, image->indexed_cell, &stot) >= 0.5;
}


struct integr_ind
{
	double res;
	Reflection *refl;
};


static int compare_resolution(const void *av, const void *bv)
{
	const struct integr_ind *a = av;
	const struct integr_ind *b = bv;

	return a->res > b->res;
}


static struct integr_ind *sort_reflections(RefList *list, UnitCell *cell,
                                           int *np)
{
	struct integr_ind *il;
	Reflection *refl;
	RefListIterator *iter;
	int i, n;

	n = num_reflections(list);
	*np = 0;  /* For now */

	if ( n == 0 ) return NULL;

	il = calloc(n, sizeof(struct integr_ind));
	if ( il == NULL ) return NULL;

	i = 0;
	for ( refl = first_refl(list, &iter);
	      refl != NULL;
	      refl = next_refl(refl, iter) )
	{
		signed int h, k, l;
		double res;

		get_indices(refl, &h, &k, &l);
		res = resolution(cell, h, k, l);

		il[i].res = res;
		il[i].refl = refl;

		i++;
		assert(i <= n);
	}

	qsort(il, n, sizeof(struct integr_ind), compare_resolution);

	*np = n;
	return il;
}


/* Integrate the list of predicted reflections in "image" */
void integrate_reflections(struct image *image, int use_closer, int bgsub,
                           double min_snr,
                           double ir_inn, double ir_mid, double ir_out,
                           int integrate_saturated)
{
	struct integr_ind *il;
	int n, i;
	double av = 0.0;
	int first = 1;
	int **bgMasks;

	il = sort_reflections(image->reflections, image->indexed_cell, &n);
	if ( il == NULL ) {
		ERROR("Couldn't sort reflections\n");
		return;
	}

	/* Make background masks for all panels */
	bgMasks = calloc(image->det->n_panels, sizeof(int *));
	if ( bgMasks == NULL ) {
		ERROR("Couldn't create list of background masks.\n");
		return;
	}
	for ( i=0; i<image->det->n_panels; i++ ) {
		int *mask;
		mask = make_BgMask(image, &image->det->panels[i],
		                   ir_out, ir_inn);
		if ( mask == NULL ) {
			ERROR("Couldn't create background mask.\n");
			return;
		}
		bgMasks[i] = mask;
	}

	for ( i=0; i<n; i++ ) {

		double fs, ss, intensity;
		double d;
		int idx;
		double sigma, snr;
		double pfs, pss;
		int r;
		Reflection *refl;
		signed int h, k, l;
		struct panel *p;
		int pnum, j, found;
		int saturated;

		refl = il[i].refl;

		get_detector_pos(refl, &pfs, &pss);
		get_indices(refl, &h, &k, &l);

		/* Is there a really close feature which was detected? */
		if ( use_closer ) {

			struct imagefeature *f;

			if ( image->features != NULL ) {
				f = image_feature_closest(image->features,
					                  pfs, pss, &d, &idx);
			} else {
				f = NULL;
			}

			/* FIXME: Horrible hardcoded value */
			if ( (f != NULL) && (d < 10.0) ) {

				double exe;

				exe = get_excitation_error(refl);

				pfs = f->fs;
				pss = f->ss;

				set_detector_pos(refl, exe, pfs, pss);

			}

		}

		p = find_panel(image->det, pfs, pss);
		if ( p == NULL ) continue;  /* Next peak */
		found = 0;
		for ( j=0; j<image->det->n_panels; j++ ) {
			if ( &image->det->panels[j] == p ) {
				pnum = j;
				found = 1;
				break;
			}
		}
		if ( !found ) {
			ERROR("Couldn't find panel %p in list.\n", p);
			return;
		}

		r = integrate_peak(image, pfs, pss, &fs, &ss,
		                   &intensity, &sigma, ir_inn, ir_mid, ir_out,
		                   1, bgMasks[pnum], &saturated);

		if ( saturated ) {
			image->n_saturated++;
			if ( !integrate_saturated ) r = 1;
		}

		/* I/sigma(I) cutoff */
		if ( intensity/sigma < min_snr ) r = 1;

		/* Record intensity and set redundancy to 1 on success */
		if ( r == 0 ) {
			set_intensity(refl, intensity);
			set_esd_intensity(refl, sigma);
			set_redundancy(refl, 1);
		} else {
			set_redundancy(refl, 0);
		}

		snr = intensity / sigma;
		if ( snr > 1.0 ) {
			if ( first ) {
				av = snr;
				first = 0;
			} else {
				av = av + 0.1*(snr - av);
			}
			//STATUS("%5.2f A, %5.2f, av %5.2f\n",
			//       1e10/il[i].res, snr, av);
			//if ( av < 1.0 ) break;
		}
	}

	for ( i=0; i<image->det->n_panels; i++ ) {
		free(bgMasks[i]);
	}
	free(bgMasks);

	image->diffracting_resolution = 0.0;

	free(il);
}


void validate_peaks(struct image *image, double min_snr,
                    int ir_inn, int ir_mid, int ir_out)
{
	int i, n;
	ImageFeatureList *flist;
	int n_wtf, n_int, n_dft, n_snr, n_prx;

	flist = image_feature_list_new();
	if ( flist == NULL ) return;

	n = image_feature_count(image->features);

	/* Loop over peaks, putting each one through the integrator */
	n_wtf = 0;  n_int = 0;  n_dft = 0;  n_snr = 0;  n_prx = 0;
	for ( i=0; i<n; i++ ) {

		struct imagefeature *f;
		int r;
		double d;
		int idx;
		double f_fs, f_ss;
		double intensity, sigma;
		struct panel *p;

		f = image_get_feature(image->features, i);
		if ( f == NULL ) {
			n_wtf++;
			continue;
		}

		p = find_panel(image->det, f->fs, f->ss);
		if ( p == NULL ) {
			n_wtf++;
			continue;
		}

		r = integrate_peak(image, f->fs, f->ss,
		                   &f_fs, &f_ss, &intensity, &sigma,
		                   ir_inn, ir_mid, ir_out, 0, NULL, NULL);
		if ( r ) {
			n_int++;
			continue;
		}

		/* It is possible for the centroid to fall outside the image */
		if ( (f_fs < p->min_fs) || (f_fs > p->max_fs)
		  || (f_ss < p->min_ss) || (f_ss > p->max_ss) )
		{
			n_dft++;
			continue;
		}

		if ( fabs(intensity)/sigma < min_snr ) {
			n_snr++;
			continue;
		}

		/* Check for a nearby feature */
		image_feature_closest(flist, f_fs, f_ss, &d, &idx);
		if ( d < 2.0*ir_inn ) {
			n_prx++;
			continue;
		}

		/* Add using "better" coordinates */
		image_add_feature(flist, f_fs, f_ss, image, intensity, NULL);

	}

	//STATUS("HDF5: %i peaks, validated: %i.  WTF: %i, integration: %i,"
	//       " drifted: %i, SNR: %i, proximity: %i\n",
	//       n, image_feature_count(flist),
	//       n_wtf, n_int, n_dft, n_snr, n_prx);
	image_feature_list_free(image->features);
	image->features = flist;
}
