/*
 * scaling.c
 *
 * Scaling
 *
 * Copyright © 2012-2017 Deutsches Elektronen-Synchrotron DESY,
 *                       a research centre of the Helmholtz Association.
 *
 * Authors:
 *   2010-2017 Thomas White <taw@physics.org>
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
#include <assert.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_fit.h>

#include "merge.h"
#include "post-refinement.h"
#include "symmetry.h"
#include "cell.h"
#include "cell-utils.h"
#include "scaling.h"


/* Maximum number of iterations of NLSq to do for each image per macrocycle. */
#define MAX_CYCLES (10)


/* Apply the given shift to the 'k'th parameter of 'image'. */
static void apply_shift(Crystal *cr, int k, double shift)
{
	double t;

	switch ( k ) {

		case GPARAM_BFAC :
		t = crystal_get_Bfac(cr);
		t += shift;
		crystal_set_Bfac(cr, t);
		break;

		case GPARAM_OSF :
		t = -log(crystal_get_osf(cr));
		t += shift;
		crystal_set_osf(cr, exp(-t));
		break;

		default :
		ERROR("No shift defined for parameter %i\n", k);
		abort();

	}
}


/* Perform one cycle of scaling of 'cr' against 'full' */
static double scale_iterate(Crystal *cr, const RefList *full,
                            PartialityModel pmodel, int *nr)
{
	gsl_matrix *M;
	gsl_vector *v;
	gsl_vector *shifts;
	int param;
	Reflection *refl;
	RefListIterator *iter;
	RefList *reflections;
	double max_shift;
	int nref = 0;
	int num_params = 0;
	enum gparam rv[32];
	double G, B;

	*nr = 0;

	rv[num_params++] = GPARAM_OSF;
	rv[num_params++] = GPARAM_BFAC;

	M = gsl_matrix_calloc(num_params, num_params);
	v = gsl_vector_calloc(num_params);

	reflections = crystal_get_reflections(cr);
	G = crystal_get_osf(cr);
	B = crystal_get_Bfac(cr);

	/* Scaling terms */
	for ( refl = first_refl(reflections, &iter);
	      refl != NULL;
	      refl = next_refl(refl, iter) )
	{
		signed int ha, ka, la;
		double I_full, delta_I, esd;
		double w;
		double I_partial;
		int k;
		double p, L, s;
		double fx;
		Reflection *match;
		double gradients[num_params];

		/* If reflection is free-flagged, don't use it here */
		if ( get_flag(refl) ) continue;

		/* Find the full version */
		get_indices(refl, &ha, &ka, &la);
		match = find_refl(full, ha, ka, la);
		if ( match == NULL ) continue;

		/* Merged intensitty */
		I_full = get_intensity(match);

		/* Actual measurement of this reflection from this pattern */
		I_partial = get_intensity(refl);
		esd = get_esd_intensity(refl);
		p = get_partiality(refl);

		/* Scale only using strong reflections */
		if ( I_partial <= 3.0*esd ) continue; /* Also because of log */
		if ( get_redundancy(match) < 2 ) continue;
		if ( I_full <= 0 ) continue;  /* Because log */
		if ( p <= 0.0 ) continue; /* Because of log */

		L = get_lorentz(refl);
		s = resolution(crystal_get_cell(cr), ha, ka, la);

		/* Calculate the weight for this reflection */
		w = 1.0;

		/* Calculate all gradients for this reflection */
		for ( k=0; k<num_params; k++ ) {

			if ( rv[k] == GPARAM_OSF ) {
				gradients[k] = 1.0;
			} else if ( rv[k] == GPARAM_BFAC ) {
				gradients[k] = -s*s;
			} else {
				ERROR("Unrecognised scaling gradient.\n");
				abort();
			}
		}

		for ( k=0; k<num_params; k++ ) {

			int g;
			double v_c, v_curr;

			for ( g=0; g<num_params; g++ ) {

				double M_c, M_curr;

				/* Matrix is symmetric */
				if ( g > k ) continue;

				M_c = w * gradients[g] * gradients[k];

				M_curr = gsl_matrix_get(M, k, g);
				gsl_matrix_set(M, k, g, M_curr + M_c);
				gsl_matrix_set(M, g, k, M_curr + M_c);

			}

			fx = -log(G) + log(p) - log(L) - B*s*s + log(I_full);
			delta_I = log(I_partial) - fx;
			v_c = w * delta_I * gradients[k];
			v_curr = gsl_vector_get(v, k);
			gsl_vector_set(v, k, v_curr + v_c);

		}

		nref++;
	}

	*nr = nref;

	if ( nref < num_params ) {
		crystal_set_user_flag(cr, PRFLAG_FEWREFL);
		gsl_matrix_free(M);
		gsl_vector_free(v);
		return 0.0;
	}

	max_shift = 0.0;
	shifts = solve_svd(v, M, NULL, 0);
	if ( shifts != NULL ) {

		for ( param=0; param<num_params; param++ ) {
			double shift = gsl_vector_get(shifts, param);
			apply_shift(cr, rv[param], shift);
			if ( fabs(shift) > max_shift ) max_shift = fabs(shift);
		}

	} else {
		crystal_set_user_flag(cr, PRFLAG_SOLVEFAIL);
	}

	gsl_matrix_free(M);
	gsl_vector_free(v);
	gsl_vector_free(shifts);

	return max_shift;
}


double log_residual(Crystal *cr, const RefList *full, int free,
                    int *pn_used, const char *filename)
{
	double dev = 0.0;
	double G, B;
	Reflection *refl;
	RefListIterator *iter;
	int n_used = 0;
	FILE *fh = NULL;

	G = crystal_get_osf(cr);
	B = crystal_get_Bfac(cr);
	if ( filename != NULL ) {
		fh = fopen(filename, "a");
		if ( fh == NULL ) {
			ERROR("Failed to open '%s'\n", filename);
		}
	}

	for ( refl = first_refl(crystal_get_reflections(cr), &iter);
	      refl != NULL;
	      refl = next_refl(refl, iter) )
	{
		double p, L, s, w;
		signed int h, k, l;
		Reflection *match;
		double esd, I_full, I_partial;
		double fx, dc;

		if ( free && !get_flag(refl) ) continue;

		get_indices(refl, &h, &k, &l);
		match = find_refl(full, h, k, l);
		if ( match == NULL ) continue;

		p = get_partiality(refl);
		L = get_lorentz(refl);
		I_partial = get_intensity(refl);
		I_full = get_intensity(match);
		esd = get_esd_intensity(refl);
		s = resolution(crystal_get_cell(cr), h, k, l);

		if ( I_partial <= 3.0*esd ) continue; /* Also because of log */
		if ( get_redundancy(match) < 2 ) continue;
		if ( I_full <= 0 ) continue;  /* Because log */
		if ( p <= 0.0 ) continue; /* Because of log */

		fx = -log(G) + log(p) - log(L) - B*s*s + log(I_full);
		dc = log(I_partial) - fx;
		w = 1.0;
		dev += w*dc*dc;

		if ( fh != NULL ) {
			fprintf(fh, "%4i %4i %4i %e %e\n",
			        h, k, l, s, dev);
		}

	}

	if ( fh != NULL ) fclose(fh);

	if ( pn_used != NULL ) *pn_used = n_used;
	return dev;
}


static void do_scale_refine(Crystal *cr, const RefList *full,
                            PartialityModel pmodel, int *nr)
{
	int i, done;
	double old_dev;

	old_dev = log_residual(cr, full, 0, NULL, NULL);

	i = 0;
	done = 0;
	do {

		double dev;

		scale_iterate(cr, full, pmodel, nr);

		dev = log_residual(cr, full, 0, 0, NULL);
		if ( fabs(dev - old_dev) < dev*0.01 ) done = 1;

		i++;
		old_dev = dev;

	} while ( i < MAX_CYCLES && !done );
}


struct scale_args
{
	RefList *full;
	Crystal *crystal;
	PartialityModel pmodel;
	int n_reflections;
};


struct queue_args
{
	int n_started;
	int n_done;
	Crystal **crystals;
	int n_crystals;
	long long int n_reflections;
	struct scale_args task_defaults;
};


static void scale_crystal(void *task, int id)
{
	struct scale_args *pargs = task;
	do_scale_refine(pargs->crystal, pargs->full, pargs->pmodel,
	                &pargs->n_reflections);
}


static void *get_crystal(void *vqargs)
{
	struct scale_args *task;
	struct queue_args *qargs = vqargs;

	task = malloc(sizeof(struct scale_args));
	memcpy(task, &qargs->task_defaults, sizeof(struct scale_args));

	task->crystal = qargs->crystals[qargs->n_started];

	qargs->n_started++;

	return task;
}


static void done_crystal(void *vqargs, void *task)
{
	struct queue_args *qa = vqargs;
	struct scale_args *wargs = task;

	qa->n_done++;
	qa->n_reflections += wargs->n_reflections;

	progress_bar(qa->n_done, qa->n_crystals, "Scaling");
	free(task);
}


static double total_log_r(Crystal **crystals, int n_crystals, RefList *full,
                          int *ninc)
{
	int i;
	double total = 0.0;
	int n = 0;

	for ( i=0; i<n_crystals; i++ ) {

		double r;
		if ( crystal_get_user_flag(crystals[i]) ) continue;
		r = log_residual(crystals[i], full, 0, NULL, NULL);
		if ( isnan(r) ) continue;
		total += r;
		n++;

	}

	if ( ninc != NULL ) *ninc = n;
	return total;
}


/* Perform iterative scaling, all the way to convergence */
void scale_all(Crystal **crystals, int n_crystals, int nthreads,
               PartialityModel pmodel)
{
	struct scale_args task_defaults;
	struct queue_args qargs;
	double old_res, new_res;
	int niter = 0;

	task_defaults.crystal = NULL;
	task_defaults.pmodel = pmodel;

	qargs.task_defaults = task_defaults;
	qargs.n_crystals = n_crystals;
	qargs.crystals = crystals;

	/* Don't have threads which are doing nothing */
	if ( n_crystals < nthreads ) nthreads = n_crystals;

	new_res = INFINITY;
	do {
		RefList *full;
		int ninc;
		double bef_res;

		full = merge_intensities(crystals, n_crystals, nthreads,
		                         pmodel, 2, INFINITY, 0);
		old_res = new_res;
		bef_res = total_log_r(crystals, n_crystals, full, NULL);

		qargs.task_defaults.full = full;
		qargs.n_started = 0;
		qargs.n_done = 0;
		qargs.n_reflections = 0;
		run_threads(nthreads, scale_crystal, get_crystal, done_crystal,
		            &qargs, n_crystals, 0, 0, 0);
		STATUS("%lli reflections went into the scaling.\n",
		       qargs.n_reflections);

		new_res = total_log_r(crystals, n_crystals, full, &ninc);
		STATUS("Log residual went from %e to %e, %i crystals\n",
		       bef_res, new_res, ninc);

		int i;
		double meanB = 0.0;
		for ( i=0; i<n_crystals; i++ ) {
			meanB += crystal_get_Bfac(crystals[i]);
		}
		meanB /= n_crystals;
		STATUS("Mean B = %e\n", meanB);

		reflist_free(full);
		niter++;

	} while ( (fabs(new_res-old_res) >= 0.01*old_res) && (niter < 10) );

	if ( niter == 10 ) {
		ERROR("Too many iterations - giving up!\n");
	}
}


/* Calculates G, by which list2 should be multiplied to fit list1 */
int linear_scale(const RefList *list1, const RefList *list2, double *G)
{
	const Reflection *refl1;
	const Reflection *refl2;
	RefListIterator *iter;
	int max_n = 256;
	int n = 0;
	double *x;
	double *y;
	double *w;
	int r;
	double cov11;
	double sumsq;

	x = malloc(max_n*sizeof(double));
	w = malloc(max_n*sizeof(double));
	y = malloc(max_n*sizeof(double));
	if ( (x==NULL) || (y==NULL) || (w==NULL) ) {
		ERROR("Failed to allocate memory for scaling.\n");
		return 1;
	}

	int nb = 0;
	for ( refl1 = first_refl_const(list1, &iter);
	      refl1 != NULL;
	      refl1 = next_refl_const(refl1, iter) )
	{
		signed int h, k, l;
		double Ih1, Ih2;
		nb++;

		get_indices(refl1, &h, &k, &l);
		refl2 = find_refl(list2, h, k, l);
		if ( refl2 == NULL ) {
			continue;
		}

		Ih1 = get_intensity(refl1);
		Ih2 = get_intensity(refl2);

		if ( (Ih1 <= 0.0) || (Ih2 <= 0.0) ) continue;
		if ( isnan(Ih1) || isinf(Ih1) ) continue;
		if ( isnan(Ih2) || isinf(Ih2) ) continue;

		if ( n == max_n ) {
			max_n *= 2;
			x = realloc(x, max_n*sizeof(double));
			y = realloc(y, max_n*sizeof(double));
			w = realloc(w, max_n*sizeof(double));
			if ( (x==NULL) || (y==NULL) || (w==NULL) ) {
				ERROR("Failed to allocate memory for scaling.\n");
				return 1;
			}
		}

		x[n] = Ih2 / get_partiality(refl2);
		y[n] = Ih1;
		w[n] = get_partiality(refl2);
		n++;

	}

	if ( n < 2 ) {
		ERROR("Not enough reflections for scaling (had %i, but %i remain)\n", nb, n);
		return 1;
	}

	r = gsl_fit_wmul(x, 1, w, 1, y, 1, n, G, &cov11, &sumsq);

	if ( r ) {
		ERROR("Scaling failed.\n");
		return 1;
	}

	if ( isnan(*G) ) {
		ERROR("Scaling gave NaN (%i pairs)\n", n);
		abort();
		return 1;
	}

	free(x);
	free(y);
	free(w);

	return 0;
}


void scale_all_to_reference(Crystal **crystals, int n_crystals,
                            RefList *reference)
{
	int i;

	for ( i=0; i<n_crystals; i++ ) {
		double G;
		if ( linear_scale(reference,
		                  crystal_get_reflections(crystals[i]),
		                  &G) == 0 )
		{
			if ( isnan(G) ) {
				ERROR("NaN scaling factor for crystal %i\n", i);
			}
			crystal_set_osf(crystals[i], G);
			crystal_set_Bfac(crystals[i], 0.0);
		} else {
			ERROR("Scaling failed for crystal %i\n", i);
		}
		progress_bar(i, n_crystals, "Scaling to reference");
	}
	progress_bar(n_crystals, n_crystals, "Scaling to reference");
}
