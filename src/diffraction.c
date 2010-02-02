/*
 * diffraction.c
 *
 * Calculate diffraction patterns by Fourier methods
 *
 * (c) 2006-2010 Thomas White <taw@physics.org>
 *
 * Part of CrystFEL - crystallography with a FEL
 *
 */


#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>

#include "image.h"
#include "utils.h"
#include "cell.h"
#include "ewald.h"
#include "diffraction.h"
#include "sfac.h"


static double lattice_factor(struct rvec q, double ax, double ay, double az,
                                                double bx, double by, double bz,
                                                double cx, double cy, double cz,
                                                int na, int nb, int nc)
{
	struct rvec Udotq;
	double f1, f2, f3;

	Udotq.u = ax*q.u + ay*q.v + az*q.w;
	Udotq.v = bx*q.u + by*q.v + bz*q.w;
	Udotq.w = cx*q.u + cy*q.v + cz*q.w;

	/* At exact Bragg condition, f1 = na */
	if ( na > 1 ) {
		f1 = sin(M_PI*(double)na*Udotq.u) / sin(M_PI*Udotq.u);
	} else {
		f1 = 1.0;
	}

	/* At exact Bragg condition, f2 = nb */
	if ( nb > 1 ) {
		f2 = sin(M_PI*(double)nb*Udotq.v) / sin(M_PI*Udotq.v);
	} else {
		f2 = 1.0;
	}

	/* At exact Bragg condition, f3 = nc */
	if ( nc > 1 ) {
		f3 = sin(M_PI*(double)nc*Udotq.w) / sin(M_PI*Udotq.w);
	} else {
		f3 = 1.0;
	}

	/* At exact Bragg condition, this will multiply the molecular
	 * part of the structure factor by the number of unit cells,
	 * as desired (more scattering from bigger crystal!) */
	return f1 * f2 * f3;
}


/* Look up the structure factor for the nearest Bragg condition */
static double complex molecule_factor(struct molecule *mol, struct rvec q,
                                      double ax, double ay, double az,
                                      double bx, double by, double bz,
                                      double cx, double cy, double cz)
{
	double hd, kd, ld;
	signed int h, k, l;
	double complex r;

	hd = q.u * ax + q.v * ay + q.w * az;
	kd = q.u * bx + q.v * by + q.w * bz;
	ld = q.u * cx + q.v * cy + q.w * cz;
	h = (signed int)rint(hd);
	k = (signed int)rint(kd);
	l = (signed int)rint(ld);

	r = lookup_sfac(mol->reflections, h, k, l);

	return r;
}


double water_intensity(struct rvec q, double en,
                       double beam_r, double water_r)
{
	double complex fH, fO;
	double s, modq;
	double width;
	double complex ifac;

	/* Interatomic distances in water molecule */
	const double rOH = 0.09584e-9;
	const double rHH = 0.1515e-9;

	/* Volume of water column, approximated as:
	 * (2water_r) * (2beam_r) * smallest(2beam_r, 2water_r)
	 * neglecting the curvature of the faces of the volume */
	if ( beam_r > water_r ) {
		width = 2.0 * water_r;
	} else {
		width = 2.0 * beam_r;
	}
	const double water_v = 2.0*beam_r * 2.0*water_r * width;

	/* Number of water molecules */
	const double n_water = water_v * WATER_DENSITY
	                                        * (AVOGADRO / WATER_MOLAR_MASS);

	/* s = sin(theta)/lambda = 1/2d = |q|/2 */
	modq = modulus(q.u, q.v, q.w);
	s = modq / 2.0;

	fH = get_sfac("H", s, en);
	fO = get_sfac("O", s, en);

	/* Four O-H cross terms */
	ifac = 4.0*fH*fO * sin(2.0*M_PI*modq*rOH)/(2.0*M_PI*modq*rOH);

	/* Three H-H cross terms */
	ifac += 3.0*fH*fH * sin(2.0*M_PI*modq*rHH)/(2.0*M_PI*modq*rHH);

	/* Three diagonal terms */
	ifac += 2.0*fH*fH + fO*fO;

	return cabs(ifac) * n_water;
}


void get_diffraction(struct image *image, int na, int nb, int nc)
{
	int x, y;
	double ax, ay, az;
	double bx, by, bz;
	double cx, cy, cz;
	double a, b, c, d;

	/* Generate the array of reciprocal space vectors in image->qvecs */
	if ( image->qvecs == NULL ) {
		get_ewald(image);
	}

	if ( image->molecule == NULL ) return;

	cell_get_cartesian(image->molecule->cell, &ax, &ay, &az,
		                                  &bx, &by, &bz,
		                                  &cx, &cy, &cz);

	cell_get_parameters(image->molecule->cell,
	                    &a, &b, &c, &d, &d, &d);
	STATUS("Particle size = %i x %i x %i (=%5.2f x %5.2f x %5.2f nm)\n",
	       na, nb, nc, na*a/1.0e-9, nb*b/1.0e-9, nc*c/1.0e-9);

	image->sfacs = malloc(image->width * image->height
	                      * sizeof(double complex));

	if ( image->molecule->reflections == NULL ) {
		get_reflections_cached(image->molecule,
		                       ph_lambda_to_en(image->lambda));
	}

	for ( x=0; x<image->width; x++ ) {
	for ( y=0; y<image->height; y++ ) {

		double f_lattice;
		double complex f_molecule;
		struct rvec q;
		double complex val;

		q = image->qvecs[x + image->width*y];

		f_lattice = lattice_factor(q, ax,ay,az,bx,by,bz,cx,cy,cz,
		                           na, nb, nc);
		f_molecule = molecule_factor(image->molecule, q,
		                             ax,ay,az,bx,by,bz,cx,cy,cz);

		val = f_molecule * f_lattice;
		image->sfacs[x + image->width*y] = val;

	}
	progress_bar(x, image->width-1, "Calculating lattice factors");
	}
}
