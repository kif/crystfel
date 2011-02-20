/*
 * detector.c
 *
 * Detector properties
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
#include <assert.h>

#include "image.h"
#include "utils.h"
#include "diffraction.h"
#include "detector.h"
#include "beam-parameters.h"


static int atob(const char *a)
{
	if ( strcasecmp(a, "true") == 0 ) return 1;
	if ( strcasecmp(a, "false") == 0 ) return 0;
	return atoi(a);
}


static int dir_conv(const char *a, signed int *sx, signed int *sy)
{
	if ( strcmp(a, "-x") == 0 ) {
		*sx = -1;  *sy = 0;
		return 0;
	}
	if ( strcmp(a, "x") == 0 ) {
		*sx = 1;  *sy = 0;
		return 0;
	}
	if ( strcmp(a, "+x") == 0 ) {
		*sx = 1;  *sy = 0;
		return 0;
	}
	if ( strcmp(a, "-y") == 0 ) {
		*sx = 0;  *sy = -1;
		return 0;
	}
	if ( strcmp(a, "y") == 0 ) {
		*sx = 0;  *sy = 1;
		return 0;
	}
	if ( strcmp(a, "+y") == 0 ) {
		*sx = 0;  *sy = 1;
		return 0;
	}
	return 1;
}


struct rvec get_q(struct image *image, double fs, double ss,
                  unsigned int sampling, float *ttp, float k)
{
	struct rvec q;
	double twotheta, r, az;
	double rx, ry;
	struct panel *p;
	double xs, ys;

	/* Determine which panel to use */
	const unsigned int x = fs;
	const unsigned int y = ss;
	p = find_panel(image->det, x, y);
	assert(p != NULL);

	/* Convert xs and ys, which are in fast scan/slow scan coordinates,
	 * to x and y */
	xs = (fs-p->min_x)*p->fsx + (ss-p->min_y)*p->ssx;
	ys = (fs-p->min_x)*p->fsy + (ss-p->min_y)*p->ssy;

	rx = (xs + p->cx) / p->res;
	ry = (ys + p->cy) / p->res;

	/* Calculate q-vector for this sub-pixel */
	r = sqrt(pow(rx, 2.0) + pow(ry, 2.0));

	twotheta = atan2(r, p->clen);
	az = atan2(ry, rx);
	if ( ttp != NULL ) *ttp = twotheta;

	q.u = k * sin(twotheta)*cos(az);
	q.v = k * sin(twotheta)*sin(az);
	q.w = k * (cos(twotheta) - 1.0);

	return q;
}


double get_tt(struct image *image, double xs, double ys)
{
	float r, rx, ry;
	struct panel *p;

	p = find_panel(image->det, xs, ys);

	rx = ((float)xs - p->cx) / p->res;
	ry = ((float)ys - p->cy) / p->res;

	r = sqrt(pow(rx, 2.0) + pow(ry, 2.0));

	return atan2(r, p->clen);
}


void record_image(struct image *image, int do_poisson)
{
	int x, y;
	double total_energy, energy_density;
	double ph_per_e;
	double area;
	double max_tt = 0.0;

	/* How many photons are scattered per electron? */
	area = M_PI*pow(image->beam->beam_radius, 2.0);
	total_energy = image->beam->fluence * ph_lambda_to_en(image->lambda);
	energy_density = total_energy / area;
	ph_per_e = (image->beam->fluence /area) * pow(THOMSON_LENGTH, 2.0);
	STATUS("Fluence = %8.2e photons, "
	       "Energy density = %5.3f kJ/cm^2, "
	       "Total energy = %5.3f microJ\n",
	       image->beam->fluence, energy_density/1e7, total_energy*1e6);

	for ( x=0; x<image->width; x++ ) {
	for ( y=0; y<image->height; y++ ) {

		double counts;
		double cf;
		double intensity, sa;
		double pix_area, Lsq;
		double dsq, proj_area;
		struct panel *p;

		intensity = (double)image->data[x + image->width*y];
		if ( isinf(intensity) ) {
			ERROR("Infinity at %i,%i\n", x, y);
		}
		if ( intensity < 0.0 ) {
			ERROR("Negative at %i,%i\n", x, y);
		}
		if ( isnan(intensity) ) {
			ERROR("NaN at %i,%i\n", x, y);
		}

		p = find_panel(image->det, x, y);

		/* Area of one pixel */
		pix_area = pow(1.0/p->res, 2.0);
		Lsq = pow(p->clen, 2.0);

		/* Area of pixel as seen from crystal (approximate) */
		proj_area = pix_area * cos(image->twotheta[x + image->width*y]);

		/* Calculate distance from crystal to pixel */
		dsq = pow(((double)x - p->cx) / p->res, 2.0);
		dsq += pow(((double)y - p->cy) / p->res, 2.0);

		/* Projected area of pixel divided by distance squared */
		sa = proj_area / (dsq + Lsq);

		if ( do_poisson ) {
			counts = poisson_noise(intensity * ph_per_e
			                              * sa * image->beam->dqe );
		} else {
			cf = intensity * ph_per_e * sa * image->beam->dqe;
			counts = cf;
		}

		image->data[x + image->width*y] = counts
		                                  * image->beam->adu_per_photon;
		if ( isinf(image->data[x+image->width*y]) ) {
			ERROR("Processed infinity at %i,%i\n", x, y);
		}
		if ( isnan(image->data[x+image->width*y]) ) {
			ERROR("Processed NaN at %i,%i\n", x, y);
		}
		if ( image->data[x+image->width*y] < 0.0 ) {
			ERROR("Processed negative at %i,%i %f\n", x, y, counts);
		}

		if ( image->twotheta[x + image->width*y] > max_tt ) {
			max_tt = image->twotheta[x + image->width*y];
		}

	}
	progress_bar(x, image->width-1, "Post-processing");
	}

	STATUS("Max 2theta = %.2f deg, min d = %.2f nm\n",
	        rad2deg(max_tt), (image->lambda/(2.0*sin(max_tt/2.0)))/1e-9);

	double tt_side = image->twotheta[(image->width/2)+image->width*0];
	STATUS("At middle of bottom edge: %.2f deg, min d = %.2f nm\n",
	        rad2deg(tt_side), (image->lambda/(2.0*sin(tt_side/2.0)))/1e-9);

	tt_side = image->twotheta[0+image->width*(image->height/2)];
	STATUS("At middle of left edge: %.2f deg, min d = %.2f nm\n",
	        rad2deg(tt_side), (image->lambda/(2.0*sin(tt_side/2.0)))/1e-9);

	STATUS("Halve the d values to get the voxel size for a synthesis.\n");
}


struct panel *find_panel(struct detector *det, int x, int y)
{
	int p;

	for ( p=0; p<det->n_panels; p++ ) {
		if ( (x >= det->panels[p].min_x)
		  && (x <= det->panels[p].max_x)
		  && (y >= det->panels[p].min_y)
		  && (y <= det->panels[p].max_y) ) {
			return &det->panels[p];
		}
	}
	ERROR("No mapping found for %i,%i\n", x, y);

	return NULL;
}


struct detector *get_detector_geometry(const char *filename)
{
	FILE *fh;
	struct detector *det;
	char *rval;
	char **bits;
	int i;
	int reject = 0;
	int x, y, max_x, max_y;

	fh = fopen(filename, "r");
	if ( fh == NULL ) return NULL;

	det = malloc(sizeof(struct detector));
	if ( det == NULL ) {
		fclose(fh);
		return NULL;
	}
	det->n_panels = -1;
	det->panels = NULL;

	do {

		int n1, n2;
		char **path;
		char line[1024];
		int np;

		rval = fgets(line, 1023, fh);
		if ( rval == NULL ) break;
		chomp(line);

		n1 = assplode(line, " \t", &bits, ASSPLODE_NONE);
		if ( n1 < 3 ) {
			for ( i=0; i<n1; i++ ) free(bits[i]);
			free(bits);
			continue;
		}

		if ( bits[1][0] != '=' ) {
			for ( i=0; i<n1; i++ ) free(bits[i]);
			free(bits);
			continue;
		}

		if ( strcmp(bits[0], "n_panels") == 0 ) {
			if ( det->n_panels != -1 ) {
				ERROR("Duplicate n_panels statement.\n");
				fclose(fh);
				free(det);
				for ( i=0; i<n1; i++ ) free(bits[i]);
				free(bits);
				return NULL;
			}
			det->n_panels = atoi(bits[2]);
			det->panels = malloc(det->n_panels
			                      * sizeof(struct panel));
			for ( i=0; i<n1; i++ ) free(bits[i]);
			free(bits);

			for ( i=0; i<det->n_panels; i++ ) {
				det->panels[i].min_x = -1;
				det->panels[i].min_y = -1;
				det->panels[i].max_x = -1;
				det->panels[i].max_y = -1;
				det->panels[i].cx = -1;
				det->panels[i].cy = -1;
				det->panels[i].clen = -1;
				det->panels[i].res = -1;
				det->panels[i].badrow = '-';
				det->panels[i].no_index = 0;
				det->panels[i].peak_sep = 50.0;
				det->panels[i].fsx = 1;
				det->panels[i].fsy = 0;
				det->panels[i].ssx = 0;
				det->panels[i].ssy = 1;
			}

			continue;
		}

		n2 = assplode(bits[0], "/\\.", &path, ASSPLODE_NONE);
		if ( n2 < 2 ) {
			/* This was a top-level option, but not handled above. */
			for ( i=0; i<n1; i++ ) free(bits[i]);
			free(bits);
			for ( i=0; i<n2; i++ ) free(path[i]);
			free(path);
			continue;
		}

		np = atoi(path[0]);
		if ( det->n_panels == -1 ) {
			ERROR("n_panels statement must come first in "
			      "detector geometry file.\n");
			return NULL;
		}

		if ( np > det->n_panels ) {
			ERROR("The detector geometry file said there were %i "
			      "panels, but then tried to specify number %i\n",
			      det->n_panels, np);
			ERROR("Note: panel indices are counted from zero.\n");
			return NULL;
		}

		if ( strcmp(path[1], "min_x") == 0 ) {
			det->panels[np].min_x = atof(bits[2]);
		} else if ( strcmp(path[1], "max_x") == 0 ) {
			det->panels[np].max_x = atof(bits[2]);
		} else if ( strcmp(path[1], "min_y") == 0 ) {
			det->panels[np].min_y = atof(bits[2]);
		} else if ( strcmp(path[1], "max_y") == 0 ) {
			det->panels[np].max_y = atof(bits[2]);
		} else if ( strcmp(path[1], "corner_x") == 0 ) {
			det->panels[np].cx = atof(bits[2]);
		} else if ( strcmp(path[1], "corner_y") == 0 ) {
			det->panels[np].cy = atof(bits[2]);
		} else if ( strcmp(path[1], "clen") == 0 ) {
			det->panels[np].clen = atof(bits[2]);
		} else if ( strcmp(path[1], "res") == 0 ) {
			det->panels[np].res = atof(bits[2]);
		} else if ( strcmp(path[1], "peak_sep") == 0 ) {
			det->panels[np].peak_sep = atof(bits[2]);
		} else if ( strcmp(path[1], "badrow_direction") == 0 ) {
			det->panels[np].badrow = bits[2][0];
			if ( (det->panels[np].badrow != 'x')
			  && (det->panels[np].badrow != 'y')
			  && (det->panels[np].badrow != '-') ) {
				ERROR("badrow_direction must be x, y or '-'\n");
				ERROR("Assuming '-'\n.");
				det->panels[np].badrow = '-';
			}
		} else if ( strcmp(path[1], "no_index") == 0 ) {
			det->panels[np].no_index = atob(bits[2]);
		} else if ( strcmp(path[1], "fs") == 0 ) {
			if ( dir_conv(bits[2], &det->panels[np].fsx,
			                       &det->panels[np].fsy) != 0 ) {
				ERROR("Invalid fast scan direction '%s'\n",
				      bits[2]);
				reject = 1;
			}
		} else if ( strcmp(path[1], "ss") == 0 ) {
			if ( dir_conv(bits[2], &det->panels[np].ssx,
			                       &det->panels[np].ssy) != 0 ) {
				ERROR("Invalid slow scan direction '%s'\n",
				      bits[2]);
				reject = 1;
			}
		} else {
			ERROR("Unrecognised field '%s'\n", path[1]);
		}

		for ( i=0; i<n1; i++ ) free(bits[i]);
		for ( i=0; i<n2; i++ ) free(path[i]);
		free(bits);
		free(path);

	} while ( rval != NULL );

	if ( det->n_panels == -1 ) {
		ERROR("No panel descriptions in geometry file.\n");
		fclose(fh);
		if ( det->panels != NULL ) free(det->panels);
		free(det);
		return NULL;
	}

	max_x = 0;
	max_y = 0;
	for ( i=0; i<det->n_panels; i++ ) {

		if ( det->panels[i].min_x == -1 ) {
			ERROR("Please specify the minimum x coordinate for"
			      " panel %i\n", i);
			reject = 1;
		}
		if ( det->panels[i].max_x == -1 ) {
			ERROR("Please specify the maximum x coordinate for"
			      " panel %i\n", i);
			reject = 1;
		}
		if ( det->panels[i].min_y == -1 ) {
			ERROR("Please specify the minimum y coordinate for"
			      " panel %i\n", i);
			reject = 1;
		}
		if ( det->panels[i].max_y == -1 ) {
			ERROR("Please specify the maximum y coordinate for"
			      " panel %i\n", i);
			reject = 1;
		}
		if ( det->panels[i].cx == -1 ) {
			ERROR("Please specify the centre x coordinate for"
			      " panel %i\n", i);
			reject = 1;
		}
		if ( det->panels[i].cy == -1 ) {
			ERROR("Please specify the centre y coordinate for"
			      " panel %i\n", i);
			reject = 1;
		}
		if ( det->panels[i].clen == -1 ) {
			ERROR("Please specify the camera length for"
			      " panel %i\n", i);
			reject = 1;
		}
		if ( det->panels[i].res == -1 ) {
			ERROR("Please specify the resolution for"
			      " panel %i\n", i);
			reject = 1;
		}
		/* It's OK if the badrow direction is '0' */
		/* It's not a problem if "no_index" is still zero */
		/* The default peak_sep is OK (maybe) */

		if ( det->panels[i].max_x > max_x ) {
			max_x = det->panels[i].max_x;
		}
		if ( det->panels[i].max_y > max_y ) {
			max_y = det->panels[i].max_y;
		}

	}

	for ( x=0; x<=max_x; x++ ) {
	for ( y=0; y<=max_y; y++ ) {
		if ( find_panel(det, x, y) == NULL ) {
			ERROR("Detector geometry invalid: contains gaps.\n");
			reject = 1;
			goto out;
		}
	}
	}
out:
	det->max_x = max_x;
	det->max_y = max_y;

	if ( reject ) return NULL;

	fclose(fh);

	return det;
}


void free_detector_geometry(struct detector *det)
{
	free(det->panels);
	free(det);
}
