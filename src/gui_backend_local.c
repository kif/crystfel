/*
 * gui_backend_local.c
 *
 * GUI backend for running jobs on the local machine
 *
 * Copyright © 2020 Deutsches Elektronen-Synchrotron DESY,
 *                  a research centre of the Helmholtz Association.
 *
 * Authors:
 *   2020 Thomas White <taw@physics.org>
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

#include <pty.h>
#include <glib.h>
#include <sys/wait.h>
#include <gtk/gtk.h>

#include "crystfel_gui.h"


struct local_backend_priv
{
	int indexamajig_running;
	guint indexamajig_watch;
	GPid indexamajig_pid;
};


static void watch_indexamajig(GPid pid, gint status, gpointer vp)
{
	struct crystfelproject *proj = vp;
	struct local_backend_priv *priv = proj->backend_private;
	STATUS("Indexamajig exited with status %i\n", status);
	priv->indexamajig_running = 0;
	g_spawn_close_pid(priv->indexamajig_pid);
	remove_infobar(proj);
}


static gboolean index_readable(GIOChannel *source, GIOCondition cond,
                               void *vp)
{
	GIOStatus r;
	GError *err = NULL;
	struct crystfelproject *proj = vp;
	struct local_backend_priv *priv = proj->backend_private;
	gchar *line;

	r = g_io_channel_read_line(source, &line, NULL, NULL, &err);
	if ( r == G_IO_STATUS_EOF ) {
		STATUS("End of output.\n");
		return FALSE;
	}
	if ( r != G_IO_STATUS_NORMAL ) {
		if ( priv->indexamajig_pid != 0 ) {
			STATUS("Read error?\n");
		} else {
			STATUS("End of output (indexamajig exited)\n");
		}
		return FALSE;
	}
	chomp(line);

	if ( strstr(line, " images processed, ") != NULL ) {
		double frac;
		int n_proc;
		sscanf(line, "%i ", &n_proc);
		frac = (double)n_proc/proj->n_frames;
		gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(proj->progressbar),
		                              frac);
	} else {
		STATUS("%s\n", line);
	}

	g_free(line);

	return TRUE;
}


static int write_file_list(struct crystfelproject *proj)
{
	FILE *fh;
	int i;

	fh = fopen("files.lst", "w");
	if ( fh == NULL ) return 1;

	for ( i=0; i<proj->n_frames; i++ ) {
		fprintf(fh, "%s", proj->filenames[i]);
		if ( proj->events[i] != NULL ) {
			fprintf(fh, " %s\n", proj->events[i]);
		} else {
			fprintf(fh, "\n");
		}
	}

	fclose(fh);

	return 0;
}


static void add_arg(char **args, int pos, const char *label,
                    float val)
{
	char *str;

	str = malloc(64);
	if ( str == NULL ) return;

	snprintf(str, 63, "--%s=%f", label, val);
	args[pos] = str;
}


static int run_unitcell(struct crystfelproject *proj,
                        const char *algo)
{
	GIOChannel *ioch;
	char *args[64];
	char index_str[64];
	char peaks_str[64];
	int n_args;
	int i;
	int r;
	int ch_stderr;
	GError *error;
	struct local_backend_priv *priv = proj->backend_private;

	if ( priv->indexamajig_running != 0 ) {
		STATUS("Indexamajig already running.\n");
		return 1;
	}

	if ( write_file_list(proj) ) {
		STATUS("Failed to write list\n");
		return 1;
	}

	strcpy(index_str, "--indexing=");
	strncat(index_str, algo, 50);

	strcpy(peaks_str, "--peaks=");
	strncat(peaks_str,
	        str_peaksearch(proj->peak_search_params.method), 50);

	args[0] = "indexamajig";
	args[1] = "-i";
	args[2] = "files.lst";
	args[3] = "-g";
	args[4] = proj->geom_filename;
	args[5] = "-o";
	args[6] = "test.stream";
	args[7] = index_str;
	args[8] = "--no-check-cell";
	args[9] = "-j";
	args[10] = "1";
	args[11] = "--integration=none";
	args[12] = peaks_str;
	n_args = 13;

	if ( proj->peak_search_params.method == PEAK_ZAEF ) {
		add_arg(args, n_args++, "threshold",
		        proj->peak_search_params.threshold);
		add_arg(args, n_args++, "min-squared-gradient",
		        proj->peak_search_params.min_sq_gradient);
		add_arg(args, n_args++, "min-snr",
		        proj->peak_search_params.min_snr);
	} else if ( proj->peak_search_params.method == PEAK_PEAKFINDER8 ) {
		add_arg(args, n_args++, "threshold",
		        proj->peak_search_params.threshold);
		add_arg(args, n_args++, "min-snr",
		        proj->peak_search_params.min_snr);
		add_arg(args, n_args++, "min-pix-count",
		        proj->peak_search_params.min_pix_count);
		add_arg(args, n_args++, "max-pix-count",
		        proj->peak_search_params.max_pix_count);
		add_arg(args, n_args++, "local-bg-radius",
		        proj->peak_search_params.local_bg_radius);
		add_arg(args, n_args++, "min-res",
		        proj->peak_search_params.min_res);
		add_arg(args, n_args++, "max-res",
		        proj->peak_search_params.max_res);
	}

	args[n_args] = NULL;

	for ( i=0; i<n_args; i++ ) {
		STATUS("%s ", args[i]);
	}
	STATUS("\n");

	r = g_spawn_async_with_pipes(NULL, args, NULL,
	                             G_SPAWN_SEARCH_PATH
	                           | G_SPAWN_DO_NOT_REAP_CHILD,
	                             NULL, NULL,
	                             &priv->indexamajig_pid,
	                             NULL, NULL, &ch_stderr,
	                             &error);
	if ( r == FALSE ) {
		ERROR("Failed to run indexamajig: %s\n",
		      error->message);
		return 1;
	}
	priv->indexamajig_running = 1;

	g_child_watch_add(priv->indexamajig_pid,
	                  watch_indexamajig, proj);

	ioch = g_io_channel_unix_new(ch_stderr);
	g_io_add_watch(ioch, G_IO_IN | G_IO_ERR | G_IO_HUP,
	               index_readable, proj);

	return 0;
}


static void cancel(struct crystfelproject *proj)
{
	struct local_backend_priv *priv = proj->backend_private;

	if ( !priv->indexamajig_running ) return;

	ERROR("Stopping indexamajig.\n");
	kill(priv->indexamajig_pid, SIGQUIT);
}


static void shutdown_backend(struct crystfelproject *proj)
{
	struct local_backend_priv *priv = proj->backend_private;

	/* FIXME: Shutdown indexamajig child process */

	free(priv);
}


static void init_backend(struct crystfelproject *proj)
{
	struct local_backend_priv *priv;

	priv = malloc(sizeof(struct local_backend_priv));
	if ( priv == NULL ) {
		ERROR("Failed to initialise backend\n");
		return;
	}

	priv->indexamajig_running = 0;

	proj->backend_private = priv;
	STATUS("Local backend initialised.\n");
}


struct crystfel_backend _backend_local =
	{
	 .name = "local",
	 .init = init_backend,
	 .shutdown = shutdown_backend,
	 .run_unitcell = run_unitcell,
	 .cancel = cancel,
	};

struct crystfel_backend *backend_local = &_backend_local;
