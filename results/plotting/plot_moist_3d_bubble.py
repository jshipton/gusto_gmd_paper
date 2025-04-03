"""
Plots the moist 3d bubble test case, plotting the theta_e field at the final
time (1000 s), in the x-z slice.
"""
from os.path import abspath, dirname
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, tomplot_field_title, extract_gusto_vertical_slice
)

test = 'moist_3d_bubble'

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file
plot_stem = f'{abspath(dirname(__file__))}/../figures/{test}'
results_file_name = f'{abspath(dirname(__file__))}/../results/{test}/field_output.nc'

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
contours = np.linspace(319.5, 322.5, 13)
colour_scheme = 'OrRd'
field_label = r'$\theta_e$ (K)'
contour_method = 'tricontour'
xlims = [0.0, 10.0]
ylims = [0., 10.0]
field_name = 'Theta_e'
slice_along = 'y'
slice_at = 5.0
time_idx = -1

# Things that are likely the same for all plots --------------------------------
set_tomplot_style(20)
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# PLOTTING
# ---------------------------------------------------------------------------- #
fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharex='all', sharey='all')

# Data extraction --------------------------------------------------------------
field_data, coords_X, _, coords_Z = \
    extract_gusto_vertical_slice(
        data_file, field_name, time_idx=time_idx,
        slice_along=slice_along, slice_at=slice_at
    )
time = data_file['time'][time_idx]

# Plot data --------------------------------------------------------------------
cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=320.0)
cf, lines = plot_contoured_field(
    ax, coords_X, coords_Z, field_data, contour_method, contours,
    cmap=cmap, line_contours=lines
)

add_colorbar_ax(ax, cf, field_label, location='right', cbar_labelpad=-30)

tomplot_field_title(
    ax, None, minmax=True, field_data=field_data
)

# Labels -----------------------------------------------------------------------
ax.set_ylabel(r'$z$ (km)', labelpad=-20)
ax.set_ylim(ylims)
ax.set_yticks(ylims)
ax.set_yticklabels(ylims)

ax.set_xlabel(r'$x$ (km)', labelpad=-10)
ax.set_xlim(xlims)
ax.set_xticks(xlims)
ax.set_xticklabels(xlims)

# Save figure ------------------------------------------------------------------
fig.subplots_adjust(wspace=0.15)
plot_name = f'{plot_stem}.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight', dpi=600)
plt.close()
