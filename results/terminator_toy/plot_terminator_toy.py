"""
Plot from the Terminator Toy test case.
We examine the distribution of the dry density (rho_d)
 and species (X and X2) at different times.

This plots:
(a) rho_d @ t = 0 days, (b) rho_d @ t = 518400 s (6 days), (c) rho_d @ t = 1036800 s (12 days, final time)
(d) X @ t = 0 days, (e) X @ t = 518400 s (6 days), (f) X @ t = 1036800 s (12 days, final time)
(g) X2 @ t = 0 days, (h) X2 @ t = 518400 s (6 days), (i) X2 @ t = 1036800 s (12 days, final time)
"""
from os.path import abspath, dirname
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field, add_colorbar_fig,
    tomplot_field_title, extract_gusto_coords, extract_gusto_field,
    plot_icosahedral_sphere_panels
)

test = 'terminator_toy_rf4_dt450'

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file
results_file_name = f'{abspath(dirname(__file__))}/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}'

# ---------------------------------------------------------------------------- #
# Plot details
# ---------------------------------------------------------------------------- #
subplots_x = 2
subplots_y = 3
field_names = ['X', 'X',
               'X2', 'X2',
               'X_plus_X2_plus_X2', 'X_plus_X2_plus_X2']
time_idxs = [0, -1,
             0, -1,
             0, -1]
cbars = [False, True,
         False, True,
         False, True]

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #

projection = ccrs.Robinson()

Xt_contours = np.linspace(-1e-13, 1e-13, 11)
Xt_colour_scheme = 'BrBG'
Xt_field_label = r'$X_T - X_T^0$ (kg kg$^{-1}$)'

X_contours = np.linspace(0.0, 4.e-6, 15)
X_colour_scheme = 'PuRd'
X_field_label = r'$X$ (kg kg$^{-1}$)'

X2_contours_first = np.linspace(0.0, 2.000000000001e-6, 15)
X2_contours = np.linspace(0.0, 2.0e-6, 15)
X2_colour_scheme = 'PuBu'
X2_field_label = r'$X_2$ (kg kg$^{-1}$)'

contour_method = 'tricontour'

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# PLOTTING
# ---------------------------------------------------------------------------- #

fig = plt.figure(figsize=(12, 12))

# Save filled contour data for adding colourbars at end
all_cf = []

for i, (time_idx, field_name) in enumerate(zip(time_idxs, field_names)):

    ax = fig.add_subplot(subplots_y, subplots_x, 1+i, projection=projection)

    if time_idx == 'midpoint':
        time_idx = int((len(data_file['time'][:]) - 1) / 2)

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    # Quote time in days:
    time = data_file['time'][time_idx] / (24.*60.*60.)

    # Select options for each field --------------------------------------------
    if field_name == 'X_plus_X2_plus_X2':
        field_data -= 4.0e-6
        contours = Xt_contours
        colour_scheme = Xt_colour_scheme
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=None)
        data_format = '.2e'

    elif field_name == 'X':
        contours = X_contours
        colour_scheme = X_colour_scheme
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=None)
        data_format = '.2e'

    elif field_name == 'X2':
        if time_idx == 0:
            contours = X2_contours_first
        else:
            contours = X2_contours
        colour_scheme = X2_colour_scheme
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=None)
        data_format = '.2g'

    # Plot data ----------------------------------------------------------------

    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines, projection=projection
    )
    all_cf.append(cf)

    tomplot_field_title(
        ax, None, minmax=True, minmax_format=data_format,
        field_data=field_data
    )

    if i in [0, 1]:
        ax.text(
            0.5, 1.2, r'$t =$ '+f'{time:.0f} days',
            horizontalalignment='center', transform=ax.transAxes
        )

# Colour bars need adding at end when the full figure exists -------------------
fig.subplots_adjust(wspace=0.12, hspace=-0.05)

for i, (cbar, field_name) in enumerate(zip(cbars, field_names)):

    cbar_labelpad = -50
    data_format = '.1e'

    # Get information for field
    if field_name == 'X_plus_X2_plus_X2':
        field_label = Xt_field_label

    elif field_name == 'X':
        field_label = X_field_label

    elif field_name == 'X2':
        field_label = X2_field_label

    if cbar:
        add_colorbar_fig(
            fig, all_cf[i], field_label, ax_idxs=[i-1, i], location='right',
            cbar_labelpad=cbar_labelpad, cbar_format=data_format
        )

# Save figure ------------------------------------------------------------------

plot_name = f'{plot_stem}.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
