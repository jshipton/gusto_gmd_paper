"""
Plots the Baroclinic wave test case.

This plots:
(a) Surface Pressure , (b) Temperature
"""
from os.path import abspath, dirname
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from netCDF4 import Dataset
import numpy as np
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, tomplot_field_title, tomplot_contours,
    extract_gusto_coords, extract_gusto_field, reshape_gusto_data,
    area_restriction
)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file
results_dir = f'/media/thomas/1670402270400B47/results/baroclinic_wave'
plot_stem = f'{abspath(dirname(__file__))}/../../source/figures/'

# ---------------------------------------------------------------------------- #
# Final plot details
# ---------------------------------------------------------------------------- #
field_names = ['Pressure_Vt', 'Temperature']
slice_along = 'z'
colour_schemes = ['PiYG', 'RdBu_r']
centre_titles = ['Surface pressure',  'Temperature']
field_labels = [r'$P \ / $hPa', r'$T \ / $K']

contours = [
    np.linspace(94000, 102000, 17),
    np.linspace(230, 330, 11)
]
norms = [colors.TwoSlopeNorm(vcenter=100000, vmin=min(contours[0]), vmax=max(contours[0])), None]
cmaps = ['PRGn', 'Reds']

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
contour_method = 'tricontour'
domain_limit = {'X' : (-180, 180), 'Y' : (-90, 90)}

time_idx = 10*4
# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
level = 0

y_ticks = [-90, 90]
y_tick_labels = ['90S', '90N']
xticks = [-180, 180]
xtick_labels = ['180W', '180E']

pressure_ticks = [94000, 96000, 98000, 100000, 102000]
pressure_labels = ['940', '960', '980', '1000', '1020']
temp_ticks = [240, 280, 320]
temp_labels = ['240', '280', '320']

cbar_ticks = [pressure_ticks, temp_ticks]
cbar_ticklabels = [pressure_labels, temp_labels]

# ---------------------------------------------------------------------------- #
# PLOTTING
# ---------------------------------------------------------------------------- #

results_file_name = f'{results_dir}/field_output.nc'
data_file = Dataset(results_file_name, 'r')
time = data_file['time'][time_idx]
time_in_days = time / (24*60*60)

fig, axarray = plt.subplots(1, 2, figsize=(15, 6), sharex='col', sharey='row')

for i, (ax, field_name, colour_scheme, field_label, contour, centre_title, cmap,
    norm, cbar_tick, cbar_ticklabel) in enumerate(zip(
        axarray, field_names, colour_schemes, field_labels, contours,
        centre_titles, cmaps, norms, cbar_ticks, cbar_ticklabels)):

    # Data extraction ------------------------------------------------------
    field_full = extract_gusto_field(data_file, field_name, time_idx)
    coords_X_full, coords_Y_full, coords_Z_full = \
        extract_gusto_coords(data_file, field_name)

    # Reshape
    field_full, coords_X_full, coords_Y_full, _ = \
        reshape_gusto_data(field_full, coords_X_full,
                            coords_Y_full, coords_Z_full)

    # Domain restriction
    field_data, coords_hori, coords_Z = \
        area_restriction(field_full[:, level], coords_X_full[:, level],
                                coords_Y_full[:, level], domain_limit)


    _, lines = tomplot_cmap(contour, colour_scheme)

    # Configure cmap -------------------------------------------------------

    # Plot data ------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_hori, coords_Z, field_data, contour_method, contour,
        cmap=cmap, line_contours=lines, norm=norm
    )

    cb = add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_ticks=cbar_tick, pad=0.07)
    cb.ax.set_xticklabels(cbar_ticklabel)

    # Labels ---------------------------------------------------------------
    # Adding titles
    ax.set_title(centre_title, fontsize='17.0')

    # Setting Y / X ticks
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize = '15.0')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize='15.0')

# Save figure ------------------------------------------------------------------
fig.suptitle(rf'Baroclinic wave, Day: {int(time_in_days):02d}', y=0.98, fontsize=20)
plot_name = 'baroclinic_wave_lonlat.png'

print(f'Saving figure to {plot_stem}/{plot_name}')
png_name = f"{plot_stem}/{plot_name}"
fig.savefig(png_name, bbox_inches='tight')
