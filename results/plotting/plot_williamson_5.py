"""
Plot from the Terminator Toy test case.
"""
from os.path import abspath, dirname
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field, add_colorbar_fig,
    tomplot_field_title, extract_gusto_coords, extract_gusto_field,
)

results_dirs = ['williamson_5_ssprk3', 'williamson_5_semiimplicit',
                'williamson_5_ssprk3', 'williamson_5_semiimplicit']

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
results_stem = f'{abspath(dirname(__file__))}/../data/'  # where it should eventually be
results_stem = f'{abspath(dirname(__file__))}/../results/'  # where it currently is
plot_stem = f'{abspath(dirname(__file__))}/../figures/williamson_5'

# ---------------------------------------------------------------------------- #
# Plot details
# ---------------------------------------------------------------------------- #
subplots_x = 2
subplots_y = 2
field_name = 'RelativeVorticity'
time_idxs = [3, 3, -1, -1]
subtitles = ['Explicit, 15 days', 'Semi-Implicit, 15 days',
             'Explicit, 50 days', 'Semi-Implicit, 50 days']

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #

projection = ccrs.Robinson()
contours = np.linspace(-1e-4, 1e-4, 17)
colour_scheme = 'RdBu_r'
field_label = r'$\zeta$ (s$^{-1}$)'
contour_method = 'tricontour'

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()

# ---------------------------------------------------------------------------- #
# PLOTTING
# ---------------------------------------------------------------------------- #

fig = plt.figure(figsize=(15, 10))

# Save filled contour data for adding colourbars at end
all_cf = []

for i, (subtitle, results_dir, time_idx) in enumerate(zip(subtitles, results_dirs, time_idxs)):

    results_file_name = f'{results_stem}{results_dir}/field_output.nc'
    data_file = Dataset(results_file_name, 'r')

    ax = fig.add_subplot(subplots_y, subplots_x, 1+i, projection=projection)

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)

    # Plot data ----------------------------------------------------------------

    cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=0.0)

    # Plot data ----------------------------------------------------------------

    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines, projection=projection
    )
    all_cf.append(cf)

    tomplot_field_title(
        ax, subtitle, minmax=True, field_data=field_data
    )

# Colour bars need adding at end when the full figure exists -------------------
fig.subplots_adjust(wspace=0.12, hspace=-0.05)

cbar_labelpad = -50
data_format = '.1e'

add_colorbar_fig(
    fig, all_cf[-1], field_label, ax_idxs=[1, 3], location='right',
    cbar_labelpad=cbar_labelpad, cbar_format=data_format
)

# Save figure ------------------------------------------------------------------

plot_name = f'{plot_stem}.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
