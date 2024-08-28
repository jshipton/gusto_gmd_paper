"""
Using tomplot to make plots showing Galewsky jet potential vorticity at 6 days.
"""

import matplotlib.pyplot as plt
from netCDF4 import Dataset
from os.path import abspath, dirname
from tomplot import (set_tomplot_style, tomplot_contours, tomplot_cmap,
                     plot_contoured_field, add_colorbar_fig,
                     tomplot_field_title, extract_gusto_coords,
                     extract_gusto_field, apply_gusto_domain)

# -------------------------------------------------------------------------- #
# Directory for results and plots
# -------------------------------------------------------------------------- #
results_dir = f'{abspath(dirname(__file__))}'
plot_dir = f'{abspath(dirname(__file__))}'
results_file_name = f'{results_dir}/field_output.nc'
plot_name = f'{plot_dir}/GJ_vorticity'

# -------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# -------------------------------------------------------------------------- #
field_name = 'PotentialVorticity'
colour_scheme = 'RdBu_r'
field_label = '6 days'
# Things that are the same for all subplots
time_idx = -1
contour_method = 'tricontour'

# -------------------------------------------------------------------------- #
# Things that are likely the same for all plots
# -------------------------------------------------------------------------- #
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# ----------------------------------------------------------------#
# Data extraction
# --------------------------------------------------------------- #
field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
coords_X, coords_Z = extract_gusto_coords(data_file, field_name)
time = data_file['time'][time_idx]
# --------------------------------------------------------------- #
# Plot data
# --------------------------------------------------------------- #
contours = tomplot_contours(field_data)
cmap, lines = tomplot_cmap(contours, colour_scheme)
cf, _ = plot_contoured_field(ax, coords_X, coords_Z, field_data,
                             contour_method,
                             contours, cmap=cmap, line_contours=lines,
                             plot_contour_lines=True)

ylabel = True
ylabelpad = -30
xlabel = True
xlabelpad = -30

apply_gusto_domain(ax, data_file, ylabel=ylabel, xlabel=xlabel,
                   xlabelpad=-10, ylabelpad=-20)
days = time/(24*60*60)
tomplot_field_title(ax, field_label, minmax=True, field_data=field_data)

add_colorbar_fig(fig, cf, 'Potential Vorticity', cbar_labelpad=-40)

# ----------------------------------------------------------------- #
# Save figure
# ----------------------------------------------------------------- #
print(f'Saving figure to {plot_name}.png')
fig.savefig(f'{plot_name}.png', bbox_inches='tight')
plt.close()
