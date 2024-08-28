"""
Using tomplot to make plots comparing Williamson 5 potential vorticity data at
15 days and 50 days.
"""

import matplotlib.pyplot as plt
from netCDF4 import Dataset
from os.path import abspath, dirname
from tomplot import (set_tomplot_style, tomplot_contours, tomplot_cmap,
                     plot_contoured_field, add_colorbar_fig,
                     tomplot_field_title, extract_gusto_coords,
                     extract_gusto_field, apply_gusto_domain)

# --------------------------------------------------------------------------- #
# Directory for results and plots
# --------------------------------------------------------------------------- #
results_dir = f'{abspath(dirname(__file__))}'
plot_dir = f'{abspath(dirname(__file__))}'
results_file_name = f'{results_dir}/field_output.nc'
plot_name = f'{plot_dir}/W5_vorticity'

# --------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# --------------------------------------------------------------------------- #
field_name = 'PotentialVorticity'
colour_scheme = 'RdBu_r'
field_labels = ['(a) 15 days', '(b) 50 days']
# Things that are the same for all subplots
times = [15, 50]
contour_method = 'tricontour'

# --------------------------------------------------------------------------- #
# Things that are likely the same for all plots
# --------------------------------------------------------------------------- #
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')
fig, axarray = plt.subplots(2, 1, figsize=(14, 10), sharey='row', sharex='col')

for i, (ax, field_label, time) in \
    enumerate(zip(axarray.flatten(), field_labels, times)):
    # ----------------------------------------------------------------#
    # Data extraction
    # --------------------------------------------------------------- #
    field_data = extract_gusto_field(data_file, field_name, time)
    coords_X, coords_Z = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time]
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
    xlabel = True if i == 1 else None
    xlabelpad = -30

    apply_gusto_domain(ax, data_file, ylabel=ylabel, xlabel=xlabel,
                       xlabelpad=-10, ylabelpad=-20)
    days = time/(24*60*60)
    tomplot_field_title(ax, field_label, minmax=True, field_data=field_data)

    # move the subplots further apart
    fig.subplots_adjust(wspace=0.1, hspace=0.15)

    # give the overall figure a title
    # fig.suptitle('Potential vorticity in Williamson 5')

add_colorbar_fig(fig, cf, 'Potential Vorticity', cbar_labelpad=-40)

# ----------------------------------------------------------------- #
# Save figure
# ----------------------------------------------------------------- #
print(f'Saving figure to {plot_name}.png')
fig.savefig(f'{plot_name}.png', bbox_inches='tight')
plt.close()
