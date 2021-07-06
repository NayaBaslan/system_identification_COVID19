from matplotlib import pyplot as plt
import numpy as np
import pdb

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
# x = np.linspace(0, 2*np.pi, 400)
# y = np.sin(x**2)
# ax1.plot(x, y)
# ax1.set_title('Sharing Y axis')
# ax2.scatter(x, y)
# ax3.plot(x, y)
# ax4.scatter(x, y)
#
# plt.xlabel("time")
# plt.ylabel("susceptible")

# x = np.linspace(0, 30, 30)
# y = np.sin(x/6*np.pi)
# error = np.random.normal(0.1, 0.02, size=y.shape)
# y += np.random.normal(0, 0.1, size=y.shape)
#
# pdb.set_trace()
# plt.plot(x, y, 'k-')
# plt.fill_between(x, y-error, y+error, alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99',
#     linewidth=0)
#
#
# # First create some toy data:
# x = np.linspace(0, 2*np.pi, 400)
# y = np.sin(x**2)
#
# # Create just a figure and only one subplot
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_title('Simple plot')
#
# # Create two subplots and unpack the output array immediately
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(x, y)
# ax1.set_title('Sharing Y axis')
# ax2.scatter(x, y)
#
# # f, (ax1, ax2) = plt.subplots(4, 1, sharey=True)
# # ax1.plot(x, y)
# # ax1.set_title('Sharing Y axis')
# # ax2.scatter(x, y)
#
# # Create four polar axes and access them through the returned array
# fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
# axs[0, 0].plot(x, y)
# axs[1, 1].scatter(x, y)
#
# # # Share a X axis with each column of subplots
# # plt.subplots(2, 2, sharex='col')
# #
# # # Share a Y axis with each row of subplots
# # plt.subplots(2, 2, sharey='row')
# #
# # # Share both X and Y axes with all subplots
# # plt.subplots(2, 2, sharex='all', sharey='all')
# #
# # # Note that this is the same as
# # plt.subplots(2, 2, sharex=True, sharey=True)
# #
# # # Create figure number 10 with a single subplot
# # # and clears it if it already exists.
# # fig, ax = plt.subplots(num=10, clear=True)

# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import numpy as np
# import pandas as pd
#
# # Define time range with 12 different months:
# # `MS` stands for month start frequency
# x_data = pd.date_range('2018-01-01', periods=12, freq='MS')
# # Check how this dates looks like:
# print(x_data)
# y_data = np.random.rand(12)
# fig, ax = plt.subplots()
# ax.plot(x_data, y_data)
# # Make ticks on occurrences of each month:
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# # Get only the month to show in the x-axis:
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
# # '%b' means month as locale’s abbreviated name
# plt.show()

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pdb

def plot_colortable(colors, title, sort_colors=True, emptycols=0):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    pdb.set_trace()
    return fig

plot_colortable(mcolors.BASE_COLORS, "Base Colors",
                sort_colors=False, emptycols=1)
plot_colortable(mcolors.TABLEAU_COLORS, "Tableau Palette",
                sort_colors=False, emptycols=2)

plot_colortable(mcolors.CSS4_COLORS, "CSS Colors")

# Optionally plot the XKCD colors (Caution: will produce large figure)
#xkcd_fig = plot_colortable(mcolors.XKCD_COLORS, "XKCD Colors")
#xkcd_fig.savefig("XKCD_Colors.png")


plt.show()
