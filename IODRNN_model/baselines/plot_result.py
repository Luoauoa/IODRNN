import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

X = [15, 30, 60]
DCRNN_MAE = [2.362, 2.371, 2.459]
DCRNN_SDD = [2.271, 2.007, 1.921]

GWN_MAE = [2.29, 2.459, 2.806]
GWN_SDD = [10.95, 9.29, 8.125]

IODRNN_MAE = [2.276, 2.298, 2.336]
IODRNN_SDD = [1.960, 1.874, 1.811]

AGCRN_MAE = [2.128, 2.201, 2.286]
AGCRN_SDD = [7.543, 5.154, 3.786]

DGCRN_MAE = [2.264, 2.396, 2.604]
DGCRN_SDD = [6.345, 3.869, 2.87]

ADN_MAE = [2.673, 2.753, 2.938]
ADN_SDD = [2.417, 2.15, 2.025]

MSDR_MAE = [2.182, 2.264, 2.356]
MSDR_SDD = [4.68, 3.815, 3.105]

SDD_array = np.array([np.array(x) for x in
                      (DCRNN_SDD, IODRNN_SDD, ADN_SDD, GWN_SDD, AGCRN_SDD, DGCRN_SDD, MSDR_SDD)]).reshape(-1, 3)

SDD_array_1 = np.array([np.array(x) for x in
                      (DCRNN_SDD, IODRNN_SDD, ADN_SDD)]).reshape(-1, 3)

SDD_array_2 = np.array([np.array(x) for x in
                      (GWN_SDD, AGCRN_SDD, DGCRN_SDD, MSDR_SDD)]).reshape(-1, 3)
# ------plot with hidden interval--------- #
f, (ax2, ax) = plt.subplots(2, 1, figsize=(6, 4))

# plot the same data on both axes
# SDD
ax.plot(X, DCRNN_SDD, label='DCRNN_SDD', marker='^', linestyle='--')
ax.plot(X, IODRNN_SDD, label='IODRNN_SDD', marker='*', linestyle='dotted')
ax.plot(X, ADN_SDD, label='ADN_SDD', marker='h', linestyle='--')
ax2.plot(X, MSDR_SDD, label='MSDR_SDD', color='purple', marker='x')
ax2.plot(X, AGCRN_SDD, label='AGCRN_SDD', color='black', marker='D', linestyle='--')
ax2.plot(X, GWN_SDD,  label='GWN_SDD', color='brown', marker='.')
ax2.plot(X, DGCRN_SDD, label='DGCRN_SDD', color='gray', marker='+')

# MAE
# ax.plot(X, DCRNN_MAE, label='DCRNN_SDD', marker='^', linestyle='--')
# ax.plot(X, IODRNN_MAE, label='IODRNN_SDD', marker='*', linestyle='dotted')
# ax.plot(X, ADN_MAE, label='ADN_SDD', marker='h', linestyle='--')
# ax.plot(X, MSDR_MAE, label='MSDR_SDD', color='purple', marker='x')
# ax.plot(X, AGCRN_MAE, label='AGCRN_SDD', color='black', marker='D', linestyle='--')
# ax.plot(X, GWN_MAE,  label='GWN_SDD', color='brown', marker='.')
# ax.plot(X, DGCRN_MAE, label='DGCRN_SDD', color='gray', marker='+')

# # zoom-in / limit the view to different portions of the data
ax.set_ylim(1.5, 2.5)  # outliers only
ax2.set_ylim(2.6, 13.5)  # most of the data

# hide the spines between ax and ax2
ax.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

ax.xaxis.tick_bottom()
ax2.tick_params(top=False, bottom=False, labelbottom=False, labeltop=False)  # don't put tick labels at the top

d = .015  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
ax.legend(loc='best', prop={'size': 6})
ax2.legend(loc='upper right', prop={'size': 6})
ax.grid()
ax2.grid()
# plt.legend(loc='best', prop={"size": 6})
plt.xticks(X, [15, 30, 60])
plt.xlabel('Horizon / min')
plt.ylabel('SDD', y=1.1)
# plt.style.use('seaborn')
ax.set_alpha(0.6)
plt.grid(alpha=1)
plt.subplots_adjust(hspace=0.01)
plt.tight_layout()
plt.show()

# bar
# labels = ['0.0', '5.0', '7.5', '10.0', '15.0', '20.0']
# SDD = [2.08, 1.96, 1.91, 2.04, 1.90, 1.89]
# MAE = [2.30, 2.28, 2.37, 2.26, 2.29, 2.31]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, SDD, width, label='SDD')
# rects2 = ax.bar(x + width/2, MAE, width, label='MAE')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Metrics')
# # ax.set_title('')
# plt.xlabel('Threshold')
# plt.xticks(x, labels)
# plt.ylim(1.6, 2.5)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
#
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
#
# fig.tight_layout()
#
# plt.show()
#
# ----------bar plot------------ #
# if __name__ == "__main__":
#     fig, ax1 = plt.subplots(nrows=1, ncols=1)
#     data = SDD_array
#     columns = [r'$15min$', r'$30min$', r'$60min$']
#     rows = [x for x in (100, 70, 50, 20, 10, 5, 1)]
#     names = ['1', '2', '3', '4', '5', '6', '7']
#
#     # Get some pastel shades for the colors
#     n_rows = len(data)
#     colors = plt.cm.BuPu(np.linspace(0, 1, n_rows))
#     bottom = np.zeros(len(columns))
#     index = np.arange(len(columns))
#
#     bar_width = 0.25
#     multiplier = 0
#
#     # Plot bars and create text labels for the table
#     for row in range(n_rows):
#         offset = bar_width * multiplier
#         rects = ax1.bar(index + offset, data[row], bar_width,  color=colors[row], bottom=bottom, label=names[row])
#         ax1.bar_label(rects, padding=3)
#         # bottom += data[row]
#         multiplier += 1
#
#     # Adjust layout to make room for the table:
#     ax1.legend(loc='upper left')
#     ax1.set_xticks(index)
#     ax1.set_xticklabels(columns)
#     ax1.set_ylim(1.5, 12)
#     plt.tight_layout()
#     plt.show()
