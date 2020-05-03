# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

md = pd.read_csv('../data_aggregation/LowLevel_1,3,5-12_splitkernels.csv', delimiter=',', header=None).T
md_musket = pd.read_csv('../data_aggregation/Musket_1,3,5-12_kernel_sumofiterations.csv', delimiter=',', header=None).T

# define size
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 10))
fig.subplots_adjust(hspace=0.15, wspace=0.05)
fig.suptitle('Kernel Percentage', fontsize=16, y=0.95)


legendlabels = ["" for x in range(2)]
#legendlabels = pd.DataFrame(np.zeros((1, 10)))
legendlabels[0] = 'Other'
legendlabels[1] = 'Calculate Route'

axes[0].set_title('Low Level')
axes[1].set_title('High Level')
axes[1].set_ylabel('%')
axes[0].set_ylabel('%')

axes[1].set_xlabel('tsp instance')
axes[0].set_xlabel('tsp instance')

# From raw value to percentage

setup_index = 0
barWidth = 0.85
colors = np.array(['#1f77b4', '#ff7f0e'])
pos = np.arange(10)

percents = np.zeros((2, 10))

othertimes = np.array(md[1 + (8 * setup_index)] + md[2 + (8 * setup_index)] + md[3 + (8 * setup_index)] +
                       md[4 + (8 * setup_index)] + md[6 + (8 * setup_index)] +
                       md[7 + (8 * setup_index)] + md[8 + (8 * setup_index)])
percents[0] = np.array(md[5 + (8 * setup_index)] / (othertimes + md[5 + (8 * setup_index)]))
percents[1] = np.array(othertimes / (othertimes + md[5 + (8 * setup_index)]))
bottoms = np.array([0,0,0,0,0,0,0,0,0,0])
print percents
for x in range(2):
    # Create green Bars
    a = percents[x]
    axes[0].bar(pos, a, color=colors[x], bottom=bottoms, edgecolor='white', width=barWidth, label=legendlabels[x])
    bottoms = bottoms + percents[x]

colors = np.array(['#1f77b4', '#ff7f0e'])


percents = np.zeros((2, 10))

othertimes = np.array(md_musket[1 + (9 * setup_index)] + md_musket[2 + (9 * setup_index)] + md_musket[3 + (9 * setup_index)] +
                       md_musket[4 + (9 * setup_index)] + md_musket[6 + (9 * setup_index)] + md_musket[7 + (9 * setup_index)] +
                       md_musket[8 + (9 * setup_index)] + md_musket[9 + (9 * setup_index)])
routetime = np.array(md_musket[5 + (9 * setup_index)])
percents[0] = np.array(routetime / (othertimes + routetime))
percents[1] = np.array(othertimes / (othertimes + routetime))
bottoms = np.array([0,0,0,0,0,0,0,0,0,0])

for x in range(2):
    # Create green Bars
    a = percents[x]
    axes[1].bar(pos, a, color=colors[x], bottom=bottoms, edgecolor='white', width=barWidth, label=legendlabels[x])
    bottoms = bottoms + percents[x]

width = 0.45
axes[0].set_xticks((pos + (width / 2)) - 0.15)
axes[1].set_xticks((pos + (width / 2)) - 0.15)

axes[0].set_xticklabels(['dj38', 'qa194', 'd198', 'lin318', 'pcb442', 'rat783', 'pr1002', 'pcb1173', 'd1291', 'pr2392'])
axes[1].set_xticklabels(['dj38', 'qa194', 'd198', 'lin318', 'pcb442', 'rat783', 'pr1002', 'pcb1173', 'd1291', 'pr2392'])

# Put a legend below current axis
axes[1].legend(loc='center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5)
# 1.05, -1.30
for ax in axes.flat:
    ax.label_outer()
for ax in fig.get_axes():
    ax.label_outer()
# Show graphic
plt.show()