import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

my_data = pd.read_csv('../data_aggregation/Musket_1,3,5-12_average.csv', delimiter=',', header=None)
breno_data = pd.read_csv('../data_aggregation/LowLevel_1,3,5-12_average.csv', delimiter=',', header=None)

#define size
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 10))

# X-Achsis
many_years = my_data.iloc[0]
years = [str(int(year)) for year in many_years][:-1]
ind = np.arange(len(years))    # the x locations for the groups
width = 0.35         # the width of the bars

# Data from Musket
Musket1024 = my_data.iloc[1].astype(float)[:-1]
Musket2048 = my_data.iloc[2].astype(float)[:-1]
Musket4096 = my_data.iloc[3].astype(float)[:-1]
Musket8192 = my_data.iloc[4].astype(float)[:-1]

# Data Breno
Breno1024 = breno_data.iloc[1].astype(float)[:-1]
Breno2048 = breno_data.iloc[2].astype(float)[:-1]
Breno4096 = breno_data.iloc[3].astype(float)[:-1]
Breno8192 = breno_data.iloc[4].astype(float)[:-1]

BrenoBars_1024 = axes[0, 0].bar(ind+width, Breno1024, width, bottom=0)
BrenoBars_2048 = axes[1, 0].bar(ind+width, Breno2048, width, bottom=0)
BrenoBars_4096 = axes[0, 1].bar(ind+width, Breno4096, width, bottom=0)
BrenoBars_8192 = axes[1, 1].bar(ind+width, Breno8192, width, bottom=0)

MusketBars_1024 = axes[0, 0].bar(ind, Musket1024, width,  bottom=0)
MusketBars_2048 = axes[1, 0].bar(ind, Musket2048, width,  bottom=0)
MusketBars_4096 = axes[0, 1].bar(ind, Musket4096, width,  bottom=0)
MusketBars_8192 = axes[1, 1].bar(ind, Musket8192, width,  bottom=0)

axes[0, 0].set_title('1024 Ants')
axes[1, 0].set_title('2048 Ants')
axes[0, 1].set_title('4096 Ants')
axes[1, 1].set_title('8192 Ants')

axes[0, 0].set_ylabel('seconds')
axes[1, 0].set_ylabel('seconds')
axes[0, 1].set_ylabel('seconds')
axes[1, 1].set_ylabel('seconds')

axes[0, 0].set_xlabel('cities')
axes[1, 0].set_xlabel('cities')
axes[0, 1].set_xlabel('cities')
axes[1, 1].set_xlabel('cities')

axes[0, 0].set_xticks(ind + width / 2)
axes[0, 0].set_xticklabels(years)
for ax in axes.flat:
    ax.label_outer()
for ax in fig.get_axes():
    ax.label_outer()
axes[0, 0].legend((BrenoBars_1024[0], MusketBars_1024[0]), ('Low-level', 'Musket'))
axes[0, 0].autoscale_view()

plt.show()
