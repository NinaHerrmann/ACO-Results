import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

my_data = pd.read_csv('/home/ninaherrmann/Research/ACO-Results/Musket-Palma-Results/1,3,5-11.csv', delimiter=',', header=None)

#plt.subplot(2, 2, 1)
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
# X-Achsis
many_years = my_data.iloc[0]
many_years.pop(0)
years = [str(int(year)) for year in many_years]
ind = np.arange(len(years))    # the x locations for the groups
width = 0.35         # the width of the bars

# Data from Musket
Musket1024 = my_data.iloc[1]
Musket2048 = my_data.iloc[2]
Musket4096 = my_data.iloc[3]
Musket8192 = my_data.iloc[4]

Musket1024.pop(0)
Musket2048.pop(0)
Musket4096.pop(0)
Musket8192.pop(0)
BrenoBars_1024 = axes[0, 0].bar(ind, Musket1024, width, bottom=0)
BrenoBars_2048 = axes[1, 0].bar(ind, Musket2048, width, bottom=0)
BrenoBars_4096 = axes[0, 1].bar(ind, Musket4096, width, bottom=0)
BrenoBars_8192 = axes[1, 1].bar(ind, Musket8192, width, bottom=0)

MusketBars_1024 = axes[0, 0].bar(ind+width, Musket1024, width,  bottom=0)
MusketBars_2048 = axes[1, 0].bar(ind+width, Musket2048, width,  bottom=0)
MusketBars_4096 = axes[0, 1].bar(ind+width, Musket4096, width,  bottom=0)
MusketBars_8192 = axes[1, 1].bar(ind+width, Musket8192, width,  bottom=0)

axes[0, 0].set_title('Runtime of Programs')
axes[0, 0].set_xticks(ind + width / 2)
axes[0, 0].set_xticklabels(years)

axes[0, 0].legend((BrenoBars_1024[0], MusketBars_1024[0]), ('Breno', 'Musket'))
axes[0, 0].yaxis.set_units("ms")
axes[0, 0].autoscale_view()
plt.show()