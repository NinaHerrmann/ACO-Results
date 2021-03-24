#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:58:45 2020

@author: bamm, Nina Herrmann
"""
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ll_data = pd.read_csv('../../../data_aggregation/Knapsack/Knapsack_LL_Quadro.csv', delimiter=';', header=None)
hl_data = pd.read_csv('../../../data_aggregation/Knapsack/Knapsack_HL_Quadro.csv', delimiter=';', header=None)
#extract data from files
Musket1024Average = [0,0,0,0,0,0]
Musket2048Average = [0,0,0,0,0,0]
Musket4096Average = [0,0,0,0,0,0]
Musket8192Average = [0,0,0,0,0,0]

HL_Average = [Musket1024Average, Musket2048Average, Musket4096Average, Musket8192Average]

ll1024Average = [0,0,0,0,0,0]
ll2048Average = [0,0,0,0,0,0]
ll4096Average = [0,0,0,0,0,0]
ll8192Average = [0,0,0,0,0,0]

LL_Average = [ll1024Average, ll2048Average, ll4096Average, ll8192Average]

Mixed_average = [ll1024Average, Musket1024Average, ll2048Average, Musket2048Average, ll4096Average ,Musket4096Average, ll8192Average, Musket8192Average]

#iterate over files
for f in range(6):
    for x in range(4):
        HL_Average[x][f] = hl_data[4][x + (f*4)]
        LL_Average[x][f] = ll_data[4][x + (f*4)]

###############################################################################

# setup the figure and axes
fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(121, projection='3d')
#ax2 = fig.add_subplot(122, projection='3d')

# fake data
_x = np.arange(6)
_y = np.arange(8)
xx, yy = np.meshgrid(_x, _y)
x, y = xx.ravel(), yy.ravel()


t1 = np.array(Mixed_average)
top = t1.ravel()

bottom = np.zeros_like(top)
width = 1
depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade=True, color = '#348ABD',edgecolor = "black")
ax1.set_title('MKP - Quadro RTX 6000')

ind = np.arange(len(_x))
ax1.set_xticks(ind + width / 2)
ax1.set_xticklabels([6,5,4,3,2,1])

ind = np.arange(4)
ax1.set_yticks(ind * (width * 2) + width)
ax1.set_yticklabels([1024,2048,4096,8192])

ax1.set_xlabel('Problem')
ax1.set_ylabel('Colony Size')
ax1.set_zlabel('Seconds')

#ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
#ax2.set_title('Not Shaded')

plt.show()