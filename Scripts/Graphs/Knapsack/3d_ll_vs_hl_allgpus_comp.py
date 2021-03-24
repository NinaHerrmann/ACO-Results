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
_y = np.arange(4)
xx, yy = np.meshgrid(_x, _y)
x, y = xx.ravel(), yy.ravel()


t1 = np.array(LL_Average)
top = t1.ravel()

t2 = np.array(HL_Average)
top2 = t2.ravel()

bottom = np.zeros_like(top)
width = 1
depth = 0.50

x = np.arange(6)

for count in range(4):

    y = np.full((1, 6), count).ravel()

    t1 = np.array(LL_Average[count])
    top = t1.ravel()
    
    t2 = np.array(HL_Average[count])
    top2 = t2.ravel()
    
    bottom = np.zeros_like(top)
        
    ax1.bar3d(x, y, bottom, width, depth, top, shade=True, color = '#348ABD',edgecolor = "black",alpha=0.9)
    
    ax1.bar3d(x, y+depth, bottom, width, depth, top2, shade=True, color = '#A60628', edgecolor = "black" ,alpha=0.9)
    
    
#ax1.bar3d(x, 2*y, bottom, width, depth, top, shade=True, color = '#348ABD',edgecolor = "black",alpha=0.5)
#ax1.bar3d(x, 2*y +depth, bottom, width, depth, top2, shade=True, color = '#A60628', edgecolor = "black" ,alpha=1)

ax1.set_title('MKP - Quadro RTX 6000')

ind = np.arange(len(_x))
ax1.set_xticks(ind + width / 2)
ax1.set_xticklabels([6,5,4,3,2,1])

ind = np.arange(len(_y))
ax1.set_yticks(ind + depth)
ax1.set_yticklabels([1024,2048,4096,8192])

ax1.set_xlabel('Problem')
ax1.set_ylabel('Colony Size')
ax1.set_zlabel('Seconds')

blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="#348ABD")
red_proxy = plt.Rectangle((0, 0), 1, 1, fc="#A60628")
ax1.legend([blue_proxy,red_proxy],['Low-level','Musket'])

#ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
#ax2.set_title('Not Shaded')

plt.show()