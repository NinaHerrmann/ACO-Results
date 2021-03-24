#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:58:45 2020

@author: bamm
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ma_path = '../bpp_output_data/musket/70paking.dat'
ma_path2080 = '../bpp_output_data/musket/paking2.dat'

ll_path = '../bpp_output_data/ll/256_v100_complete_bpp.out'
ll_path2080 = '../bpp_output_data/ll/256_ll_aco_bpp_p'

#extract data from files
Musket1024Average = [0,0,0,0,0,0]
Musket2048Average = [0,0,0,0,0,0]
Musket4096Average = [0,0,0,0,0,0]
Musket8192Average = [0,0,0,0,0,0]

MusketAverage = [Musket1024Average, Musket2048Average, Musket4096Average, Musket8192Average]

Musket1024Average2 = [0,0,0,0,0,0]
Musket2048Average2 = [0,0,0,0,0,0]
Musket4096Average2 = [0,0,0,0,0,0]
Musket8192Average2 = [0,0,0,0,0,0]

MusketAverage2 = [Musket1024Average2, Musket2048Average2, Musket4096Average2, Musket8192Average2]

ll1024Average = [0,0,0,0,0,0]
ll2048Average = [0,0,0,0,0,0]
ll4096Average = [0,0,0,0,0,0]
ll8192Average = [0,0,0,0,0,0]

lowlevelAverage = [ll1024Average, ll2048Average, ll4096Average, ll8192Average]

ll1024Average2 = [0,0,0,0,0,0]
ll2048Average2 = [0,0,0,0,0,0]
ll4096Average2 = [0,0,0,0,0,0]
ll8192Average2 = [0,0,0,0,0,0]

lowlevelAverage2 = [ll1024Average2, ll2048Average2, ll4096Average2, ll8192Average2]

#iterate over files
for f in range(6):
    
    ma_path_str = ma_path
    ma_path_str2080 = ma_path2080
    
    file_path_str = ll_path
    file_path_str2080 = ll_path2080 + str(f) + ".out"
    
    ll_data = pd.read_csv(file_path_str, delimiter=',', header=None)
    ll_data2080 = pd.read_csv(file_path_str2080, delimiter=',', header=None)
    
    ma_data = pd.read_csv(ma_path_str, delimiter=';', header=None)
    ma_data2080 = pd.read_csv(ma_path_str2080, delimiter=',', header=None)
    
    for x in range(4):
        ma = 0
        ma2080 = 0
        
        lla = 0
        lla2080 = 0
        
        for y in range(5):
            lla = lla + ll_data.at[((f*20)+(x*5)+y), 3]            
            
        for y in range(10):
            lla2080 = lla2080 + ll_data2080.at[((x*10)+y), 4]  
            ma = ma + ma_data.at[((f*40)+(x*10)+y), 19]
            ma2080 = ma2080 + ma_data2080.at[((f*40)+(x*10)+y), 5]  


        ma = ma/10
        ma2080 = ma2080/10
        lla2080 = lla2080/10
        lla = lla/5
        
        MusketAverage[x][f] = ma
        lowlevelAverage[x][f] = lla
        MusketAverage2[x][f] = ma2080
        lowlevelAverage2[x][f] = lla2080

#define size
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 10))

# X-Achsis
years = [0,1,2,3,4,5]
ind = np.arange(len(years))
width = 0.20       # the width of the bars

BrenoBars_1024 = axes[0, 0].bar(ind, ll1024Average, width, bottom=0, color = '#A60628')
BrenoBars_2048 = axes[1, 0].bar(ind, ll2048Average, width, bottom=0, color = '#A60628')
BrenoBars_4096 = axes[0, 1].bar(ind, ll4096Average, width, bottom=0, color = '#A60628')
BrenoBars_8192 = axes[1, 1].bar(ind, ll8192Average, width, bottom=0, color = '#A60628')

MusketBars_1024 = axes[0, 0].bar(ind+width, Musket1024Average, width,  bottom=0, color = '#348ABD')
MusketBars_2048 = axes[1, 0].bar(ind+width, Musket2048Average, width,  bottom=0, color = '#348ABD')
MusketBars_4096 = axes[0, 1].bar(ind+width, Musket4096Average, width,  bottom=0, color = '#348ABD')
MusketBars_8192 = axes[1, 1].bar(ind+width, Musket8192Average, width,  bottom=0, color = '#348ABD')

BrenoBars_1024_2 = axes[0, 0].bar(ind+(2*width), ll1024Average2, width, bottom=0, color = '#4C9900')
BrenoBars_2048_2 = axes[1, 0].bar(ind+(2*width), ll2048Average2, width, bottom=0, color = '#4C9900')
BrenoBars_4096_2 = axes[0, 1].bar(ind+(2*width), ll4096Average2, width, bottom=0, color = '#4C9900')
BrenoBars_8192_2 = axes[1, 1].bar(ind+(2*width), ll8192Average2, width, bottom=0, color = '#4C9900')

MusketBars_1024_2 = axes[0, 0].bar(ind+(3*width), Musket1024Average2, width,  bottom=0, color = '#D27E2A')
MusketBars_2048_2 = axes[1, 0].bar(ind+(3*width), Musket2048Average2, width,  bottom=0, color = '#D27E2A')
MusketBars_4096_2 = axes[0, 1].bar(ind+(3*width), Musket4096Average2, width,  bottom=0, color = '#D27E2A')
MusketBars_8192_2 = axes[1, 1].bar(ind+(3*width), Musket8192Average2, width,  bottom=0, color = '#D27E2A')

axes[0, 0].set_title('1024 Ants')
axes[1, 0].set_title('2048 Ants')
axes[0, 1].set_title('4096 Ants')
axes[1, 1].set_title('8192 Ants')

axes[0, 0].set_ylabel('seconds')
axes[1, 0].set_ylabel('seconds')
axes[0, 1].set_ylabel('seconds')
axes[1, 1].set_ylabel('seconds')

axes[0, 0].set_xlabel('problem')
axes[1, 0].set_xlabel('problem')
axes[0, 1].set_xlabel('problem')
axes[1, 1].set_xlabel('problem')

axes[0, 0].set_xticks(ind + ((3*width)/2))
axes[0, 0].set_xticklabels(years)

#for ax in axes.flat:
    #ax.label_outer()
#for ax in fig.get_axes():
 #   ax.label_outer()
axes[0, 0].legend((BrenoBars_1024[0],MusketBars_1024[0],BrenoBars_1024_2[0],MusketBars_1024_2[0]), ('Low-level - v100','Musket - v100', 'Low-level - 2080 Ti','Musket - 2080 Ti'))
axes[0, 0].autoscale_view()

plt.show()
