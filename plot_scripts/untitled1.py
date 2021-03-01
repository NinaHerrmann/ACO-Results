#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 13:42:27 2020

@author: bamm
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:58:45 2020

@author: bamm
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ll_path = '../bpp_output_data/ll/256_ll_aco_bpp_p'
musket_data = pd.read_csv('../bpp_output_data/musket/paking2.dat', delimiter=',', header=None)

f  = open("speedup-table.txt", "w+") 

#extract data from files
Musket1024Average = [0,0,0,0,0,0]
Musket2048Average = [0,0,0,0,0,0]
Musket4096Average = [0,0,0,0,0,0]
Musket8192Average = [0,0,0,0,0,0]

MusketAverage = [Musket1024Average, Musket2048Average, Musket4096Average, Musket8192Average]

ll1024Average = [0,0,0,0,0,0]
ll2048Average = [0,0,0,0,0,0]
ll4096Average = [0,0,0,0,0,0]
ll8192Average = [0,0,0,0,0,0]

lowlevelAverage = [ll1024Average, ll2048Average, ll4096Average, ll8192Average]

#iterate over files
for f in range(6):
    
    file_path_str = ll_path + str(f) + ".out"
    
    breno_data = pd.read_csv(file_path_str, delimiter=',', header=None)
    
    f.write(" %d & " , f)
     
    for x in range(4):
        ma = 0
        lla = 0
        
        for y in range(10):
            lla = lla + breno_data.at[((x*10)+y), 4]
            
            m_index = ((f*40)+((x*10)+y))
            ma = ma + musket_data.at[m_index, 5]
        
        ma = ma/10
        lla = lla/10
        
        MusketAverage[x][f] = ma
        lowlevelAverage[x][f] = lla
        
        speed = ma / lla
        
        f.write(" %d & " , lla)
        f.write(" %d & " , ma)
        f.write(" %d & " , speed)
    f.write(" \r\n ")

f.close()
