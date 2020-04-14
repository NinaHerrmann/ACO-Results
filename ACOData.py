import csv
import numpy as np
import pandas as pd

my_data=pd.read_csv('/home/n_herr03/Research/ACO-Results/75_5-12.dat', delimiter=';', header=None)
currentcity = 0
results = pd.DataFrame(np.zeros((4, 12)))
sum = 0
counter = 0
counterants = 0
numberants= 0
lastnumberants = 1024
print my_data
lastcity = my_data.iloc[0][2]
for index, row in my_data.iterrows():
    currentcity = row[2]
    numberants = row[3]
    if currentcity != lastcity:
        print counter
        results.iloc[counterants][lastcity-1] = sum / counter
        lastcity = currentcity
        sum = 0
        counter = 0
        counterants = 0
        numberants = 0
        lastnumberants = row[3]
    elif numberants != lastnumberants:
        print counter
        results.iloc[counterants][lastcity-1] = sum / counter
        counterants = counterants + 1
        counter = 0
        sum = 0
        lastnumberants = row[3]
    sum += row[6]
    counter = counter + 1
results.to_csv('result.csv')
