import os
import datetime as dt
import numpy as np

from util import load_feat_data, column

################################################################################

# Write to file
filename = 'dataset/stats.txt'
file_exists = os.path.exists(filename)
append_write = 'a' if(file_exists) else 'w'
f = open(filename, append_write)
f.write("Date: " + str(dt.datetime.now().strftime("%m-%d-%Y")) + '\n')

# Write dataset 
dir_array = ['badeer-r', 'benson-r', 'corman-s', 'hain-m']
dataset = load_feat_data(dir_array)
f.write("Data Set 1: " + str(len(dataset)) + "\n")
for i in range(0, len(dataset[0])):
    f.write("Feature[" + str(i) + "]: " + str(np.mean(column(dataset, i))) + "\n") 
    
f.write("---------------------------------------------------------------------\n")

# Write dataset2
dir_array2 = ['cash-m', 'blair-l']
dataset2 = load_feat_data(dir_array2)
f.write("Data Set 2: " + str(len(dataset2)) + "\n")
for i in range(0, len(dataset2[0])):
    f.write("Feature[" + str(i) + "]: " + str(np.mean(column(dataset2, i))) + "\n") 

f.write("---------------------------------------------------------------------\n")
f.close()