import pandas as pd
import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt

#read preprocessed data file
emg_drowsy_df = pd.read_csv('../../../data/data-preprocess/EMG/EMG-drowsy/EMG-drowsydata2.csv')
print(emg_drowsy_df)
drowsySet = emg_drowsy_df.values
# print(awakeSet)
X = drowsySet[:,0]
print(X)
print(len(X))

Y = drowsySet[:,1]

#divide the dataset into 11 value categories
xSplit = np.array_split(X, 242)
print(xSplit)

#defining the array to store calculated median values
medianArray = []
meanArray = []
sdArray = []
#calculate the median
for i in range(len(xSplit)):
    # medianArray.append(np.median(xSplit[i]))
    medianArray += [np.median(xSplit[i])]
    meanArray += [np.mean(xSplit[i])]
    sdArray += [np.std(xSplit[i])]
meanArray = np.round(meanArray, decimals=3)
sdArray = np.round(sdArray, decimals=3)
print(meanArray)
print(len(meanArray))
print(sdArray)
print(len(sdArray))

#create a csv file to store prerocessed data
with open('../../../data/data-preprocess/EMG/EMG-drowsy/EMG-drowsy_10Sec_processed_madian_mean_sd_data2.csv','w', newline='') as ncsv:
    newcsv = csv.writer(ncsv)
    newcsv.writerow(['MedianX', 'MeanX', 'SDX', 'Class'])  # Class (drowsyness level)
    for i in range(len(meanArray)):
        newcsv.writerow([medianArray[i], meanArray[i], sdArray[i], Y[i]])

#read the processed data for 10sec period
emg_drowsy_10Sec_processed_df = pd.read_csv('../../../data/data-preprocess/EMG/EMG-drowsy/EMG-drowsy_10Sec_processed_data2.csv')
#plot the boxplot
sns.boxplot(x=emg_drowsy_10Sec_processed_df['X'])
plt.show()