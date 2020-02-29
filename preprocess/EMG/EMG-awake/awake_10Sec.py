import pandas as pd
import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt

#read preprocessed data file
emg_awake_df = pd.read_csv('../../../data/data-preprocess/EMG/EMG-awake/EMG-awakedata.csv')
print(emg_awake_df)
awakeSet = emg_awake_df.values
# print(awakeSet)
X = awakeSet[:,0]
print(X)
print(len(X))

Y = awakeSet[:,1]

#divide the dataset into 11 value categories
xSplit = np.array_split(X, 273)
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
with open('../../../data/data-preprocess/EMG/EMG-awake/EMG-awake_10Sec_madian_mean_sd_data.csv','w', newline='') as ncsv:
    newcsv = csv.writer(ncsv)
    newcsv.writerow(['MedianX', 'MeanX', 'SDX', 'Class'])  # Class (drowsyness level)
    for i in range(len(meanArray)):
        newcsv.writerow([medianArray[i], meanArray[i], sdArray[i], Y[i]])

#read the processed data for 10sec period
emg_awake_10Sec_processed_df = pd.read_csv('../../../data/data-preprocess/EMG/EMG-awake/EMG-awake_10Sec_processed_data.csv')
#plot the boxplot
sns.boxplot(x=emg_awake_10Sec_processed_df['X'])
plt.show()