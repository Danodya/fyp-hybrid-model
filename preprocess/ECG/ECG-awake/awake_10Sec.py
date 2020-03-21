import pandas as pd
import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt

#read preprocessed data file
ecg_awake_processed_df = pd.read_csv('../../../data/data-preprocess/ECG/ECG-awake/ECG-awake_processed_data.csv')
print(ecg_awake_processed_df)
awakeProcessedSet = ecg_awake_processed_df.values
print(awakeProcessedSet)
X = awakeProcessedSet[:,0]
print(X)
print(len(X))

Y = awakeProcessedSet[:,1]

#divide the dataset into 12 value categories
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
print(medianArray)
print(len(medianArray))
print(meanArray)
print(len(meanArray))
print(sdArray)
print(len(sdArray))

#create a csv file to store prerocessed data
# with open('../../../data/data-preprocess/ECG/ECG-awake/ECG-awake_10Sec_processed_madian_mean_sd_data.csv','w', newline='') as ncsv:
#     newcsv = csv.writer(ncsv)
#     newcsv.writerow(['MedianNN', 'MeanNN', 'SDNN', 'Class']) # Class (drowsyness level)
#     for i in range(len(meanArray)):
#         newcsv.writerow([medianArray[i], meanArray[i], sdArray[i], Y[i]])
#
# #read the processed data for 10sec period
# ecg_awake_10Sec_processed_df = pd.read_csv('../../../data/data-preprocess/ECG/ECG-awake/ECG-awake_10Sec_processed_data.csv')
# #plot the boxplot
# sns.boxplot(x=ecg_awake_10Sec_processed_df['X'])
# plt.show()