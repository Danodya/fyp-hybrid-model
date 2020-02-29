import pandas as pd
import numpy as np
from scipy import stats
import csv
import matplotlib.pyplot as plt
import seaborn as sns

# ecg_moderate_df = pd.read_csv('../../../data/data-preprocess/ECG/ECG-moderate/ECG-moderatedata.csv') #for the first run
ecg_moderate_df = pd.read_csv('../../../data/data-preprocess/ECG/ECG-moderate/ECG-moderate_processed_data.csv')
ecg_moderate_processed_df = pd.read_csv('../../../data/data-preprocess/ECG/ECG-moderate/ECG-moderate_processed_data2.csv')
print(ecg_moderate_df)
moderateSet = ecg_moderate_df.values
# print(awakeSet)
X = moderateSet[:,0]
print(X)
print(len(X))

Y = moderateSet[:,1]
#calculating z-scores
z = np.abs(stats.zscore(X))
# score = np.where(z>0)
print(z)

#settingup a threshold value
# threshold = 0.9 #threshold for the first run
threshold = 2
outlier = np.where(z > threshold)
print(outlier)

#indexes of outliers
outlierArray = outlier[0]
print(len(outlierArray))

# outliervalue = []
# for i in range(len(outlierArray)):
#     for j in range(len((X))):
#         if outlierArray[i] == j:
#             print(X[j])

#removing outliers
X = np.delete(X, outlierArray, None)
print(X)
# print(len(X))

# #create a csv file to store prerocessed data
# with open('../../../data/data-preprocess/ECG/ECG-moderate/ECG-moderate_processed_data2.csv','w', newline='') as ncsv:
#     newcsv = csv.writer(ncsv)
#     newcsv.writerow(['X','Class']) # Class (drowsyness level)
#     for i in range(len(X)):
#         newcsv.writerow([X[i], Y[i]])

# #create a csv file to store zscore data
# with open('../../../data/data-preprocess/ECG/ECG-moderate/ECG-moderate_zscoreset3.csv','w', newline='') as ncsv:
#     newcsv = csv.writer(ncsv)
#     newcsv.writerow(['z']) # z-zscore
#     for i in range(len(z)):
#         newcsv.writerow([z[i]])

#plot the boxplot
sns.boxplot(x=ecg_moderate_processed_df['X'])
plt.show()