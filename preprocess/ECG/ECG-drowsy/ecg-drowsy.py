import pandas as pd
import numpy as np
from scipy import stats
import csv
import matplotlib.pyplot as plt
import seaborn as sns

# ecg_drowsy_df = pd.read_csv('../../../data/data-preprocess/ECG/ECG-drowsy/ECG-drowsydata.csv') #for the first run
ecg_drowsy_df = pd.read_csv('../../../data/data-preprocess/ECG/ECG-drowsy/ECG-drowsy_processed_data.csv')
# ecg_drowsy_processed_df = pd.read_csv('../../../data/data-preprocess/ECG/ECG-drowsy/ECG-drowsy_processed_data2.csv')
# ecg_drowsy_processed_df = pd.read_csv('../../../data/data-preprocess/ECG/ECG-drowsy/ECG-drowsy_processeddatafiltered2.csv')
ecg_drowsy_processed_df = pd.read_csv('../../../data/data-preprocess/ECG/ECG-drowsy/ECG-drowsyFiltereddata3.csv')
print(ecg_drowsy_df)
drowsySet = ecg_drowsy_df.values
# print(awakeSet)
X = drowsySet[:,0]
print(X)
print(len(X))

Y = drowsySet[:,1]
#calculating z-scores
z = np.abs(stats.zscore(X))
# score = np.where(z>0)
print(z)

#settingup a threshold value
# threshold = 2 #threshold for the first run
# threshold = 1 #threshold for the second run
threshold = 1.15 #threshold for the third run
outlier = np.where(z > threshold)
print(outlier)

#indexes of outliers
outlierArray = outlier[0]
print(len(outlierArray))

outliervalue = []
for i in range(len(outlierArray)):
    for j in range(len((X))):
        if outlierArray[i] == j:
            print(X[j])

#removing outliers
X = np.delete(X, outlierArray, None)
print(X)
print(len(X))

# #create a csv file to store prerocessed data
# with open('../../../data/data-preprocess/ECG/ECG-drowsy/ECG-drowsy_processed_data2.csv','w', newline='') as ncsv:
#     newcsv = csv.writer(ncsv)
#     newcsv.writerow(['X','Class']) # Class (drowsyness level)
#     for i in range(len(X)):
#         newcsv.writerow([X[i], Y[i]])

# #create a csv file to store zscore data
# with open('../../../data/data-preprocess/ECG/ECG-drowsy/ECG-drowsy_zscoreset2.csv','w', newline='') as ncsv:
#     newcsv = csv.writer(ncsv)
#     newcsv.writerow(['z']) # z-zscore
#     for i in range(len(z)):
#         newcsv.writerow([z[i]])

#plot the boxplot
sns.boxplot(x=ecg_drowsy_processed_df['X'])
plt.show()