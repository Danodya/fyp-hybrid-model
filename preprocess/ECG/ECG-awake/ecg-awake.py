import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import csv
import matplotlib.pyplot as plt

ecg_awake_df = pd.read_csv('../../../data/data-preprocess/ECG/ECG-awake/ECG-awakedata.csv')

print(ecg_awake_df)
awakeSet = ecg_awake_df.values
X = awakeSet[:,0]
print(X)
print(len(X))
Y = awakeSet[:,1]

# sns.boxplot(x=X)
# plt.title("ecg")
# plt.show()


#calculating z-scores
z = np.abs(stats.zscore(X))
print(z)
# #plot the boxplot
# sns.boxplot(x=z)
# plt.title("z-score ecg")
# plt.show()

#settingup a threshold value
# threshold = 1.15 #first run
threshold = 1.12
# threshold = 1.05
outlier = np.where(z > threshold)
print(outlier)
#indexes of outliers
outlierArray = outlier[0]
print(len(outlierArray))
#
# outliervalue = []
# for i in range(len(outlierArray)):
#     for j in range(len((X))):
#         if outlierArray[i] == j:
#             print(X[j])

z_o = z[(z < threshold)]
# print(len(x_o))
#removing outliers
X = np.delete(X, outlierArray, None)
print(X)
print(len(X))
# print(len(X))
sns.boxplot(x=z_o)
plt.title("outliers removed ecg zscore")
plt.show()

sns.boxplot(x=X)
plt.title("outliers removed ecg")
plt.show()
# #create a csv file to store prerocessed data
# with open('../../../data/data-preprocess/ECG/ECG-awake/ECG-awake_processed_data.csv','w', newline='') as ncsv:
#     newcsv = csv.writer(ncsv)
#     newcsv.writerow(['X','Class']) # Class (drowsyness level)
#     for i in range(len(X)):
#         newcsv.writerow([X[i], Y[i]])
#
# #read the preprocessed data file
# ecg_awake_processed_df = pd.read_csv('../../../data/data-preprocess/ECG/ECG-awake/ECG-awake_processed_data.csv')
# #plot the boxplot
# sns.boxplot(x=ecg_awake_processed_df['X'])
# plt.title("Preprocessed awake")
# plt.show()

# #create a csv file to store zscore data
# with open('../../../data/data-preprocess/ECG/ECG-awake/ECG-awake_zscoreset.csv','w', newline='') as ncsv:
#     newcsv = csv.writer(ncsv)
#     newcsv.writerow(['z']) # z-zscore
#     for i in range(len(z)):
#         newcsv.writerow([z[i]])

