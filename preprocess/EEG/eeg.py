import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import csv
import matplotlib.pyplot as plt

eeg_df = pd.read_csv('../../data/relativeEEG.csv')

print(eeg_df)
eegSet = eeg_df.values
X = eegSet[:,0:4]
print(X)
print(len(X))
Y = eegSet[:,4]
# sns.boxplot(x=X)
# plt.title("eeg")
# plt.show()
#calculating z-scores
z = np.abs(stats.zscore(X))
print(z)
# #plot the boxplot
# sns.boxplot(x=z)
# plt.title("z-score eeg")
# plt.show()
#settingup a threshold value
threshold = 1.15 #first run
threshold = 1.53
outlier = np.where(z > threshold)
print(outlier)
#removing outliers
z_o = z[(z < threshold).all(axis=1)]
sns.boxplot(x=z_o)
plt.title("outliers removed eeg zscore")
plt.show()
x_o = X[(z < threshold).all(axis=1)]
print(len(x_o))
sns.boxplot(x=x_o)
plt.title("outliers removed eeg")
plt.show()
#create a csv file to store prerocessed data
with open('../../data/data-preprocess/eeg-processedrelative_data.csv','w', newline='') as ncsv:
    newcsv = csv.writer(ncsv)
    newcsv.writerow(['Theta', 'Alpha', 'Beta', 'Delta', 'Class']) # Class (drowsyness level)
    for i in range(len(x_o)):
        newcsv.writerow([x_o[i][0], x_o[i][1], x_o[i][2], x_o[i][3], Y[i]])
#indexes of outliers
# outlierArray = outlier[0]
# print(len(outlierArray))
#
# outliervalue = []
# for i in range(len(outlierArray)):
#     for j in range(len((X))):
#         if outlierArray[i] == j:
#             print(X[j])
# #removing outliers
# X = np.delete(X, outlierArray, None)
# print(X)
# print(len(X))

# #create a csv file to store prerocessed data
# with open('../../../data/data-preprocess/ECG/ECG-awake/ECG-awake_processed_data.csv','w', newline='') as ncsv:
#     newcsv = csv.writer(ncsv)
#     newcsv.writerow(['X','Class']) # Class (drowsyness level)
#     for i in range(len(X)):
#         newcsv.writerow([X[i], Y[i]])
#
#read the preprocessed data file
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

