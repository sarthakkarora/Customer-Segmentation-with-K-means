# Importing necessary packages
import pandas as pd  # Working with data
import numpy as np  # Working with arrays
import matplotlib.pyplot as plt  # Visualization
import seaborn as sb  # Visualization
from mpl_toolkits.mplot3d import Axes3D  # 3D plot
from termcolor import colored as cl  # Text customization

from sklearn.preprocessing import StandardScaler  # Data normalization
from sklearn.cluster import KMeans  # K-means algorithm

# Setting plot parameters
plt.rcParams['figure.figsize'] = (20, 10)
sb.set_style('whitegrid')

# Importing data
df = pd.read_csv('cust_seg.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.set_index('Customer Id', inplace=True)

# Displaying the first few rows of the dataframe
print(cl(df.head(), attrs=['bold']))

# Data Analysis

# Age distribution
print(cl(df['Age'].describe(), attrs=['bold']))

sb.histplot(df['Age'], color='orange', kde=True)
plt.title('Age Distribution', fontsize=18)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig('age_distribution.png')
plt.show()

# Credit card default cases
sb.countplot(df['Defaulted'], palette=['coral', 'deepskyblue'], edgecolor='darkgrey')
plt.title('Credit Card Default Cases (1) and Non-Default Cases (0)', fontsize=18)
plt.xlabel('Default Value', fontsize=16)
plt.ylabel('Number of People', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig('default_cases.png')
plt.show()

# Age vs Income
sb.scatterplot(x='Age', y='Income', data=df, color='deepskyblue', s=150, alpha=0.6, edgecolor='b')
plt.title('Age vs. Income', fontsize=18)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Income', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig('age_income.png')
plt.show()

# Years Employed vs Income
area = df['DebtIncomeRatio'] ** 2

sb.scatterplot(x='Years Employed', y='Income', data=df, s=area, alpha=0.6, edgecolor='white', hue='Defaulted', palette='spring')
plt.title('Years Employed vs. Income', fontsize=18)
plt.xlabel('Years Employed', fontsize=16)
plt.ylabel('Income', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left', fontsize=14)

plt.savefig('years_income.png')
plt.show()

# Data Processing
X = df.values
X = np.nan_to_num(X)  # Replace NaN with zero and infinity with large finite numbers

scaler = StandardScaler()
cluster_data = scaler.fit_transform(X)
print(cl('Cluster data samples:', attrs=['bold']), cluster_data[:5])

# Modeling
clusters = 3
kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=12)
kmeans.fit(cluster_data)

labels = kmeans.labels_
print(cl(labels[:100], attrs=['bold']))

# Adding the cluster number to the original dataframe
df['Cluster'] = labels
print(cl(df.head(), attrs=['bold']))

# Displaying the mean values for each cluster
print(cl(df.groupby('Cluster').mean(), attrs=['bold']))

# Visualization of Age vs. Income with clusters
area = np.pi * (df['Edu']) ** 4

sb.scatterplot(x='Age', y='Income', data=df, s=area, hue='Cluster', palette='spring', alpha=0.6, edgecolor='darkgrey')
plt.title('Age vs. Income (Clustered)', fontsize=18)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Income', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left', fontsize=14)

plt.savefig('clustered_age_income.png')
plt.show()

# 3D plot for Education, Age, and Income
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Edu'], df['Age'], df['Income'], c=df['Cluster'], s=200, cmap='spring', alpha=0.5, edgecolor='darkgrey')
ax.set_xlabel('Education', fontsize=16)
ax.set_ylabel('Age', fontsize=16)
ax.set_zlabel('Income', fontsize=16)

plt.savefig('3d_plot.png')
plt.show()
