import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Data generation
def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    # Create random data and store in feature matrix X and response vector y.
    X, y = make_blobs(n_samples=numSamples, centers=centroidLocation,
                      cluster_std=clusterDeviation)

    # Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(X)
    return X, y


X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)

#Modeling
epsilon = 0.3
minimumSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_
print(labels)

#Distinguish outliers

# Firts, create an array of booleans using the labels from db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
print(core_samples_mask)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)

# Remove repetition in labels by turning it into a set.
unique_labels = set(labels)
print(unique_labels)

# Data visualization
# Create colors for the clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
# Plot the points with colors
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    # Plot the datapoints that are clustered
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

    # Plot the outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)
plt.show()

from sklearn.cluster import KMeans
k = 3
k_means3 = KMeans(init = "k-means++", n_clusters = k, n_init = 12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(k), colors):
    my_members = (k_means3.labels_ == k)
    plt.scatter(X[my_members, 0], X[my_members, 1],  c=col, marker=u'o', alpha=0.5)
plt.show()

#Weather Station Clustering using DBSCAN & scikit-learn
#Environment Canada Monthly Values for July - 2015
#    Name in the table	Meaning
#Stn_Name	Station Name
#Lat	Latitude (North+, degrees)
#Long	Longitude (West - , degrees)
#Prov	Province
#Tm	Mean Temperature (°C)
#DwTm	Days without Valid Mean Temperature
#D	Mean Temperature difference from Normal (1981-2010) (°C)
#Tx	Highest Monthly Maximum Temperature (°C)
#DwTx	Days without Valid Maximum Temperature
#Tn	Lowest Monthly Minimum Temperature (°C)
#DwTn	Days without Valid Minimum Temperature
#S	Snowfall (cm)
#DwS	Days without Valid Snowfall
#S%N	Percent of Normal (1981-2010) Snowfall
#P	Total Precipitation (mm)
#DwP	Days without Valid Precipitation
#P%N	Percent of Normal (1981-2010) Precipitation
#S_G	Snow on the ground at the end of the month (cm)
#Pd	Number of days with Precipitation 1.0 mm or more
#BS	Bright Sunshine (hours)
#DwBS	Days without Valid Bright Sunshine
#BS%	Percent of Normal (1981-2010) Bright Sunshine
#HDD	Degree Days below 18 °C
#CDD	Degree Days above 18 °C
#Stn_No	Climate station identifier (first 3 digits indicate drainage basin, last 4 characters are for sorting alphabetically).
#NA	Not Available

#1-Download data
import csv
import pandas as pd
import numpy as np

#2- Load the dataset
filename='../resources/weather-stations20140101-20141231.csv'

#Read csv
pdf = pd.read_csv(filename)
print(pdf.head(5))

#3-Cleaning: Let's remove rows that don't have any value in the Tm field.
pdf = pdf[pd.notnull(pdf["Tm"])]
pdf = pdf.reset_index(drop=True)
print(pdf.head(5))

#4-Visualization
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = (14,10)

llon=-140
ulon=-50
llat=40
ulat=65

pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) &(pdf['Lat'] < ulat)]

my_map = Basemap(projection='merc',
                 resolution = 'l', area_thresh = 1000.0,
                 llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
                 urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To collect data based on stations

xs,ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm']= xs.tolist()
pdf['ym'] =ys.tolist()

#Visualization1
for index,row in pdf.iterrows():
    #   x,y = my_map(row.Long, row.Lat)
    my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
#plt.text(x,y,stn)
plt.show()
