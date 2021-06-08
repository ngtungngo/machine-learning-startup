import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


df = pd.read_csv("../resources/FuelConsumption.csv")

# take a look at the dataset
print(df.head().to_string())
print(df.info)
print(df.columns)
print(df.dtypes)

# summarize the data
print("Descriptive exploration: ")
print(df.describe().to_string())

#
print("Features: ")
cdf = df[['MODELYEAR', 'ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9).to_string())
#
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB', 'MODELYEAR']]
viz.hist()
plt.show()
#
plt.scatter(cdf.MODELYEAR, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("MODELYEAR")
plt.ylabel("Emission")
plt.show()
# #
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='red')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()
# #
# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Cylinders")
# plt.ylabel("Emission")
# plt.show()

# Split dataset into train and test sets:
# 80% of the entire data for training, and the 20% for testing
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
#
# plt.scatter(train.CYLINDERS, train.CO2EMISSIONS,  color='blue')
# plt.xlabel("Cylinders")
# plt.ylabel("Emission")
# plt.show()
#Using sklearn package to model data
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

#Plot outputs
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
print('**** end ****')

