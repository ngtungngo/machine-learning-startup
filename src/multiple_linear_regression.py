import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
df = pd.read_csv("../resources/FuelConsumption.csv")

# take a look at the dataset
print(df.head())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

def fitAndPredict(colname: []):
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    cols = ['ENGINESIZE','CYLINDERS']
    cols.extend(colname)
    x = np.asanyarray(train[cols])
    y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit (x, y)
    # The coefficients
    print ('Coefficients: ', regr.coef_)

    # Prediction
    print("================================")
    print("Prediction " + str(colname) + ": ")
    y_hat= regr.predict(test[cols])
    x = np.asanyarray(test[cols])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    print("Residual sum of squares: %.2f"
        % np.mean((y_hat - y) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x, y))

fitAndPredict(['FUELCONSUMPTION_COMB'])
fitAndPredict(['FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY'])
