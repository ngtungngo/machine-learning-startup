import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

my_data = pd.read_csv("../resources/drug200.csv", delimiter=",")
print(my_data[0:5])

#What is the size of data?
print("size of data: " + str(my_data.shape))

#Pre-processing
#Remove the column containing the target name since it doesn't contain numeric values.
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

#As you may figure out, some features in this dataset are categorical,
# such as Sex or BP. Unfortunately, Sklearn Decision Trees does not handle categorical variables.
# We can still convert these features to numerical values using pandas.get_dummies()
# to convert the categorical variable into dummy/indicator variables.

from sklearn import preprocessing
# F =>  0, M => 1
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

# LOW => 0, NORMAL: 1, HIGH: 2
le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

# NORMAL => 0, HIGH: 3
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

print(X[0:5])


y = my_data["Drug"]
print(y[0:5])

#Setting up the Decision Tree
#Now train_test_split will return 4 different parameters. We will name them:
#X_trainset, X_testset, y_trainset, y_testset

#The train_test_split will need the parameters:
# X, y, test_size=0.3, and random_state=3.
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))

#Modeling

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(drugTree) # it shows the default parameters

drugTree.fit(X_trainset,y_trainset)

print(drugTree)

#Prediction
predTree = drugTree.predict(X_testset)

print (predTree [0:5])
print (y_testset [0:5])

#Evaluation
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#Visualization
from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')

