from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# LOAD DATA SET 
iris = datasets.load_iris()

# PRINT 
# print(iris.keys())
# print(iris.DESCR)
features = iris.data
labels = iris.target

# TRAIN 
clf = KNeighborsClassifier()
clf.fit(features, labels)

# PREDICT 
pred = clf.predict([[11,1,1,1]])
print(pred)

