from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy
iris=datasets.load_iris()

features=iris.data
labels=iris.target

clf=KNeighborsClassifier()
clf.fit(features, labels)

print("enter features:")
a=input()
b=input()
c=input()
d=input()


preds=clf.predict([[a,b,c,d]])
print(preds)