#IMPORTING LIBRARIES

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy
iris=datasets.load_iris()


#SETTING FEATURES AND LABELS TO CLASSIFY
features=iris.data
labels=iris.target


#FITTING DATA TO KMEANS CLASSIFIER
clf=KNeighborsClassifier()
clf.fit(features, labels)


#INPUT SEPAL LENGTH, SEPAL WIDTH, PETAL LENGTH AND PETAL WIDTH
print("enter features:")
a=input()
b=input()
c=input()
d=input()

#OUTPUT 
preds=clf.predict([[a,b,c,d]])
print(preds)
