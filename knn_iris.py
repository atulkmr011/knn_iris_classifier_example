#knn classifier example using iris dataset
# first load important modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Now loading datasets

iris = datasets.load_iris()

# now printing the descriptions

#print(iris.DESCR) 

features = iris.data
labels = iris.target

# now training the classifier

clf = KNeighborsClassifier()
clf.fit(features, labels)

# now predict the label if the features are these

preds = clf.predict([[ 3.1, 5.1, 1, 51]]) # give the input here as required
print(preds) 


#output
# If 0 then Iris-Setosa
# If 1 thenIris-Versicolour
# If 2 thenIris-Virginica
