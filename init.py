import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Size of dataset , no of rows and no of attributes
# print(dataset.shape)

# Peek at the data
# print(dataset.head(20))

# Print statistical data: count, mean, std, etc
# print(dataset.describe())

# Class Distribution: How many instances of Iris-setosa, Iris-versicolor, etc
# print(dataset.groupby('class').size())

# # box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# Histogram (Some attributes follow Gaussian Bell curve)
# dataset.hist()
# plt.show()

# scatter plot matrix: To view how data is scattered (using dots).
# scatter_matrix(dataset)
# plt.show()

# Create validation dataset: 20% of data that model doesnt see
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = \
model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    # Evaluate each algo 10 times (10 fold cross validation)
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)

# Accuracy is just total_correct_predictions / total_predictions
print("Accuracy: ", accuracy_score(Y_validation, predictions))

# But accuracy doesn't display if all classes were evaluated correctly
# or only some were evaluated correctly. Enter confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(Y_validation, predictions))
# Read it horizontally  then vertically
# ex:
# men classified as men: 3
# women classified as women: 4
# men classified as women: 2
# woman classified as men: 1


# confusion_matrix:
# 		men	women
# men	3	1
# women	2	4

print("Classification Report:")
print(classification_report(Y_validation, predictions))
