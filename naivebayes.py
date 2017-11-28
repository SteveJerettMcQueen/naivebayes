import numpy as np

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from util import load_data

################################################################################

dir_label = [
    ['badeer-r', 1], ['benson-r', 1], ['blair-l', 0],
    ['cash-m', 0], ['corman-s', 1], ['hain-m', 1]]

dataset = load_data(dir_label)

X = np.array(dataset[0])
y = dataset[1]
print("Data Set: ", len(X))

# Sci-Kit Learn Naive Baye's Classifiers
# Train/Test split model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 50)
print("X Tranining Set: ", len(X_train), "; X Test Set: ", len(X_test))
print("Y Tranining Set: ", len(y_train), "; Y Test Set: ", len(y_test))

gauss = GaussianNB().fit(X_train, y_train)
y_pred = gauss.predict(X_test)
print("Gaussian Accuracy: ", "{0:.4f}".format(metrics.accuracy_score(y_test, y_pred)))
print "Confusioin Matrix: "
print metrics.confusion_matrix(y_test, y_pred)
print "Classification Report: "
print classification_report(y_test, y_pred)
print("---------------------------------------------------------------------")
    