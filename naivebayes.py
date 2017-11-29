import os
import datetime as dt
import numpy as np

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from util import load_data

################################################################################

# Classification data
target_names=['Female', 'Male']

dir_label = [
    ['badeer-r', 1], ['benson-r', 1], ['blair-l', 0],
    ['cash-m', 0], ['corman-s', 1], ['hain-m', 1]]

dataset = load_data(dir_label)

X = np.array(dataset[0])
y = dataset[1]

# Sci-Kit Learn Naive Baye's Classifiers
# Train/Test split model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 50)

# Gaussian
gauss = GaussianNB().fit(X_train, y_train)
y_pred_gauss = gauss.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred_gauss)
cfm = metrics.confusion_matrix(y_test, y_pred_gauss)
report = classification_report(y_test, y_pred_gauss, target_names=target_names)
    
# Bernoulli
bern = BernoulliNB().fit(X_train, y_train)
y_pred_bern = bern.predict(X_test)
acc2 = metrics.accuracy_score(y_test, y_pred_bern)
cfm2 = metrics.confusion_matrix(y_test, y_pred_bern)
report2 = classification_report(y_test, y_pred_bern, target_names=target_names)

# Write to file
filename = 'dataset/results.txt'
file_exists = os.path.exists(filename)
append_write = 'a' if(file_exists) else 'w'
f = open(filename, append_write)

# Write Data information
f.write("---------------------------------------------------------------------\n")
f.write("Date: " + str(dt.datetime.now().strftime("%m-%d-%Y")) + '\n')
f.write("Data Set: " + str(len(X)) + "\n")
f.write("X Tranining Set: " + str(len(X_train)) + " ; X Test Set: " + str(len(X_test)) + "\n")
f.write("Y Tranining Set: " + str(len(y_train)) + " ; Y Test Set: " + str(len(y_test)) + "\n")
f.write("\n")

# Write Gaussian metrics
f.write("Gaussian Accuracy: " + "{0:.4f}".format(acc) + "\n")
f.write("Confusioin Matrix: " + str(cfm.ravel()) + "\n")
f.write("Classification Report:\n" + report + "\n")

# Write Bernoulli metrics
f.write("Bernoulli Accuracy: " + "{0:.4f}".format(acc2) + "\n")
f.write("Confusioin Matrix: " + str(cfm2.ravel()) + "\n")
f.write("Classification Report:\n" + report2 + "\n")
f.write("\n")
f.close()
