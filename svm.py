import pandas as pd
data = pd.read_excel(r'E:\courses\Machine Learning\SVM\fruits.xlsx')
print(data.head)
from sklearn.model_selection import train_test_split
training_set,test_set = train_test_split(data,test_size=0.2,random_state=1)
print("Training Set\n")
print(training_set)
print("Test Set\n")
print(test_set)
X_train = training_set.iloc[:,0:2].values
Y_train = training_set.iloc[:,2].values
X_test = test_set.iloc[:,0:2].values
Y_test = test_set.iloc[:,2].values
print("-------------X_train---------------")
print(X_train)
print("-------------Y_train---------------")
print(Y_train)
print("------------X_test-----------------")
print(X_test)
print("------------Y_test-----------------")
print(Y_test)
#Starting SVM
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state = 1)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
print("Prediction is :",Y_pred)
test_set["prediction"] = Y_pred
print(test_set)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)

print("*******KNN CLASSIFICATION********")
#starting knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
print(knn.fit(X_train,Y_train))
knn.score(X_test,Y_test)
pred=knn.predict(X_test)
print(pred)
cm1 = confusion_matrix(Y_test,pred)
accuracy1 = float(cm1.diagonal().sum())/len(Y_test)
print(accuracy1)

