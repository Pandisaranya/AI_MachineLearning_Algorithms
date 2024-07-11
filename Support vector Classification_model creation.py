import pandas as p
dataset=p.read_csv("Social_Network_Ads.csv")
dataset=p.get_dummies(dataset,drop_first=True)
#print(dataset.info())
dataset=dataset.drop("User ID",axis=1)
#print(dataset["Purchased"].value_counts())
ind_input=dataset[["Age","EstimatedSalary","Gender_Male"]]
dep_output=dataset[["Purchased"]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(ind_input,dep_output,test_size=0.35,random_state=0)
from sklearn.svm import SVC
classifier=SVC(kernel="linear",gamma="scale")
classifier.fit(x_train,y_train)
#print(classifier.coef_)
#print(classifier.intercept_)
y_pred=classifier.predict(x_test)
#print(y_pred)
from sklearn.metrics import confusion_matrix
score=confusion_matrix(y_test,y_pred)
print("Score is ",score)
from sklearn.metrics import classification_report
clf_report=classification_report(y_test,y_pred)
print("clf_report ",clf_report)

