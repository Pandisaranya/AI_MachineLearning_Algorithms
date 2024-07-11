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
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
#classifier.fit(x_train,y_train)
#print(classifier.coef_)
#print(classifier.intercept_)
from sklearn.model_selection import GridSearchCV
para_grid={'criterion':['gini','entropy','log_loss'],'max_features':['sqrt','log2']}
grid=GridSearchCV(RandomForestClassifier(),para_grid,refit=True)
grid.fit(x_train,y_train)
y_pred=grid.predict(x_test)
from sklearn.metrics import confusion_matrix
score=confusion_matrix(y_test,y_pred)
print("Score is ",score)
from sklearn.metrics import classification_report
clf_report=classification_report(y_test,y_pred)
print("clf_report ",clf_report)
best_para=grid.best_params_
res=grid.cv_results_
table=p.DataFrame.from_dict(res)
print("best parameters",best_para)
#print("result",res)
#print(table)
import pickle as pkl
pkl.dump(classifier,open("Social_Network_Ads-Logistic Regression-model creation.sav","wb"))
print("file witten")

