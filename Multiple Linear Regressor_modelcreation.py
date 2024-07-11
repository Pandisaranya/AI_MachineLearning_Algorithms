import pandas as p
dataset=p.read_csv("50_Startups.csv")
dataset=p.get_dummies(dataset,drop_first=True)
print(dataset.info())
independent_input=dataset[["R&D Spend","Administration","Marketing Spend","State_Florida","State_New York"]]
dependent_output=dataset[["Profit"]]

from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(independent_input,dependent_output,test_size=0.35,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,x_test)

y_pred=reg.predict(y_train)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print("alogorithm accurace",score)
