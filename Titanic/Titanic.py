###import numpy as np
import pandas as pd
from sklearn import tree

print "load data..."

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print train.info(), test.info()

train["Age"] = train["Age"].fillna(train["Age"].median());
test["Age"] = test["Age"].fillna(train["Age"].median());

train["Sex"] = train["Sex"].apply(lambda x: 1 if x == "male" else 0)
test["Sex"] = test["Sex"].apply(lambda x: 1 if x =="male" else 0)

feature = ["Age", "Sex"]

dt = tree.DecisionTreeClassifier()
dt = dt.fit(train[feature], train["Survived"])

predict_data = dt.predict(test[feature])

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predict_data
})

submission.to_csv("data/decision_tree.csv", index = False)