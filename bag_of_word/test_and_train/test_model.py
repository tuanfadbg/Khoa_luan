from joblib import dump, load
from numpy import genfromtxt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MODEL_FILENAME = "model/logistic_regression/8_9000_items.joblib"

data = pd.read_csv("data/data_8_9000_items.csv", header=None, dtype=object).values.astype(float)
# data = genfromtxt("data/data_8_9000_items.csv")
is_covid = genfromtxt("data/data_8_9000_items_value.csv")

print(data.shape)
print(is_covid.shape)
X_train, X_test, y_train, y_test = train_test_split(data, is_covid, test_size=0.05, random_state=1)
clf = LogisticRegression(random_state=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))

dump(clf, MODEL_FILENAME)
