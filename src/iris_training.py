import mlflow
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class IrisTraining:
    def __init__(self):
        self.df = None
        self.data = {}
        self.model = None
        mlflow.sklearn.autolog()

    def training_pipeline(self):
        self.data_extract()
        self.data_transform()
        self.data_training()
        y_pred = self.data_predict(self.data["y_train"])
        self.data_evaluation(self.data["y_test"], y_pred)

    def data_extract(self):
        self.df = datasets.load_iris()

    def data_transform(self):
        X = self.df.data[:, [2, 3]]
        y = self.df.target
        X_train, X_test, self.data["y_train"], self.data["y_test"] = train_test_split(X, y, test_size=0.3, random_state=0)
        sc = StandardScaler()
        sc.fit(X_train)
        self.data["x_train"] = sc.transform(X_train)
        self.data["x_test"] = sc.transform(X_test)

    def data_training(self):
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.model.fit(self.data["x_train"], self.data["y_train"])
    
    def data_predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

    def data_evaluation(self, y_test, y_pred):
        print ('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))