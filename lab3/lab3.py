import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        #print(self.pima['pedigree'].value_counts())
        #print(self.pima.describe())
        #elf.plot_correlation_matrix()
        self.X_test = None
        self.y_test = None

    def plot_correlation_matrix(self):
        corrmat = self.pima.corr()
        top_corr_features = corrmat.index
        plt.figure(figsize=(10,10))
        g=sns.heatmap(self.pima[top_corr_features].corr(),annot=True,cmap="RdYlGn")
        plt.show()

    def define_feature(self):
        feature_cols = ['pregnant', 'glucose', 'bmi', 'age', 'pedigree']
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y
    
    def perform_min_max_scaling(self, X_train):
        scaler = MinMaxScaler(feature_range = (0,1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        self.X_test = scaler.transform(self.X_test)
        return X_train

    def train(self):
        # split X and y into training and testing sets
        X, y = self.define_feature()
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)

        X_train = self.perform_min_max_scaling(X_train)
        # train a logistic regression model on the training set
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self):
        model = self.train()
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":
    classifer = DiabetesClassifier()
    result = classifer.predict()
    print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix=${con_matrix}")
    
