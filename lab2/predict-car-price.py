import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self):
        np.random.seed(2)

        n = len(self.df)

        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)

        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]

        df_train = df_shuffled.iloc[:n_train].copy()

        y_train = np.log1p(df_train.msrp.values)
        del df_train['msrp']

        X_train = self.prepare_X(df_train)

        w_0, w = self.linear_regression(X_train, y_train)

        df_top5 = self.prepare_X(self.df.head())

        y_log = w_0 + df_top5.dot(w)

        df_ans = self.df.head().copy()
        df_ans['msrp_pred'] = np.expm1(y_log)

        print(df_ans[['make', 'model', 'engine_cylinders', 'transmission_type', 'driven_wheels', 'number_of_doors', 'market_category', 'vehicle_size', 'vehicle_style', 'highway_mpg', 'city_mpg', 'popularity', 'msrp', 'msrp_pred']].head().to_markdown())

    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
    
        return w[0], w[1:]

    def prepare_X(self, df_prep):
        base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        df_num = df_prep[base]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X    


def test() -> None:
    carprice = CarPrice()
    carprice.trim()
    carprice.validate()

if __name__ == "__main__":
    test()