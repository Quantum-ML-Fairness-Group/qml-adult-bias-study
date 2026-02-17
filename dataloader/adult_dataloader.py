import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
    'marital-status', 'occupation', 'relationship', 'race', 'sex', 
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    if not os.path.exists('data/adult/adult.data'):
         print("Please ensure the data file path is correct (data/adult/adult.data)")
         assert os.path.exists('data/adult/adult.data'), "Data file not found"
    if not os.path.exists('data/adult/adult.test'):
         print("Please ensure the data file path is correct (data/adult/adult.test)")
         assert os.path.exists('data/adult/adult.test'), "Data file not found"
    
    train_df = pd.read_csv('data/adult/adult.data', names=columns, skipinitialspace=True, na_values='?')
    test_df = pd.read_csv('data/adult/adult.test', names=columns, skipinitialspace=True, na_values='?', skiprows=1)

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    train_df['income'] = train_df['income'].str.replace('.', '', regex=False).map({'<=50K': 0, '>50K': 1})
    test_df['income'] = test_df['income'].str.replace('.', '', regex=False).map({'<=50K': 0, '>50K': 1})

    sensitive_train = train_df['sex'].map({'Female': 0, 'Male': 1}).values
    sensitive_test = test_df['sex'].map({'Female': 0, 'Male': 1}).values

    y_train = train_df['income'].values
    y_test = test_df['income'].values

    X_train = train_df.drop(columns=['income', 'sex'])
    X_test = test_df.drop(columns=['income', 'sex'])

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)

    X_train = X_train.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)

    return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test