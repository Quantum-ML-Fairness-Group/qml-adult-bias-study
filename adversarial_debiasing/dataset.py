import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_adult_dataset():
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "gender",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    data = pd.read_csv(url, names = columns, sep = ",\s*", engine = "python")

    data = data.replace("?", pd.NA).dropna()

    return data

def preprocess_data(df):
    df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
    
    df['gender'] = df['gender'].apply(lambda x: 1 if x.strip() == 'Male' else 0)
    
    X = df.drop(columns=['income'])
    y = df['income'].values
    A = df['gender'].values
    
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col])
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    return X.values, y, A

def get_train_test_split(test_size=0.2, random_state=42):
    df = load_adult_dataset()
    X, y, A = preprocess_data(df)
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, A_train, A_test