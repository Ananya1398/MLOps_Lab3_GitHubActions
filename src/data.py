import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():
    """
    Load the Health dataset and return features, target values, and target names.
    Returns:
        X (numpy.ndarray): The features of the Health dataset.
        y (numpy.ndarray): Encoded target values (integers).
        target_names (list): Original class names (status categories).
    """
    df = pd.read_csv("../data/health_data.csv")

    X = df.drop("status", axis=1).values

    le = LabelEncoder()
    y = le.fit_transform(df["status"])
    target_names = list(le.classes_)

    return X, y, target_names

def split_data(X, y):
    """
    Split the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12
    )
    return X_train, X_test, y_train, y_test
