import pandas as pd 

def load_data(path):
    """Loads the Iris dataset without headers."""
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv(path, header=None, names=column_names)
    return df

def get_features_and_target(df):
    """Separates features and target."""
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    return X, y