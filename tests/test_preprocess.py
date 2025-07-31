import pandas as pd
from src.preprocess import load_data, get_features_and_target

def test_load_data():
    """Test that data loading returns a non-empty DataFrame."""
    df = load_data('data/iris.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df.columns) == 5

def test_get_features_and_target():
    """Test feature and target separation."""
    df = pd.DataFrame({
        'sepal_length': [1.0], 'sepal_width': [1.0],
        'petal_length': [1.0], 'petal_width': [1.0],
        'species': ['Iris-setosa']
    })
    X, y = get_features_and_target(df)
    assert 'species' not in X.columns
    assert y.name == 'species'
    assert len(X.columns) == 4
    