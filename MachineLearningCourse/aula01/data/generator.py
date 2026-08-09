import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

def get_iris_data(num_features=2):
    """
    Loads the Iris dataset and prepares it for binary classification.
    - Selects the first two classes (Setosa and Versicolor).
    - Selects the specified number of features.
    - Converts labels (0, 1) to (-1, 1).
    - Adds a bias term to the data.
    """
    iris = load_iris()
    X = iris.data[:100, :num_features]  # First 100 samples, num_features
    y = iris.target[:100]  # Labels for the first 100 samples (0 and 1)

    # Convert labels from {0, 1} to {-1, 1}
    y = np.where(y == 0, -1, 1)

    return X, y

def Gen_data():
    """
    Provides 2D Iris data for classification.
    Returns:
        - data (np.array): Shape (3, 100). Rows are [feature1, feature2, bias].
        - labels (np.array): Shape (100,). Labels are -1 or 1.
    """
    X, y = get_iris_data(num_features=2)
    
    # Add bias term and transpose
    data = np.vstack([X.T, np.ones(X.shape[0])])
    
    return data, y

def Data_3D():
    """
    Provides 3D Iris data for classification.
    Returns:
        - data (pd.DataFrame): 100 rows, 3 columns (features).
        - labels (np.array): Shape (100,). Labels are -1 or 1.
    """
    X, y = get_iris_data(num_features=3)
    
    data = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length'])
    
    return data, y
