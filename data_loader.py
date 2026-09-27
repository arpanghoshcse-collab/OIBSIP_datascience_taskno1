"""
Data loading and preprocessing module for Iris Flower Classification.
"""

from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]
TARGET_NAME = "species"


def load_iris_dataframe(save_csv_path: str = None) -> pd.DataFrame:
    """
    Load the Iris dataset from sklearn and format as a pandas DataFrame.
    
    Args:
        save_csv_path: Optional path to save raw dataset as CSV.
        
    Returns:
        pd.DataFrame containing 4 features, target code, and species name.
    """
    iris_raw = load_iris(as_frame=True)
    df = iris_raw.frame.copy()
    
    # Standardize column names
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species_id"
    ]
    
    # Map species ID to human-readable target labels
    target_names = iris_raw.target_names
    df[TARGET_NAME] = df["species_id"].map(lambda x: target_names[x])
    
    if save_csv_path:
        df.to_csv(save_csv_path, index=False)
        print(f"Dataset successfully saved to: {save_csv_path}")
        
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Compute essential exploratory statistics for data validation.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        dict containing shape, null counts, dtypes, and class counts.
    """
    return {
        "shape": df.shape,
        "null_counts": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "class_distribution": df[TARGET_NAME].value_counts().to_dict(),
        "descriptive_stats": df[FEATURE_NAMES].describe().to_dict()
    }


def split_features_target(
    df: pd.DataFrame, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets with stratification.
    
    Args:
        df: Input DataFrame.
        test_size: Fraction of samples assigned to test set (default 0.20).
        random_state: Seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_iris_dataframe(save_csv_path="data/iris.csv")
    summary = get_data_summary(df)
    print("Iris Data Summary:")
    print(f"Shape: {summary['shape']}")
    print(f"Null count total: {sum(summary['null_counts'].values())}")
    print(f"Class distribution: {summary['class_distribution']}")
