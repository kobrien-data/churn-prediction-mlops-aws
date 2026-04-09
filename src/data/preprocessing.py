import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import argparse
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(file_path)


def drop_unnecessary_columns(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    """Drop specified columns from the DataFrame."""
    return df.drop(columns=columns_to_drop)


def encode_categorical_variables(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    """Encode categorical variables using one-hot encoding."""
    return pd.get_dummies(df, columns=categorical_columns)


def scale_numerics(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'Exited' in numeric_cols:
        numeric_cols.remove('Exited')

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """Split the data into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance the training data."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def save_preprocessed_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str
):
    """Save the preprocessed data to the given output path."""
    train_dir = os.path.join(output_path, 'train')
    test_dir = os.path.join(output_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    X_train.to_csv(os.path.join(train_dir, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(train_dir, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(test_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(test_dir, 'y_test.csv'), index=False)


def main(input_path: str, output_path: str) -> None:
    df = load_data(input_path)
    df = drop_unnecessary_columns(df, columns_to_drop=['RowNumber', 'CustomerId', 'Surname', 'Complain'])
    df = encode_categorical_variables(df, categorical_columns=['Geography', 'Gender', 'Card Type'])
    df, _ = scale_numerics(df)
    X_train, X_test, y_train, y_test = split_data(df, target_column='Exited')
    X_train, y_train = apply_smote(X_train, y_train)
    save_preprocessed_data(X_train, y_train, X_test, y_test, output_path)
    print("Preprocessing complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/opt/ml/processing/input/Customer-Churn-Records.csv')
    parser.add_argument('--output-path', type=str, default='/opt/ml/processing/output')
    args = parser.parse_args()
    main(args.input_path, args.output_path)
