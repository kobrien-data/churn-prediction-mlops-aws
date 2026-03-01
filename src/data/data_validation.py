import pandas as pd


class DataValidationError(Exception):
    """Raised when data validation fails."""


def validate_schema(df: pd.DataFrame, expected_schema: dict) -> None:
    """
    Validate dataframe schema against an expected schema.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate.
    expected_schema : dict
        Mapping of column name -> expected pandas dtype (e.g. 'float64', 'int64', 'object').

    Raises
    ------
    DataValidationError
        If columns are missing, unexpected, or of incorrect type.
    """
    errors = []

    # Check for missing columns
    missing_cols = [c for c in expected_schema.keys() if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Check for unexpected extra columns (schema drift)
    extra_cols = [c for c in df.columns if c not in expected_schema.keys()]
    if extra_cols:
        errors.append(f"Unexpected columns: {extra_cols}")

    # Check dtypes for columns that exist in both
    for col, expected_dtype in expected_schema.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if actual_dtype != str(expected_dtype):
                errors.append(
                    f"Column '{col}' has dtype '{actual_dtype}' "
                    f"but expected '{expected_dtype}'"
                )

    if errors:
        raise DataValidationError("Schema validation failed: " + "; ".join(errors))


def validate_nulls(df: pd.DataFrame, allow_nulls: dict | None = None) -> None:
    """
    Validate that dataframe has no unexpected nulls.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate.
    allow_nulls : dict, optional
        Mapping of column name -> bool indicating whether nulls are allowed.
        If None, nulls are not allowed in any column.

    Raises
    ------
    DataValidationError
        If unexpected nulls are found.
    """
    if allow_nulls is None:
        allow_nulls = {}

    errors = []

    for col in df.columns:
        col_nulls = df[col].isna().sum()
        if col_nulls == 0:
            continue

        if allow_nulls.get(col, False):
            continue

        errors.append(f"Column '{col}' has {col_nulls} null values but nulls are not allowed.")

    if errors:
        raise DataValidationError("Null validation failed: " + "; ".join(errors))


def validate_ranges(df: pd.DataFrame, expected_ranges: dict) -> None:
    """
    Validate that numeric columns fall within expected ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate.
    expected_ranges : dict
        Mapping of column name -> (min_value, max_value). Either bound can be None to skip that side.

    Raises
    ------
    DataValidationError
        If values fall outside expected ranges.
    """
    errors = []

    for col, (min_val, max_val) in expected_ranges.items():
        if col not in df.columns:
            errors.append(f"Column '{col}' not found for range validation.")
            continue

        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            errors.append(f"Column '{col}' is not numeric and cannot be range-validated.")
            continue

        if min_val is not None:
            below_min = series[series < min_val]
            if not below_min.empty:
                errors.append(
                    f"Column '{col}' has {below_min.size} values below minimum {min_val}."
                )

        if max_val is not None:
            above_max = series[series > max_val]
            if not above_max.empty:
                errors.append(
                    f"Column '{col}' has {above_max.size} values above maximum {max_val}."
                )

    if errors:
        raise DataValidationError("Range validation failed: " + "; ".join(errors))


def run_data_validation(
    df: pd.DataFrame,
    expected_schema: dict,
    expected_ranges: dict | None = None,
    allow_nulls: dict | None = None,
) -> None:
    """
    Run all data validation checks on the given dataframe.

    This function will raise a DataValidationError if any of the checks fail.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate.
    expected_schema : dict
        Mapping of column name -> expected pandas dtype.
    expected_ranges : dict, optional
        Mapping of column name -> (min_value, max_value).
    allow_nulls : dict, optional
        Mapping of column name -> bool indicating whether nulls are allowed.
    """
    validate_schema(df, expected_schema)
    validate_nulls(df, allow_nulls=allow_nulls)

    if expected_ranges:
        validate_ranges(df, expected_ranges)


# -------------------------------
# Defaults for churn dataset
# -------------------------------

CHURN_EXPECTED_SCHEMA: dict[str, str] = {
    "RowNumber": "int64",
    "CustomerId": "int64",
    "Surname": "object",
    "CreditScore": "int64",
    "Geography": "object",
    "Gender": "object",
    "Age": "int64",
    "Tenure": "int64",
    "Balance": "float64",
    "NumOfProducts": "int64",
    "HasCrCard": "int64",
    "IsActiveMember": "int64",
    "EstimatedSalary": "float64",
    "Exited": "int64",
    "Complain": "int64",
    "Satisfaction Score": "int64",
    "Card Type": "object",
    "Point Earned": "int64",
}

CHURN_EXPECTED_RANGES: dict[str, tuple[float | int | None, float | int | None]] = {
    "RowNumber": (1, None),
    "CustomerId": (0, None),
    "CreditScore": (300, 900),
    "Age": (18, 120),
    "Tenure": (0, 15),
    "Balance": (0.0, None),
    "NumOfProducts": (1, 10),
    "HasCrCard": (0, 1),
    "IsActiveMember": (0, 1),
    "EstimatedSalary": (0.0, None),
    "Exited": (0, 1),
    "Complain": (0, 1),
    "Satisfaction Score": (1, 5),
    "Point Earned": (0, None),
}

CHURN_ALLOW_NULLS: dict[str, bool] = {}


def validate_churn_csv(
    csv_path: str = "data/raw/Customer-Churn-Records.csv",
) -> pd.DataFrame:
    """
    Load the churn CSV and run all validations.

    Returns the validated dataframe if all checks pass, otherwise raises DataValidationError.
    """
    df = pd.read_csv(csv_path)
    run_data_validation(
        df=df,
        expected_schema=CHURN_EXPECTED_SCHEMA,
        expected_ranges=CHURN_EXPECTED_RANGES,
        allow_nulls=CHURN_ALLOW_NULLS,
    )
    return df


