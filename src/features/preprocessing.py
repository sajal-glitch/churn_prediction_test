import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

NUMERIC_COLS = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]
GENDER_COL = ["Gender"]
ORDINAL_COLS = ["Subscription Type", "Contract Length"]

SUBSCRIPTION_ORDER = [["Basic", "Standard", "Premium"]]
CONTRACT_ORDER = [["Monthly", "Quarterly", "Annual"]]


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer for the churn pipeline.

    Three transformers:
    - numeric: median imputation + RobustScaler
    - gender: most-frequent imputation + OrdinalEncoder (Female=0, Male=1)
    - ordinal_cat: most-frequent imputation + OrdinalEncoder with explicit ordering

    remainder='drop' discards CustomerID if present.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    gender_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[["Female", "Male"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=SUBSCRIPTION_ORDER + CONTRACT_ORDER,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLS),
            ("gender", gender_transformer, GENDER_COL),
            ("ordinal_cat", ordinal_transformer, ORDINAL_COLS),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return human-readable feature names from a fitted ColumnTransformer."""
    raw_names = preprocessor.get_feature_names_out()
    cleaned = []
    for name in raw_names:
        # strip transformer prefix (e.g. "numeric__Age" -> "Age")
        if "__" in name:
            cleaned.append(name.split("__", 1)[1])
        else:
            cleaned.append(name)
    return cleaned
