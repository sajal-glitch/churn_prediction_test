import pandera as pa
from pandera.typing import Series
import pandas as pd


class ChurnRawSchema(pa.SchemaModel):
    """Pandera schema for raw churn data (tolerates the single all-null row)."""

    CustomerID: Series[float] = pa.Field(nullable=True, coerce=True)
    Age: Series[float] = pa.Field(ge=18, le=65, nullable=True, coerce=True)
    Gender: Series[object] = pa.Field(
        isin=["Male", "Female"], nullable=True
    )
    Tenure: Series[float] = pa.Field(ge=1, le=60, nullable=True, coerce=True)
    Usage_Frequency: Series[float] = pa.Field(
        ge=1, le=30, nullable=True, coerce=True, alias="Usage Frequency"
    )
    Support_Calls: Series[float] = pa.Field(
        ge=0, le=10, nullable=True, coerce=True, alias="Support Calls"
    )
    Payment_Delay: Series[float] = pa.Field(
        ge=0, le=30, nullable=True, coerce=True, alias="Payment Delay"
    )
    Subscription_Type: Series[object] = pa.Field(
        isin=["Basic", "Standard", "Premium"],
        nullable=True,
        alias="Subscription Type",
    )
    Contract_Length: Series[object] = pa.Field(
        isin=["Monthly", "Quarterly", "Annual"],
        nullable=True,
        alias="Contract Length",
    )
    Total_Spend: Series[float] = pa.Field(
        ge=100.0, le=1000.0, nullable=True, coerce=True, alias="Total Spend"
    )
    Last_Interaction: Series[float] = pa.Field(
        ge=1, le=30, nullable=True, coerce=True, alias="Last Interaction"
    )
    Churn: Series[float] = pa.Field(isin=[0.0, 1.0], nullable=True, coerce=True)

    class Config:
        coerce = True


class ChurnCleanSchema(pa.SchemaModel):
    """Stricter schema after cleaning — no nulls permitted."""

    CustomerID: Series[float] = pa.Field(nullable=False, coerce=True)
    Age: Series[float] = pa.Field(ge=18, le=65, nullable=False, coerce=True)
    Gender: Series[object] = pa.Field(isin=["Male", "Female"], nullable=False)
    Tenure: Series[float] = pa.Field(ge=1, le=60, nullable=False, coerce=True)
    Usage_Frequency: Series[float] = pa.Field(
        ge=1, le=30, nullable=False, coerce=True, alias="Usage Frequency"
    )
    Support_Calls: Series[float] = pa.Field(
        ge=0, le=10, nullable=False, coerce=True, alias="Support Calls"
    )
    Payment_Delay: Series[float] = pa.Field(
        ge=0, le=30, nullable=False, coerce=True, alias="Payment Delay"
    )
    Subscription_Type: Series[object] = pa.Field(
        isin=["Basic", "Standard", "Premium"],
        nullable=False,
        alias="Subscription Type",
    )
    Contract_Length: Series[object] = pa.Field(
        isin=["Monthly", "Quarterly", "Annual"],
        nullable=False,
        alias="Contract Length",
    )
    Total_Spend: Series[float] = pa.Field(
        ge=100.0, le=1000.0, nullable=False, coerce=True, alias="Total Spend"
    )
    Last_Interaction: Series[float] = pa.Field(
        ge=1, le=30, nullable=False, coerce=True, alias="Last Interaction"
    )
    Churn: Series[float] = pa.Field(isin=[0.0, 1.0], nullable=False, coerce=True)

    class Config:
        coerce = True


def validate_raw(df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
    """Validate raw dataframe against ChurnRawSchema.

    Raises SchemaError with context string on failure.
    """
    try:
        return ChurnRawSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise pa.errors.SchemaError(
            schema=e.schema,
            data=e.data,
            message=f"Schema violation in [{dataset_name}]: {e.args[0]}",
        ) from e


def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate cleaned dataframe — no nulls allowed, Churn must be 0 or 1."""
    try:
        return ChurnCleanSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(
            f"Clean data failed post-cleaning validation: {e.args[0]}"
        ) from e
