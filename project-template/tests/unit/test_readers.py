from __future__ import annotations

import pytest

from churn_model.data.schemas import RAW_CUSTOMER_CHURN_SCHEMA
from churn_model.exceptions import DataReadError
from churn_model.io.readers import read_csv, read_jdbc, validate_schema


def test_read_csv_with_explicit_schema(spark):
    df = read_csv(spark, "data/sample/customers_churn.csv", schema=RAW_CUSTOMER_CHURN_SCHEMA, header=True)
    assert df.count() > 0
    assert set(df.columns) == set(f.name for f in RAW_CUSTOMER_CHURN_SCHEMA.fields)


def test_read_csv_without_schema_requires_explicit_opt_in(spark):
    with pytest.raises(DataReadError):
        read_csv(spark, "data/sample/customers_churn.csv", schema=None, allow_schema_inference=False)


def test_read_csv_allows_inference_when_explicitly_enabled(spark):
    df = read_csv(spark, "data/sample/customers_churn.csv", schema=None, allow_schema_inference=True)
    assert df.count() > 0


def test_read_jdbc_requires_credentials(spark):
    with pytest.raises(DataReadError):
        read_jdbc(spark, url="jdbc:postgresql://host/db", table="t", properties={"user": "u"})


def test_validate_schema_raises_on_mismatch(spark, sample_customers_df):
    from pyspark.sql.types import StringType, StructField, StructType

    from churn_model.exceptions import SchemaValidationError

    wrong_schema = StructType([StructField("does_not_exist", StringType())])
    with pytest.raises(SchemaValidationError):
        validate_schema(sample_customers_df, wrong_schema, source="test")
