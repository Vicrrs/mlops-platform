from __future__ import annotations

from churn_model.io.readers import read_delta, read_parquet
from churn_model.io.writers import write_delta, write_parquet


def test_delta_round_trip(spark, sample_customers_df, tmp_path):
    path = str(tmp_path / "delta_tbl")
    write_delta(sample_customers_df, path, mode="overwrite", environment="dev")
    result = read_delta(spark, path)
    assert result.count() == sample_customers_df.count()
    assert set(result.columns) == set(sample_customers_df.columns)


def test_parquet_round_trip(spark, sample_customers_df, tmp_path):
    path = str(tmp_path / "parquet_tbl")
    write_parquet(sample_customers_df, path, mode="overwrite", environment="dev")
    result = read_parquet(spark, path)
    assert result.count() == sample_customers_df.count()


def test_delta_append_accumulates_rows(spark, sample_customers_df, tmp_path):
    path = str(tmp_path / "delta_append")
    half = sample_customers_df.limit(50)
    write_delta(half, path, mode="overwrite", environment="dev")
    write_delta(half, path, mode="append", environment="dev")
    assert read_delta(spark, path).count() == 100


def test_delta_partitioned_write_creates_partition_columns(spark, sample_customers_df, tmp_path):
    path = str(tmp_path / "delta_partitioned")
    write_delta(
        sample_customers_df, path, mode="overwrite", partition_by=["contract_type"], environment="dev"
    )
    result = read_delta(spark, path)
    assert result.count() == sample_customers_df.count()
