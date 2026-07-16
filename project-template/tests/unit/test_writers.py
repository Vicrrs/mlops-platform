from __future__ import annotations

import pytest

from churn_model.exceptions import DataWriteError, EmptyDataFrameError
from churn_model.io.writers import merge_delta, write_delta


def test_write_delta_and_read_back(spark, sample_customers_df, tmp_path):
    target = str(tmp_path / "customers_delta")
    write_delta(sample_customers_df, target, mode="overwrite", environment="dev")
    result = spark.read.format("delta").load(target)
    assert result.count() == sample_customers_df.count()


def test_write_delta_rejects_empty_dataframe(spark, sample_customers_df, tmp_path):
    empty_df = sample_customers_df.limit(0)
    with pytest.raises(EmptyDataFrameError):
        write_delta(empty_df, str(tmp_path / "empty"), mode="overwrite", environment="dev")


def test_write_delta_blocks_prd_overwrite_without_explicit_flag(spark, sample_customers_df, tmp_path):
    with pytest.raises(DataWriteError):
        write_delta(sample_customers_df, str(tmp_path / "prd_tbl"), mode="overwrite", environment="prd")


def test_write_delta_allows_prd_overwrite_when_explicit(spark, sample_customers_df, tmp_path):
    target = str(tmp_path / "prd_tbl_explicit")
    write_delta(sample_customers_df, target, mode="overwrite", environment="prd", allow_prd_overwrite=True)
    assert spark.read.format("delta").load(target).count() == sample_customers_df.count()


def test_write_delta_rejects_invalid_partition_column(spark, sample_customers_df, tmp_path):
    with pytest.raises(DataWriteError):
        write_delta(
            sample_customers_df,
            str(tmp_path / "bad_partition"),
            mode="overwrite",
            partition_by=["does_not_exist"],
            environment="dev",
        )


def test_write_delta_dry_run_does_not_write(spark, sample_customers_df, tmp_path):
    target = str(tmp_path / "dry_run_tbl")
    write_delta(sample_customers_df, target, mode="overwrite", environment="dev", dry_run=True)
    assert not (tmp_path / "dry_run_tbl").exists()


def test_merge_delta_upsert(spark, sample_customers_df, tmp_path):
    target = str(tmp_path / "merge_tbl")
    first_batch = sample_customers_df.limit(100)
    merge_delta(first_batch, target, merge_keys=["customer_id"], environment="dev")
    assert spark.read.format("delta").load(target).count() == 100

    second_batch = sample_customers_df.limit(150)
    merge_delta(second_batch, target, merge_keys=["customer_id"], environment="dev")
    assert spark.read.format("delta").load(target).count() == 150
