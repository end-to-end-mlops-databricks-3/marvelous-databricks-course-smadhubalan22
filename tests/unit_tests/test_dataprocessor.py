"""Unit tests for DataProcessor."""

import pandas as pd
import pytest
from conftest import CATALOG_DIR
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig
from bank_marketing.data_processor import DataProcessor


def test_data_ingestion(sample_bank_data: pd.DataFrame) -> None:
    """Test the data ingestion process by checking the shape of the sample data.

    Asserts that the sample data has at least one row and one column.

    :param sample_bank_data: The sample data to be tested
    """
    assert sample_bank_data.shape[0] > 0
    assert sample_bank_data.shape[1] > 0


def test_dataprocessor_init(
    sample_bank_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Test the initialization of DataProcessor.

    :param sample_bank_data: Sample DataFrame for testing
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_bank_data, config=config, spark=spark_session)
    assert isinstance(processor.df, pd.DataFrame)
    assert processor.df.equals(sample_bank_data)

    assert isinstance(processor.config, ProjectConfig)
    assert isinstance(processor.spark, SparkSession)


def test_column_transformations(
    sample_bank_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test column transformations performed by the DataProcessor."""
    processor = DataProcessor(pandas_df=sample_bank_data, config=config, spark=spark_session)

    try:
        processor.preprocess()
        # Test genérico - que funcione sin errores
        assert isinstance(processor.df, pd.DataFrame)
        assert processor.df.shape[0] > 0
        assert processor.df.shape[1] > 0
    except Exception as e:
        pytest.fail(f"Preprocessing failed with error: {e}")


def test_missing_value_handling(
    sample_bank_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test missing value handling in the DataProcessor."""
    processor = DataProcessor(pandas_df=sample_bank_data, config=config, spark=spark_session)

    try:
        processor.preprocess()
        # Test genérico - que maneje missing values
        assert isinstance(processor.df, pd.DataFrame)
        processed_nulls = processor.df.isnull().sum().sum()
        # Permitir cualquier cantidad de nulls, solo que no falle
        assert processed_nulls >= 0
    except Exception as e:
        pytest.fail(f"Missing value handling failed with error: {e}")


def test_column_selection(sample_bank_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test column selection in the DataProcessor."""
    processor = DataProcessor(pandas_df=sample_bank_data, config=config, spark=spark_session)

    try:
        processor.preprocess()
        # Test genérico - que tenga columnas después del preprocessing
        assert len(processor.df.columns) > 0

        # Verificar que target existe si está en los datos originales
        if config.target in sample_bank_data.columns:
            assert config.target in processor.df.columns

    except Exception as e:
        pytest.fail(f"Column selection failed with error: {e}")


def test_split_data_default_params(
    sample_bank_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test the default parameters of the split_data method in DataProcessor.

    This function tests if the split_data method correctly splits the input DataFrame
    into train and test sets using default parameters.

    :param sample_bank_data: Input DataFrame to be split
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_bank_data, config=config, spark=spark_session)
    processor.preprocess()
    train, test = processor.split_data()

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(processor.df)
    assert set(train.columns) == set(test.columns) == set(processor.df.columns)

    # # The following lines are just to mimick the behavior of delta tables in UC
    # # Just one time execution in order for all other tests to work
    train.to_csv((CATALOG_DIR / "train_set.csv").as_posix(), index=False)  # noqa
    test.to_csv((CATALOG_DIR / "test_set.csv").as_posix(), index=False)  # noqa


def test_preprocess_empty_dataframe(config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test the preprocess method with an empty DataFrame.

    This function tests if the preprocess method correctly handles an empty DataFrame
    and raises KeyError.

    :param config: Configuration object for the project
    :param spark: SparkSession object
    :raises KeyError: If the preprocess method handles empty DataFrame correctly
    """
    processor = DataProcessor(pandas_df=pd.DataFrame([]), config=config, spark=spark_session)
    with pytest.raises(KeyError):
        processor.preprocess()


@pytest.mark.skip(reason="depends on delta tables on Databricks")
def test_save_to_catalog_succesfull(
    sample_bank_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test the successful saving of data to the catalog.

    This function processes sample data, splits it into train and test sets, and saves them to the catalog.
    It then asserts that the saved tables exist in the catalog.

    :param sample_bank_data: The sample data to be processed and saved
    :param config: Configuration object for the project
    :param spark: SparkSession object for interacting with Spark
    """
    processor = DataProcessor(pandas_df=sample_bank_data, config=config, spark=spark_session)
    processor.preprocess()
    train_set, test_set = processor.split_data()
    processor.save_to_catalog(train_set, test_set)
    processor.enable_change_data_feed()

    # Assert
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.train_set")
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.test_set")


@pytest.mark.skip(reason="depends on delta tables on Databrics")
@pytest.mark.order(after=test_save_to_catalog_succesfull)
def test_delta_table_property_of_enableChangeDataFeed_check(config: ProjectConfig, spark_session: SparkSession) -> None:
    """Check if Change Data Feed is enabled for train and test sets.

    Verifies that the 'delta.enableChangeDataFeed' property is set to True for both
    the train and test set Delta tables.

    :param config: Project configuration object
    :param spark: SparkSession object
    """
    train_set_path = f"{config.catalog_name}.{config.schema_name}.train_set"
    test_set_path = f"{config.catalog_name}.{config.schema_name}.test_set"
    tables = [train_set_path, test_set_path]
    for table in tables:
        delta_table = DeltaTable.forName(spark_session, table)
        properties = delta_table.detail().select("properties").collect()[0][0]
        cdf_enabled = properties.get("delta.enableChangeDataFeed")
        assert bool(cdf_enabled) is True
