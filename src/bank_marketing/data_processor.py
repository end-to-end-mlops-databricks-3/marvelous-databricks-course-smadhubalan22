"""Preprocessing, split, and store the Bank Marketing dataset.

It handles various data cleaning and transformation steps, prepares the data for machine learning,
and provides functionalities to save the processed data to both Unity Catalog tables and Databricks Volumes.

Methods:
    - __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession):
        Initializes the DataProcessor with a pandas DataFrame, project configuration, and a SparkSession.

    - preprocess(self) -> None:
        Preprocesses the raw dataset by converting column types, handling missing values (replacing 'unknown' with NaN,
        filling NaNs in categorical columns with 'missing' and in numeric columns with 0), encoding the target variable
        to binary (1 for 'yes', 0 for 'no'), and selecting relevant features as defined in the project configuration.

    - split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        Splits the processed DataFrame into training and testing sets using scikit-learn's `train_test_split`.

    - save_to_catalog(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        Saves the training and testing DataFrames to Delta tables in Unity Catalog. It saves two versions:
        a 'raw' version where the target variable is kept as a string ('yes'/'no') for audit purposes,
        and a 'processed' version where the target is numeric (1/0) for model training. It also adds
        an `update_timestamp_utc` column and overwrites existing tables.

    - enable_change_data_feed(self) -> None:
        Enables Delta Lake Change Data Feed (CDF) on all the created tables in Unity Catalog
        ('train_raw', 'test_raw', 'train_processed', 'test_processed'). CDF allows tracking changes
        made to the tables over time.

    - save_to_volume(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        Saves the training and testing DataFrames to Databricks Volumes in two formats, similar to `save_to_catalog`:
        'raw' (target as string) and 'processed' (target as numeric). It initializes the VolumeManager,
        ensures the volume exists, and saves the data as Delta tables with the `overwriteSchema` option
        enabled, adding an `update_timestamp_utc` column.

    - enable_volume_change_data_feed(self) -> None:
        Enables Delta Lake Change Data Feed (CDF) on all the Delta tables saved within the Databricks Volume
        ('raw/train', 'raw/test', 'processed/train', 'processed/test').

"""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from bank_marketing.config import ProjectConfig
from infrastructure.volume_manager import VolumeManager


class DataProcessor:
    """Preprocesses, splits, and stores the Bank Marketing dataset."""

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df
        self.config = config
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the raw dataset.

        Converts column types, handles missing values, encodes categorical variables,
        and selects only relevant features.
        """
        target_column = self.config.target
        print(f"Target column defined in config: {target_column}")

        if target_column not in self.df.columns:
            raise KeyError(f"The column '{target_column}' is not present in the DataFrame.")

        # Convert numeric columns to numeric type (with error coercion)
        for col in self.config.num_features:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Convert categorical columns to category type
        for col in self.config.cat_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

        # Replace 'unknown' with pd.NA
        self.df.replace("unknown", pd.NA, inplace=True)

        # Add 'missing' category to categorical columns and fill NaNs
        for col in self.config.cat_features:
            if col in self.df.columns:
                if "missing" not in self.df[col].cat.categories:
                    self.df[col] = self.df[col].cat.add_categories("missing")
                self.df[col] = self.df[col].fillna("missing")

        # Fill NaNs in numeric columns (example: with 0, you could use the mean if preferred)
        for col in self.config.num_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

        # Convert the target column to binary
        if target_column in self.df.columns:
            self.df[target_column] = self.df[target_column].map({"yes": 1, "no": 0}).fillna(0).astype(int)

        # Ensure that categorical columns have the correct type
        for col in self.config.cat_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

        # Filter the relevant columns
        features = self.config.cat_features + self.config.num_features + [target_column]
        print(f"Features to keep: {features}")

        # Verify existence of all necessary columns
        missing_columns = [col for col in features if col not in self.df.columns]
        if missing_columns:
            raise KeyError(f"The following columns are not in the DataFrame: {missing_columns}")

        # Keep only the desired columns
        self.df = self.df[features]

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into training and testing sets."""
        train_df, test_df = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_df, test_df

    def save_to_catalog(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Save processed data into Delta tables in the Unity Catalog with MLOps best practices."""
        # 1. Save RAW version (target as string) for audit and reference
        train_raw = train_df.copy()
        test_raw = test_df.copy()

        # Convert target to string if it is numeric
        target_column = self.config.target
        if target_column in train_raw.columns and pd.api.types.is_numeric_dtype(train_raw[target_column]):
            train_raw[target_column] = train_raw[target_column].map({1: "yes", 0: "no"})
            test_raw[target_column] = test_raw[target_column].map({1: "yes", 0: "no"})

        # Create Spark DataFrames
        train_raw_sdf = self.spark.createDataFrame(train_raw).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_raw_sdf = self.spark.createDataFrame(test_raw).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Save raw version
        train_raw_sdf.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_raw"
        )
        test_raw_sdf.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_raw"
        )

        # Create Spark DataFrames for processed data
        train_processed_sdf = self.spark.createDataFrame(train_df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_processed_sdf = self.spark.createDataFrame(test_df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Save processed version with schema overwrite to handle type changes
        train_processed_sdf.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_processed"
        )
        test_processed_sdf.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_processed"
        )

        print("Data saved in Unity Catalog:")
        print(f"  - Raw data (target as string): {self.config.catalog_name}.{self.config.schema_name}.train_raw")
        print(
            f"  - Processed data (numeric target): {self.config.catalog_name}.{self.config.schema_name}.train_processed"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Delta Lake Change Data Feed on all catalog tables."""
        # Raw tables
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_raw "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_raw "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        # Processed tables
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_processed "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_processed "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

    def save_to_volume(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Save data to volumes in both raw (string target) and processed (numeric target) formats."""
        # Initialize volume
        volume_manager = VolumeManager(self.spark, self.config)
        volume_manager.ensure_volume_exists()

        # Get target column name
        target_column = self.config.target

        # 1. Save raw data (with target as string)
        train_raw = train_df.copy()
        test_raw = test_df.copy()

        # Convert target back to string if it is numeric
        if target_column in train_raw.columns and pd.api.types.is_numeric_dtype(train_raw[target_column]):
            train_raw[target_column] = train_raw[target_column].map({1: "yes", 0: "no"})
            test_raw[target_column] = test_raw[target_column].map({1: "yes", 0: "no"})

        # Create Spark DataFrames
        train_raw_sdf = self.spark.createDataFrame(train_raw)
        test_raw_sdf = self.spark.createDataFrame(test_raw)

        # Add timestamp
        train_raw_sdf = train_raw_sdf.withColumn("update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))
        test_raw_sdf = test_raw_sdf.withColumn("update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

        # Save raw version WITH overwriteSchema option
        raw_train_path = volume_manager.get_path("raw", "train")
        raw_test_path = volume_manager.get_path("raw", "test")

        train_raw_sdf.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(raw_train_path)

        test_raw_sdf.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(raw_test_path)

        # 2. Save processed version (with numeric target)
        train_processed_sdf = self.spark.createDataFrame(train_df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_processed_sdf = self.spark.createDataFrame(test_df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Save processed version WITH overwriteSchema option
        processed_train_path = volume_manager.get_path("processed", "train")
        processed_test_path = volume_manager.get_path("processed", "test")

        train_processed_sdf.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(
            processed_train_path
        )

        test_processed_sdf.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(
            processed_test_path
        )

        print(f"Data saved to volume: {volume_manager.volume_path}")
        print(f"  - Raw data: {volume_manager.get_path('raw')}")
        print(f"  - Processed data: {volume_manager.get_path('processed')}")

    def enable_volume_change_data_feed(self) -> None:
        """Enable Delta Lake Change Data Feed on the saved volume tables."""
        volume_manager = VolumeManager(self.spark, self.config)

        # Get paths for the data
        raw_train_path = volume_manager.get_path("raw", "train")
        raw_test_path = volume_manager.get_path("raw", "test")
        processed_train_path = volume_manager.get_path("processed", "train")
        processed_test_path = volume_manager.get_path("processed", "test")

        # Enable Change Data Feed for all tables
        for path in [raw_train_path, raw_test_path, processed_train_path, processed_test_path]:
            self.spark.sql(f"ALTER TABLE delta.`{path}` SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        print("=====> Change Data Feed enabled for all tables in volume")
