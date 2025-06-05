"""Load, preprocess, split, and save the bank marketing dataset.

This script performs data loading, preprocessing, and saving for a bank marketing dataset
within a Databricks environment, leveraging Unity Catalog for data storage.

It takes the environment as a command-line argument to load the appropriate project configuration.

Key steps include:
    - Loading project configuration from a YAML file based on the specified environment.
    - Setting up logging to a file within a Databricks Volume, organized by catalog and schema.
    - Initializing a Spark session.
    - Constructing the path to the dataset within Unity Catalog using the configured catalog and schema.
    - Validating the existence of the dataset file.
    - Loading the data from Unity Catalog into a pandas DataFrame using Spark.
    - Preprocessing the data using a dedicated DataProcessor class.
    - Splitting the preprocessed data into training and testing sets.
    - Saving the training and testing sets back to Unity Catalog.

The script utilizes logging to provide detailed information about the execution process,
including configuration loading, data loading, preprocessing duration, and data saving.
"""

import argparse
import os

import yaml
from loguru import logger
from marvelous.logging import setup_logging
from marvelous.timer import Timer
from pyspark.sql import SparkSession

from src.bank_marketing.config import ProjectConfig
from src.bank_marketing.data_processor import DataProcessor


def main(env: str) -> None:
    """Load, preprocess, split, and save the bank marketing dataset.

    Args:
        env (str): Environment name to load the appropriate configuration (e.g., 'dev', 'acc', 'prd').

    """
    # Robust path to the YAML configuration file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "../project_config.yml")

    # Load configuration
    config = ProjectConfig.from_yaml(config_path=config_path, env=env)

    # Setup logging (save to path based on catalog and schema)
    log_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/logs/marvelous-preprocess.log"
    setup_logging(log_file=log_path)

    logger.info("✅ Configuration loaded successfully")
    logger.info(yaml.dump(config, default_flow_style=False))

    # Initialize Spark
    spark = SparkSession.builder.getOrCreate()

    # Dataset path
    data_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/data/bank_data.csv"

    # Dataset existence validation
    if not spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration()).exists(
        spark._jvm.org.apache.hadoop.fs.Path(data_path)
    ):
        logger.error(f"❌ File does not exist at: {data_path}")
        return

    # Load data from Unity Catalog
    logger.info("📥 Loading data from Unity Catalog...")
    df = spark.read.csv(data_path, header=True, inferSchema=True).toPandas()

    # Preprocessing
    logger.info("⚙️ Starting data preprocessing...")
    with Timer() as preprocess_timer:
        data_processor = DataProcessor(df, config, spark)
        data_processor.preprocess()
    logger.info(f"⏱️ Preprocessing time: {preprocess_timer}")

    # Data split
    X_train, X_test = data_processor.split_data()
    logger.info(f"📊 Training set shape: {X_train.shape}")
    logger.info(f"📊 Test set shape: {X_test.shape}")

    # Save to Unity Catalog
    logger.info("💾 Saving data to Unity Catalog...")
    data_processor.save_to_catalog(X_train, X_test)
    logger.info("✅ Process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bank marketing data processing")
    parser.add_argument("--env", type=str, default="dev", help="Environment: dev, acc, prd")
    args = parser.parse_args()

    main(env=args.env)
