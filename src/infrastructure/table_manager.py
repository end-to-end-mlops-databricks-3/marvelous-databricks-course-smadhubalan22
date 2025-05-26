"""Table management utilities for working with Delta tables in Unity Catalog.

This module provides a TableManager class that simplifies working with tables and schemas
in Databricks Unity Catalog, supporting common operations needed in MLOps pipelines.
"""

from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig
from infrastructure.table_manager import TableManager

# Initialize components
spark = SparkSession.builder.getOrCreate()
config = ProjectConfig.from_yaml("../project_config.yml", env="dev")
table_manager = TableManager(spark, config)

# Ensure schema exists
table_manager.ensure_schema_exists()

# Load data (using your DataProcessor first to create initial tables)
raw_train = table_manager.load_table("train_raw")

# Create ML-ready version with numeric target
table_manager.create_ml_ready_table("train")

# Get table metadata
metadata = table_manager.get_table_metadata("train_processed")
print(f"Table has {metadata['num_files']} files and size {metadata['size_bytes']} bytes")

# Optimize table for better query performance
table_manager.optimize_table("train_processed", zorder_by=["age", "balance"])

# Combine features from multiple tables
table_manager.combine_features_from_tables(
    output_table="combined_features",
    source_tables=["customer_demographics", "transaction_history", "campaign_responses"],
    join_on="customer_id",
    target_column="Target",
)
