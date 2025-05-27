"""Contain Week 1's notebook code for the project."""

# COMMAND ----------
# Initialize installations

import pandas as pd
import yaml
from loguru import logger
from marvelous.logging import setup_logging
from marvelous.timer import Timer
from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig
from bank_marketing.data_processor import DataProcessor
from infrastructure.volume_manager import VolumeManager

# COMMAND ----------
# Load configuration from YAML
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
# Setup logging
setup_logging(log_file="logs/marvelous-1.log")
logger.info("✅ Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Initialize Spark and load CSV from Unity Catalog
spark = SparkSession.builder.getOrCreate()
filepath = "../data/data.csv"
# Load the data
df = pd.read_csv(filepath)
logger.info(f"📥 Data loaded from: {filepath} - Shape: {df.shape}")

# COMMAND ----------
# Preprocessing
with Timer() as preprocess_timer:
    data_processor = DataProcessor(df, config, spark)
    data_processor.preprocess()
logger.info(f"⚙️ Preprocessing completed in: {preprocess_timer}")

# COMMAND ----------
# Train/test split
X_train, X_test = data_processor.split_data()
logger.info(f"📊 Train shape: {X_train.shape} | Test shape: {X_test.shape}")

# Volume creation and verification
volume_manager = VolumeManager(spark, config)
volume_manager.ensure_volume_exists()
logger.info(f"📦 Volume configured: {volume_manager.volume_path}")

# COMMAND ----------
# Save to Volume
logger.info("💾 Saving data to volume (raw + processed)")
with Timer() as volume_timer:
    data_processor.save_to_volume(X_train, X_test)
logger.info(f"⏱️ Data saved to volume in: {volume_timer}")

# COMMAND ----------
# Save to Unity Catalog (maintained for compatibility)
try:
    logger.info("💾 Saving train/test to Unity Catalog (raw + processed)")
    data_processor.save_to_catalog(X_train, X_test)
    # Enable Change Data Feed for all tables
    if hasattr(data_processor, "enable_change_data_feed"):
        logger.info("🔁 Enabling Change Data Feed for tables")
        data_processor.enable_change_data_feed()
    else:
        logger.warning("⚠️ Method enable_change_data_feed not found")

    logger.info("✅ Data available in both raw and processed formats")
except Exception as e:
    logger.warning(f"⚠️ Could not save to Unity Catalog: {e}")
    logger.info("ℹ️ Data is available in the volume")

# COMMAND ----------
# Enable Change Data Feed for volumes (optional)
try:
    logger.info("🔁 Enabling Change Data Feed for volumes")
    data_processor.enable_volume_change_data_feed()
except Exception as e:
    logger.warning(f"⚠️ ------> Could not enable Change Data Feed for volumes: {e}")

# COMMAND ----------
