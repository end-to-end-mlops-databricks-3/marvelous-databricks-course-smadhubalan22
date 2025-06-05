"""Module for managing volume mounting and unmounting across environments."""

from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig


class VolumeManager:
    """Manages Databricks Volume operations for MLOps projects."""

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        self.spark = spark
        self.config = config
        self.volume_path = self._get_volume_path()

    def _get_volume_path(self) -> str:
        """Get the base path for the volume."""
        return f"/Volumes/{self.config.catalog_name}/{self.config.schema_name}/{self.config.volume_name}"

    def ensure_volume_exists(self) -> None:
        """Create the volume if it doesn't exist."""
        self.spark.sql(f"""
        CREATE VOLUME IF NOT EXISTS
        {self.config.catalog_name}.{self.config.schema_name}.{self.config.volume_name}
        COMMENT 'Volume for Bank Marketing MLOps project'
        """)

    def get_path(self, data_type: str, subset: str = None) -> str:
        """Get the path for a specific data type.

        Args:
            data_type: 'raw', 'processed', 'models', etc.
            subset: Optional, subset like 'train', 'test', etc.

        """
        path = f"{self.volume_path}/{data_type}"
        if subset:
            path = f"{path}/{subset}"
        return path
