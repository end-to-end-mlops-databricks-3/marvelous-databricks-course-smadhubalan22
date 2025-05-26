"""Configuration file for the Bank Marketing MLOps project."""

from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Represents the configuration parameters loaded from a YAML file.

    Includes feature definitions, model hyperparameters, catalog/schema config,
    and supports dynamic environment overrides (dev, acc, prd).
    """

    num_features: list[str]
    cat_features: list[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: dict[str, Any]
    volume_name: str
    experiment_name_basic: str | None
    experiment_name_custom: str | None

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load and parse configuration settings from a YAML file.

        Args:
            config_path (str): Path to the YAML config file.
            env (str): Environment name ('dev', 'acc', or 'prd').

        Returns:
            ProjectConfig: Parsed configuration object.

        """
        valid_envs = {"dev", "acc", "prd"}
        if env not in valid_envs:
            raise ValueError(f"Invalid environment: {env}. Expected one of: {valid_envs}")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]
            config_dict["volume_name"] = config_dict[env]["volume_name"]  # Nueva línea

            return cls(**config_dict)


class Tags(BaseModel):
    """Represents metadata tags for MLflow logging or model versioning."""

    git_sha: str = Field(..., description="Short SHA of the Git commit")
    branch: str = Field(..., description="Branch name")
    job_run_id: str = Field(..., description="Databricks job run ID")
