"""Dataloader fixture."""

import pandas as pd
import pytest
from loguru import logger
from pyspark.sql import SparkSession

from bank_marketing import PROJECT_DIR
from bank_marketing.config import ProjectConfig, Tags
from tests.unit_tests.spark_config import spark_config


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    """Create and return a SparkSession for testing.

    This fixture creates a SparkSession with the specified configuration and returns it for use in tests.
    """
    spark = (
        SparkSession.builder.master(spark_config.master)
        .appName(spark_config.app_name)
        .config("spark.executor.cores", spark_config.spark_executor_cores)
        .config("spark.executor.instances", spark_config.spark_executor_instances)
        .config("spark.sql.shuffle.partitions", spark_config.spark_sql_shuffle_partitions)
        .config("spark.driver.bindAddress", spark_config.spark_driver_bindAddress)
        .getOrCreate()
    )

    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Load and return the project configuration.

    This fixture reads the project configuration from a YAML file and returns a ProjectConfig object.

    :return: The loaded project configuration
    """
    config_file_path = (PROJECT_DIR / "project_config.yml").resolve()
    logger.info(f"Current config file path: {config_file_path.as_posix()}")

    try:
        # Load and parse configuration
        loaded_config = ProjectConfig.from_yaml(config_file_path.as_posix())
        logger.info("✅ ProjectConfig loaded successfully.")
        # Log key config attributes to ensure they are populated
        logger.info(
            f"Config: catalog_name={loaded_config.catalog_name}, schema_name={loaded_config.schema_name}, "
            f"num_features={loaded_config.num_features}, cat_features={loaded_config.cat_features}, "
            f"target={loaded_config.target}, experiment_name_basic={loaded_config.experiment_name_basic}"
        )
        return loaded_config
    except Exception as e:
        logger.error(f"❌ Error loading ProjectConfig from {config_file_path}: {e}")
        # Re-raise the exception to make the test fail clearly if config loading is the issue
        raise


@pytest.fixture(scope="function")
def sample_bank_data(config: ProjectConfig, spark_session: SparkSession) -> pd.DataFrame:
    """Create a sample DataFrame from the original data.csv file.

    This fixture loads data.csv and creates a random sample for testing.

    :return: A sampled Pandas DataFrame containing bank marketing data.
    """
    # Ruta al archivo original data.csv
    original_data_path = PROJECT_DIR / "data" / "data.csv"  # Si está en carpeta data/

    # Rutas para archivos de test
    file_path = PROJECT_DIR / "tests" / "test_data" / "sample_bank_data.csv"

    try:
        if original_data_path.exists():
            logger.info(f"Loading original data from {original_data_path}")

            # Cargar datos originales
            full_data = pd.read_csv(original_data_path.as_posix())
            logger.info(f"Original data shape: {full_data.shape}")

            # Crear muestra aleatoria (por ejemplo, 1000 filas o 10% del total)
            sample_size = min(1000, int(len(full_data) * 0.1))  # Máximo 1000 filas o 10%
            sample_bank_data = full_data.sample(n=sample_size, random_state=42)

            logger.info(f"Created sample with {len(sample_bank_data)} rows from original {len(full_data)} rows")

            # Guardar muestra para uso futuro
            test_data_dir = PROJECT_DIR / "tests" / "test_data"
            test_data_dir.mkdir(parents=True, exist_ok=True)
            sample_bank_data.to_csv(file_path.as_posix(), index=False)
            logger.info(f"Saved sample data to {file_path}")

        else:
            logger.error(f"❌ Original data file not found at {original_data_path}")
            # Fallback a datos sintéticos si no encuentra el archivo original
            logger.info("Creating synthetic fallback data...")

            data = {
                "age": [30, 35, 42, 28, 55, 37, 29, 49, 51, 33],
                "job": [
                    "admin",
                    "blue-collar",
                    "management",
                    "technician",
                    "retired",
                    "admin",
                    "services",
                    "blue-collar",
                    "management",
                    "technician",
                ],
                "marital": [
                    "married",
                    "single",
                    "divorced",
                    "married",
                    "married",
                    "single",
                    "divorced",
                    "married",
                    "single",
                    "divorced",
                ],
                "education": [
                    "secondary",
                    "primary",
                    "tertiary",
                    "secondary",
                    "secondary",
                    "tertiary",
                    "primary",
                    "secondary",
                    "tertiary",
                    "primary",
                ],
                "default": ["no", "no", "no", "no", "yes", "no", "no", "yes", "no", "no"],
                "balance": [1000, 2500, 3000, 800, 5000, 1500, 700, 2000, 4000, 950],
                "housing": ["yes", "no", "yes", "yes", "no", "yes", "no", "no", "yes", "yes"],
                "loan": ["no", "yes", "no", "no", "no", "yes", "no", "yes", "no", "no"],
                "contact": [
                    "cellular",
                    "telephone",
                    "cellular",
                    "telephone",
                    "cellular",
                    "telephone",
                    "cellular",
                    "telephone",
                    "cellular",
                    "telephone",
                ],
                "day": [10, 15, 20, 5, 25, 12, 8, 18, 22, 30],
                "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct"],
                "duration": [120, 300, 180, 240, 90, 150, 210, 270, 330, 200],
                "campaign": [2, 1, 3, 2, 1, 4, 2, 3, 1, 5],
                "pdays": [999, 10, 999, 5, 999, 8, 999, 12, 999, 15],
                "previous": [0, 1, 0, 3, 0, 2, 0, 4, 0, 1],
                "poutcome": [
                    "unknown",
                    "success",
                    "unknown",
                    "failure",
                    "unknown",
                    "success",
                    "unknown",
                    "failure",
                    "unknown",
                    "success",
                ],
                "Target": ["no", "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes"],
            }

            sample_bank_data = pd.DataFrame(data)

            # Guardar datos sintéticos
            test_data_dir = PROJECT_DIR / "tests" / "test_data"
            test_data_dir.mkdir(parents=True, exist_ok=True)
            sample_bank_data.to_csv(file_path.as_posix(), index=False)
            logger.info(f"Created and saved synthetic fallback data to {file_path}")

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Crear DataFrame mínimo en caso de error
        sample_bank_data = pd.DataFrame(
            {
                "age": [30, 40],
                "job": ["admin", "technician"],
                "balance": [1000, 2000],
                "Target": ["no", "yes"],  # ✅ Usar "y" no "Target"
            }
        )

    # Verificar que todas las columnas configuradas estén presentes
    expected_columns = config.num_features + config.cat_features + [config.target]
    missing_columns = [col for col in expected_columns if col not in sample_bank_data.columns]

    if missing_columns:
        logger.warning(f"Missing columns in sample data: {missing_columns}")
        # Añadir columnas faltantes con valores predeterminados
        for col in missing_columns:
            if col in config.num_features:
                sample_bank_data[col] = 0
            elif col in config.cat_features:
                sample_bank_data[col] = "unknown"
            elif col == config.target:
                sample_bank_data[col] = "no"

    # Dividir el DataFrame en conjunto de entrenamiento (80%) y test (20%)
    try:
        train_df = sample_bank_data.sample(frac=0.8, random_state=42)
        test_df = sample_bank_data.drop(train_df.index)

        train_path = PROJECT_DIR / "tests" / "catalog" / "train_set.csv"
        test_path = PROJECT_DIR / "tests" / "catalog" / "test_set.csv"

        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(train_path.as_posix(), index=False)
        test_df.to_csv(test_path.as_posix(), index=False)
        logger.info(f"Train and test sets saved to {train_path} and {test_path}")

    except Exception as split_error:
        logger.error(f"Failed to split and save train/test sets: {split_error}")

    return sample_bank_data


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Create and return a Tags instance for the test session.

    This fixture provides a Tags object with predefined values for git_sha, branch, and job_run_id.
    """
    return Tags(git_sha="wxyz", branch="test", job_run_id="9")
