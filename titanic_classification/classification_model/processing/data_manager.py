import logging
import re
from pathlib import Path
from typing import Any, List, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from classification_model import __version__ as _version
from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

logger = logging.getLogger(__name__)


# This needs to be able to return a float because of np.nan
def get_first_cabin(row: Any) -> Union[str, float]:
    """Returns the first cabin if a row has more than one cabin listed."""
    try:
        return row.split()[0]
    except AttributeError:
        return np.nan
    
def get_title(passenger: str) -> str:
    """Extracts the title (Mr, Ms, etc) from the name variable."""
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
    

def pre_process(*, dataframe: pd.Dataframe) -> pd.Dataframe:
    """Perform pre-processing steps before dataframe enters pipeline.
    
    Performs the listed preprocessing steps on the given dataframe
    in the following order:
    1. Replaces '?' with np.nan
    2. Calls get_first_cabin 
    3. Calls get_title
    4. Casts numerical vars as floats
    5. Drops unnecessary variables
    """

    # Replace question marks with np.nan
    data = dataframe.replace("?", np.nan)

    # Retain first cabin if multiple given for any passenger
    data["cabin"] = data["cabin"].apply(get_first_cabin)

    # Create "title" feature
    data["title"] = data["name"].apply(get_title)

    # Cast numerical variables as floats
    data["fare"]  = data["fare"].astype("float")
    data["age"] = data["age"].astype("float")

    # Drop unnecessary variables
    data.drop(label=config.model_config.unused_fields, axis=1, inplace=True)

    return data


def _load_unprocessed_dataset(*, file_name: str) -> pd.DataFrame:
    "Load in dataset that has not been split into train/test"
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_process(dataframe=dataframe)

    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
