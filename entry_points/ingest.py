"""
What's needed in this file is:
TODO: redo function with focus on only outputting the dataframe
TODO: Docstring for functions
TODO: Function that takes and reads the data, then outputs it as a dataframe
TODO: Returns: Dataframe (pickled)
"""
import pandas as pd
import mlflow
import yaml
import logging
import requests
import os
from pathlib import Path

CONFIG_PATH = "configuration/config.yaml"
DATASET_PATH = "data/"

def load_file_as_dataframe() -> None:
    with mlflow.start_run() as mlrun:
        """
        Download or read a dataset

        """
        with open(CONFIG_PATH, "r", encoding="UTF-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
        logger = logging.getLogger(__name__)
        
        csv_url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        )
        file_name = Path(DATASET_PATH, cfg["dataset_name"])
        
        if not file_name.exists():
            print("File doesn't exist")
            dataset = requests.get(csv_url, allow_redirects=True)
            open("irisdata_raw.csv","wb").write(dataset.content)
        else:
            print("File already exists")
            print("Using existing file")
            dataset = requests.get(file_name)
        
        print("Uploading dataframe: %s" % dataset.content)
        mlflow.log_artifact(dataset.content, "artifacts/")

if __name__ == '__main__':
    load_file_as_dataframe()