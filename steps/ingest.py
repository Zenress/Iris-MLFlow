"""
What's needed in this file is:
"""
import mlflow
import yaml
import logging
import requests
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
        mlflow.log_artifact(dataset.content, "dataset_path")

if __name__ == '__main__':
    load_file_as_dataframe()