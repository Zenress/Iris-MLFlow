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

CONFIG_PATH = "configuration/config.yaml"

def load_file_as_dataframe() -> None:
    with mlflow.start_run() as mlrun:
        """
        Read data from dataset

        Reads the data from the dataset and assigns the header with column_names.
        Then it encodes the categorical label column into a numerical label column.
        """
        with open(CONFIG_PATH, "r", encoding="UTF-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
        logger = logging.getLogger(__name__)
        
        csv_url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        )
        try:
            dataframe = pd.read_csv()
        except Exception as e:
            logger.exception(
                "Unable to download training & test CSV, check your internet connection. Error: %s", e
            )
        
        print("Uploading dataframe: %s" % dataframe)
        mlflow.log_artifact(dataframe, "artifacts/")

if __name__ == '__main__':
    load_file_as_dataframe()