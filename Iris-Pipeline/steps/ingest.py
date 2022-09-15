"""
Ingest step used to either Download or use the existing dataset.
The name of the dataset is gathered from the config.yaml which dictates the name
The url used for Downloading the dataset is also gathered from config.yaml
"""
import mlflow
import yaml
import requests
from pathlib import Path
import click


@click.command()
@click.option("--config_path")
def load_file_as_dataframe(config_path) -> None:
    """
    Download the dataset or use existing one

    Downloads the dataset if it doesn't already exists,
     otherwise it just passes the currently existing dataset
    """
    with mlflow.start_run() as mlrun:
        with open(config_path, "r", encoding="UTF-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        csv_url = cfg["csv_url"]

        file_name = Path(cfg["dataset_path"], cfg["dataset_name"])
        
        if not file_name.exists():
            print("File doesn't exist")
            print("Downloading Dataset \n--")
            response_object = requests.get(url=csv_url, allow_redirects=True) #TODO: lacks error control, check online
            with open(file=file_name, mode="wb") as file:
                file.write(response_object.content)
        else:
            print("File already exists")
            print("Using existing file")

        mlflow.log_artifact(file_name)

        print(f"Uploading dataframe: {file_name}")


if __name__ == "__main__":
    load_file_as_dataframe()
