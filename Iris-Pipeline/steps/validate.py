"""
Code for validating model performance using validation set that's derived from dataset
"""
import os
import mlflow
import click
import yaml
import pandas as pd
import os

CONFIG_PATH = "../configuration/config.yaml"

@click.command()
@click.option("--process_run_id")
def task(process_run_id):
    """_summary_

    Args:
        process_run_id (_type_): _description_
    """
    with mlflow.start_run() as mlrun:
        with open(CONFIG_PATH, "r", encoding="UTF-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            
        process_run = mlflow.tracking.MlflowClient().get_run(process_run_id)
        
        validate_path = os.path.join(process_run.info.artifact_uri, "validate_data.csv")
        validate_df = pd.read_csv(validate_path)
        
        X_validate = validate_df[cfg["features"].keys()]
        y_validate = validate_df[cfg["label_name"]]
        
if __name__ == "__main__":
    task()