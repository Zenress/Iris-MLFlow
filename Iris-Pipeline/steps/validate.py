"""
Code for validating model performance using validation set that's derived from dataset
"""
import os
import mlflow
import click
import yaml
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

CONFIG_PATH = "../configuration/config.yaml"

def evaluate(
    model:DecisionTreeClassifier,
    X_validate: pd.Series,
    y_validate: pd.Series
    ) -> None:
    """_summary_

    Args:
        model (DecisionTreeClassifier): _description_
        X_validate (pd.Series): _description_
        y_validate (pd.Series): _description_
    """
    mlflow.log_metric(
        "test_accuracy", metrics.accuracy_score(y_validate, model.predict(X_validate))
    )
    mlflow.log_metric(
        "test_score", model.score(X_validate, y_validate)
    )
    
    scores = cross_val_score(
        estimator=model,
        X=X_validate,
        y=y_validate,
        )
    
    mlflow.log_metric(
        "val_score", scores
    )


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