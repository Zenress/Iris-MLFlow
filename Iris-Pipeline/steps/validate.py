"""
Code for validating model performance using validation set that's derived from dataset
"""
from pathlib import Path
import mlflow
import click
import yaml
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pickle


def evaluate(
    model:DecisionTreeClassifier,
    X_validate: pd.Series,
    y_validate: pd.Series
    ) -> None:
    """
    Evaluate the validation 
    
    Args:
        model (DecisionTreeClassifier): _description_
        X_validate (pd.Series): validation features gathered from validation_data.csv
        y_validate (pd.Series): validation label gathered from validation_data.csv
    """
    mlflow.log_metric(
        "validation_accuracy", metrics.accuracy_score(y_validate, model.predict(X_validate))
    )
    mlflow.log_metric(
        "validation_score", model.score(X_validate, y_validate)
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
@click.option("--train_run_id")
@click.option("--config_path")
def task(process_run_id, train_run_id, config_path):
    """_summary_

    Args:
        process_run_id (str): run id that the process step generated
        train_run_id (str): run id that the train step generated
    """
    with mlflow.start_run() as mlrun:
        with open(config_path, "r", encoding="UTF-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            
        process_run = mlflow.tracking.MlflowClient().get_run(process_run_id)
        train_run = mlflow.tracking.MlflowClient().get_run(train_run_id)
        
        validate_path = Path(process_run.info.artifact_uri, "validate_data.csv")
        validate_df = pd.read_csv(validate_path)
        
        model_path = f"runs:/{train_run_id}/models"
        
        print(str(model_path))
        
        dtc_model = mlflow.sklearn.load_model(str(model_path))
        
        X_validate = validate_df[cfg["features"].keys()]
        y_validate = validate_df[cfg["label_name"]]
        
        evaluate(
            model=dtc_model,
            X_validate=X_validate,
            y_validate=y_validate
            )
        
        
if __name__ == "__main__":
    task()
