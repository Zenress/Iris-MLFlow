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
import json

JSON_PATH = "Iris-Pipeline/models/"

def compare_models(
    deployed_model: DecisionTreeClassifier,
    current_model: DecisionTreeClassifier,
    X_validate: pd.Series,
    y_validate: pd.Series,
    json_name: str,
    model_name: str,
    ) -> None:
    """_summary_

    Args:
        deployed_model (DecisionTreeClassifier): _description_
        current_model (DecisionTreeClassifier): _description_
        X_validate (pd.Series): _description_
        y_validate (pd.Series): _description_
        json_name (str): _description_
        model_name (str): _description_
    """
    deployed_model_score = metrics.accuracy_score(
        y_validate,
        deployed_model.predict(X_validate)
        )
    current_model_score = metrics.accuracy_score(
        y_validate,
        current_model.predict(X_validate)
        )
    
    if deployed_model_score < current_model_score:
        deployed_cross_scores = cross_val_score(
        estimator=deployed_model,
        X=X_validate,
        y=y_validate,
        )
        
        current_cross_scores = cross_val_score(
        estimator=current_model,
        X=X_validate,
        y=y_validate,
        )
        
        improved_scores = []
        
        for count, (score) in enumerate(deployed_cross_scores):
            if score < current_cross_scores[count]:
                improved_scores.append(current_cross_scores)
        
        if improved_scores > deployed_cross_scores:
            with open(json_name, 'r') as openfile:
                json_object = json.load(openfile)
                
            json_object['name'] = model_name
            
            with open(json_name, "w") as file:
                file.write(json_object)
        
def evaluate(
    model:DecisionTreeClassifier,
    X_validate: pd.Series,
    y_validate: pd.Series
    ) -> None:
    """
    Evaluate the validation 
    
    Args:
        model (DecisionTreeClassifier): trained decision tree classifier,
            from the last step
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
    
    for count, (score) in enumerate(scores):
        mlflow.log_metric(
            f"val_score{count}", score
        )


@click.command()
@click.option("--process_run_id")
@click.option("--train_run_id")
@click.option("--config_path")
def task(process_run_id, train_run_id, config_path):
    """
    validate model performance against deployed model

    Args:
        process_run_id (str): run id that the process step generated
        train_run_id (str): run id that the train step generated
        config_path (str): path to the configuration file
    """
    with mlflow.start_run() as mlrun:
        with open(config_path, "r", encoding="UTF-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            
        model_version = train_run_id[:5]    
        json_name = Path(JSON_PATH, cfg["deployment_json_name"])
         
        process_run = mlflow.tracking.MlflowClient().get_run(process_run_id)
        train_run = mlflow.tracking.MlflowClient().get_run(train_run_id)
        
        validate_path = Path(process_run.info.artifact_uri, "validate_data.csv")
        validate_df = pd.read_csv(validate_path)
        
        with open(json_name, 'r') as openfile:
            json_object = json.load(openfile)
        
        model_name = cfg["model_name"] + f"-{model_version}"
        model_path = f"runs:/{train_run_id}/{model_name}"
        deployed_path = f"..models/{json_object['name']}"
        
        print(str(model_path))
        
        dtc_model = mlflow.sklearn.load_model(str(model_path))
        deployed_model = mlflow.sklearn.load_model(str(deployed_path))
        
        X_validate = validate_df[cfg["features"].keys()]
        y_validate = validate_df[cfg["label_name"]]
        
        evaluate(
            model=dtc_model,
            X_validate=X_validate,
            y_validate=y_validate
            )
        
        compare_models(
            deployed_model=deployed_model,
            current_model=dtc_model,
            X_validate=X_validate,
            y_validate=y_validate,
            json_name=json_name,
            model_name=model_name
            )
        
if __name__ == "__main__":
    task()
