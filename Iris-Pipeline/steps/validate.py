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

JSON_PATH = "../models/"

def compare_models(
    deployed_model: DecisionTreeClassifier,
    current_model: DecisionTreeClassifier,
    X_validate: pd.Series,
    y_validate: pd.Series,
    json_path: str,
    model_name: str,
    ) -> None:
    """
    Compare 2 models evaluation metrics against eachother to find the best one
    
    Compares the 2 models evaluation metrics against eachother,
    to find out which one has better performance. 
    There is the current model which is the one that was trained in this mlflow run,
    then there is the deployed model which is the currently deployed model.

    Args:
        deployed_model (DecisionTreeClassifier): the model that is used for deployment,
            it's assigned in the json file
        current_model (DecisionTreeClassifier): the model that was trained in the current mlflow run
        X_validate (pd.Series): feature columns from the validation dataset
        y_validate (pd.Series): label column from the validation dataset
        json_path (str): the path to the json file used for deployment configuration
        model_name (str): name of the model trained in this mlflow run
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
        
        print(deployed_cross_scores)
        print(current_cross_scores)
        
        if len(improved_scores) > len(deployed_cross_scores):
            print("Improved scores better than deployed scores")
            with open(json_path, 'r') as openfile:
                json_object = json.load(openfile)
                
            json_object['name'] = model_name
            
            with open(json_path, "w") as file:
                file.write(json_object)
        
def evaluate(
    model:DecisionTreeClassifier,
    X_validate: pd.Series,
    y_validate: pd.Series
    ) -> None:
    """
    Evaluate on the validation set
    
    Evaluates the model from the train step against the validation set,
    to see if the performance is good with unknown data
    
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
    Validate model performance against deployed model

    Orchestrates the validation of model performance. 
    This is done by first evaluating the model using this function:
        evaluate(): Evaluates the model using the validation dataset
    Then it compares the model to the one that is currently deployed using this function:
        compare_models(): Compares the deployed model metrics to the current model metrics
    
    Args:
        process_run_id (str): run id that the process step generated
        train_run_id (str): run id that the train step generated
        config_path (str): path to the configuration file
    """
    with mlflow.start_run() as mlrun:
        with open(config_path, "r", encoding="UTF-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            
        model_version = train_run_id[:5]    
        json_path = Path(JSON_PATH, cfg["deployment_json_name"])
         
        process_run = mlflow.tracking.MlflowClient().get_run(process_run_id)
        train_run = mlflow.tracking.MlflowClient().get_run(train_run_id)
        
        validate_path = Path(process_run.info.artifact_uri, "validate_data.csv")
        validate_df = pd.read_csv(validate_path)
        
        with open(json_path, 'r') as openfile:
            json_object = json.load(openfile)
        
        model_name = cfg["model_name"] + f"-{model_version}"
        model_path = f"runs:/{train_run_id}/{model_name}"
        deployed_path = f"../models/{json_object['name']}"
        
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
            json_path=json_path,
            model_name=model_name
            )
        
if __name__ == "__main__":
    task()
