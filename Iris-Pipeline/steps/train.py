"""
What's needed in this file is:
"""
import os
import click
import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

CONFIG_PATH = "configuration/config.yaml"


def train_model(
    dtc_model: DecisionTreeClassifier,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_keys,
    label_name: str
    ) -> None:
    """
    Train DecisionTreeClassifier Model using Iris Dataset.

    Trains a DecisionTreeClassifier using a KFolded dataset.
    Splits dataset between feature columns and labels,
    and then further splits them into a training and testing set.

    Args:
        dtc_model (sklearn.tree.DecisionTreeClassifier): An untrained DecisionTreeClassifier,
            used for classifying the KFolded Iris Dataset
        train_index (int) indices for the training side of the kfold split
        test_index (int) indices for the testing side of the kfold split
        X (pandas.DataFrame) The Feature columns of the dataset
        y (pandas.Series) The Label column of the dataset
    """
    X_train = train_df[feature_keys]
    X_test = test_df[feature_keys]
    y_train = train_df[label_name]
    y_test = 

    dtc_model = dtc_model.fit(X_train, y_train)
    print(
        (
            f"Accuracy for fold nr. {fold_nr} on test set:"
            f" {metrics.accuracy_score(y_test, dtc_model.predict(X_test))} - "
            f"Double check: {dtc_model.score(X_test,y_test)}"
        )
    )

click.option("--process_run_id")
def train(process_run_id):
    with mlflow.start_run() as mlrun:
        with open(CONFIG_PATH, "r", encoding="UTF-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            
        process_run = mlflow.tracking.MlflowClient().get_run(process_run_id)
        train_path = os.path.join(process_run.info.artifact_uri, "train_data.csv")
        train_df = pd.read_csv(train_path)
        
        test_path = os.path.join(process_run.info.artifact_uri, "test_data.csv")
        test_df = pd.read_csv(test_path)
        
        dtc_model = DecisionTreeClassifier(
            criterion=cfg["decisiontree_settings"]["criterion"]
        )
        
        train_model(
            dtc_model=dtc_model,
            train_df=train_df,
            test_df=test_df,
            feature_keys=cfg["features"].keys()
            )
        

if __name__ == "__main__":
    train()
