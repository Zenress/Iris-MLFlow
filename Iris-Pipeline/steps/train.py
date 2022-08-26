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

def split_dataset(
    dataset_df: pd.DataFrame,
    label_name: str,
    feature_keys: str
    ):
    """
    Split dataset into training, validation and testing set
    """
    X = dataset_df[feature_keys]
    y = dataset_df[label_name]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1
        )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=1
        )
    
    return X_train, y_train, X_test, y_test, X_val, y_val

def train_model(
    dtc_model: DecisionTreeClassifier,
    train_index: int,
    test_index: int,
    X: pd.DataFrame,
    y: pd.Series,
    fold_nr: int
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
    X_train = X.iloc[train_index].values
    X_test = X.iloc[test_index].values
    y_train = y.iloc[train_index].values
    y_test = y.iloc[test_index].values

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
            
        process_run = mlflow.tracking.MlflowClient().get_run(process_run_id.run_id)
        dataset_path = os.path.join(process_run.info.artifact_uri, "process_dataset")
        dataset_df = pd.read_csv(dataset_path)
        
        #TODO: Log validation sets as artifacts
        X_train, y_train, X_test, y_test, X_val, y_val = split_dataset(
            dataset_df=dataset_df,
            label_name=cfg["label_name"],
            feature_keys=cfg["features"].keys()
            )

if __name__ == "__main__":
    train()