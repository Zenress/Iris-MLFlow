"""
Training step used for training the model using the data,
    that was prepared from the last step
"""
from pathlib import Path
from typing import Tuple
import click
import mlflow
import pandas as pd
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt


def parameter_tuning(
    nr_kfold: int,
    X_train: pd.Series,
    y_train: pd.Series,
) -> Tuple[int, str, str]:
    """Tune parameters for use in DecisionTreeClassifier model

    Tries to find the best parameters to run the DecisionTreeClassifier,
        using GridSearchCV

    Args:
        nr_kfold (int): number of kfolds
        X_train (pd.Series): Feature columns to train on
        y_train (pd.Series): Label column to train against

    Returns:
        Tuple: with datatypes in this order:
            int: max_depth
            str: criterion
            str: splitter
    """
    parameters = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    tree = DecisionTreeClassifier()
    grid = GridSearchCV(tree, parameters, cv=nr_kfold)
    grid.fit(X_train, y_train)

    best_max_depth = grid.best_params_["max_depth"]
    best_criterion = grid.best_params_["criterion"]
    best_splitter = grid.best_params_["splitter"]

    mlflow.log_param(f"best_max_depth", best_max_depth)
    mlflow.log_param(f"best_criterion", best_criterion)
    mlflow.log_param(f"best_splitter", best_splitter)

    return best_max_depth, best_criterion, best_splitter


def k_fold_cross_validation(
    dtc_model: DecisionTreeClassifier,
    X_train: pd.Series,
    y_train: pd.Series,
    nr_kfold: int,
) -> None:
    """
    Cross Validate data and model to the a score to figure out the quality

    Cross validate data and model,
        to get a score that tells us how well the data works

    Args:
        dtc_model (DecisionTreeClassifier): Untrained DTC Model
        X_train (pd.Series): Feature columns that are used for training
        y_train (pd.Series): Label column that are used for training against
        nr_kfold (int): The nr of kfold splits
    """
    kfold_scores = cross_val_score(
        estimator=dtc_model,
        X=X_train,
        y=y_train,
        cv=nr_kfold,
    )

    mlflow.log_metric("cross_val_average_accuracy", kfold_scores.mean())
    mlflow.log_metric("cross_val_std_accuracy", kfold_scores.std())


def train_model(
    dtc_model: DecisionTreeClassifier,
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Train DecisionTreeClassifier Model using 2 split dataframes

    Trains a DecisionTreeClassifier using 2 split dataframes,
    assigned to 4 pandas series. These 4 series are used for training the model

    Args:
        dtc_model (DecisionTreeClassifier): Untrained DTC model
        X_train (pd.Series): Feature columns for training
        X_test (pd.Series): Feature columns for testing accuracy
        y_train (pd.Series): Label column for training against
        y_test (pd.Series): Label column for testing against
    """
    dtc_model = dtc_model.fit(X_train, y_train)
    print(
        (
            # TODO: Add for loop
            f"Accuracy for test set:"
            f" {metrics.accuracy_score(y_test, dtc_model.predict(X_test))} - "
            f"Double check: {dtc_model.score(X_test,y_test)}"
        )
    )
    mlflow.log_metric(
        "test_accuracy", metrics.accuracy_score(y_test, dtc_model.predict(X_test))
    )
    mlflow.log_metric(
        "test_score", dtc_model.score(X_test, y_test)
    )


def plot_and_log_model(
    dtc_model: DecisionTreeClassifier,
    tree_plot_path: str
) -> None:
    """
    Plot tree figure and save model using sklearn under MLFlow

    Plots the tree figure and logs it.
    Afterwards it also logs the model using sklearn's own function.
        This is done through MLFlow.sklearn.

    Args:
        dtc_model (DecisionTreeClassifier): Trained model from train_model()
    """
    # Creates the tree topology
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    tree.plot_tree(dtc_model)

    fig.savefig(tree_plot_path)
    
    # Track the decision tree image
    mlflow.log_artifact(tree_plot_path)

    mlflow.sklearn.log_model(
        sk_model=dtc_model,
        artifact_path="models",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
    )


@click.command()
@click.option("--process_run_id")
@click.option("--config_path")
def task(process_run_id, config_path) -> None:
    """
    Task function that orchestrates the training step

    Task function organises the training step.
    Following functions and their order of execution:
        Parameter_tuning(): Checks for the best possible parameters,
            to use on the DecisionTreeClassifier
        Model_creation(): Creates the model using the variables,
            from Parameter_tuning()
        K_fold_cross_validation(): Cross validates on the data and model
        Train_model(): Trains the model using the split dataframes,
            that was further split into 4 Pandas series
        Save_model(): saves the trained model using the Sklearn save_model function

    Args:
        process_run_id (str): the run id from the process step passed through main step
    """
    with mlflow.start_run() as mlrun:
        with open(config_path, "r", encoding="UTF-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        process_run = mlflow.tracking.MlflowClient().get_run(process_run_id)

        train_path = Path(process_run.info.artifact_uri, "train_data.csv")
        train_df = pd.read_csv(train_path)

        test_path = Path(process_run.info.artifact_uri, "test_data.csv")
        test_df = pd.read_csv(test_path)

        X_train = train_df[cfg["features"].keys()]
        X_test = test_df[cfg["features"].keys()]
        y_train = train_df[cfg["label_name"]]
        y_test = test_df[cfg["label_name"]]

        max_depth, criterion, splitter = parameter_tuning(
            nr_kfold=cfg["kfold_nr_splits"],
            X_train=X_train,
            y_train=y_train,
        )

        dtc_model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
        )

        k_fold_cross_validation(
            dtc_model=dtc_model,
            X_train=X_train,
            y_train=y_train,
            nr_kfold=cfg["kfold_nr_splits"],
        )

        train_model(
            dtc_model=dtc_model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        plot_and_log_model(
            dtc_model=dtc_model,
            tree_plot_path=cfg["tree_plot_path"]
        )


if __name__ == "__main__":
    task()
