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
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


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
        "max_depth": [2, 3, 4, 5],
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


def train_model(
    dtc_model: DecisionTreeClassifier,
    X: pd.Series,
    y: pd.Series,
    train_index: list,
    test_index: list,
) -> None:
    """
    Train model using kfold indices
    
    Trains model using kfold indices that was gathered from a stratified kfold

    Args:
        dtc_model (DecisionTreeClassifier): model to be trained,
            made with best parameters possible
        X (pd.Series): feature columns from dataframe for training
        y (pd.Series): label column from dataframe for training
        train_index (list/array): indices gathered,
            from splitting the indices_kfold variable
        test_index (list/array): indices gathered,
            from splitting the indices_kfold variable
    """
    X_train = X.iloc[train_index].values
    X_test = X.iloc[test_index].values
    y_train = y.iloc[train_index].values
    y_test = y.iloc[test_index].values

    dtc_model = dtc_model.fit(X_train, y_train)
    mlflow.log_metric(
        "test_accuracy", metrics.accuracy_score(y_test, dtc_model.predict(X_test))
    )
    mlflow.log_metric(
        "test_score", dtc_model.score(X_test, y_test)
    )


def plot_tree(
    dtc_model: DecisionTreeClassifier,
    tree_plot_path: str,
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
    
    return tree_plot_path


@click.command()
@click.option("--process_run_id")
@click.option("--graphs")
@click.option("--config_path")
def task(process_run_id, graphs, config_path) -> None:
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

        mlflow.set_tag("mlflow.runName","Model training step")

        process_run = mlflow.tracking.MlflowClient().get_run(process_run_id)

        version_number = mlrun.info.run_id[:5]

        df_path = Path(process_run.info.artifact_uri, cfg["train_data_name"])
        df = pd.read_csv(df_path)

        X = df[cfg["features"].keys()]
        y = df[cfg["label_name"]]

        max_depth, criterion, splitter = parameter_tuning(
            nr_kfold=cfg["kfold_settings"]["nr_splits"],
            X_train=X,
            y_train=y,
        )

        dtc_model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
        )

        indices_kfold = StratifiedKFold(
            n_splits=cfg["kfold_settings"]["nr_splits"],
            shuffle=cfg["kfold_settings"]["shuffle"],
        )  
         
        for train_index, test_index in indices_kfold.split(X, y):
            train_model(
                dtc_model=dtc_model,
                X=X,
                y=y,
                train_index=train_index,
                test_index=test_index
            )

        tree_plot_path = plot_tree(
            dtc_model=dtc_model,
            tree_plot_path=cfg["tree_plot_path"],
        )
        
        mlflow.log_artifact(tree_plot_path)

        mlflow.sklearn.log_model(
            sk_model=dtc_model,
            artifact_path=cfg["model_name"]+f"-{version_number}",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )


if __name__ == "__main__":
    task()
