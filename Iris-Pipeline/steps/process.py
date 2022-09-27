"""
Process step that is used for Data processing in the program
"""
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import mlflow
import click
import pandas as pd
import yaml

def label_encoding_method(
    dataset_df: pd.DataFrame,
    label_name: str
    ) -> pd.DataFrame:
    """
    Encode the label
    
    Takes the DataFrame from pandas, along with the label_name from the config file,
    then it uses those 2 parameters to encode the label column,
    into more usable numerical labels
    
    Args:
        dataset_df (pandas.DataFrame): The dataset containing the data,
            that we will use to make a copy and then encode that copy
        label_name (str): The label column name derived from the configuration file
        
    Returns:
        pandas.Dataframe: Holds the data from Irisdata_raw.csv but,
            with an encoded label column
    """
    label_encoder = LabelEncoder
    encoded_df = dataset_df
    encoded_df[label_name] = label_encoder.fit_transform(label_encoder,encoded_df[label_name])
    
    return encoded_df


@click.command()
@click.option("--dataset_run_id")
@click.option("--config_path")
def task(dataset_run_id, config_path):
    """Function that runs the process step

    Starts by loading the configuration file
    Then it loads the dataset using the dataset_run_id
    
    Function that runs the process step by using the following functions:
        label_encoding_method(): Returns encoded pandas dataframe
        split_stratified_into_train_val_test(): Returns 3 dataframes with the dataset_df,
            split into those 3 dataframes
    
    Afterwards the to_csv() function writes the dataframes into files
    
    Args:
        dataset_run_id (str): ID gotten from the last step's run
    """
    with mlflow.start_run() as mlrun:
        with open(config_path, "r", encoding="UTF-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
        mlflow.set_tag("mlflow.runName","Data processing step")
        
        dataset_run = mlflow.tracking.MlflowClient().get_run(dataset_run_id)   
        dataset_path = Path(dataset_run.info.artifact_uri, cfg["dataset_name"])
        
        dataset_df = pd.read_csv(
            filepath_or_buffer=dataset_path,
            header=cfg["dataset_settings"]["header"],
            names=cfg["column_names"],
            )
        
        print(dataset_df.head(2))
        
        encoded_df = label_encoding_method(
            dataset_df=dataset_df,
            label_name=cfg["label_name"],
            )
        
        train_df, val_df = train_test_split(
            encoded_df,
            test_size=cfg["splitting_settings"]["validate_size"],
            train_size=cfg["splitting_settings"]["train_size"],
            shuffle=cfg["splitting_settings"]["shuffle"],
            )
        
        print("Uploading train dataset:")
        print(train_df.head(2))
        train_path = Path(cfg["dataset_path"], cfg["train_data_name"])
        train_df.to_csv(train_path, index=False)
        mlflow.log_artifact(train_path)
        
        print("Uploading validation dataset:")
        print(val_df.head(2))
        validate_path = Path(cfg["dataset_path"], cfg["validate_data_name"])
        val_df.to_csv(validate_path, index=False)
        mlflow.log_artifact(validate_path)

if __name__ == '__main__':
    task()