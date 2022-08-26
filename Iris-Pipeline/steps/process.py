"""

"""
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import mlflow
import click
import pandas as pd
import yaml
import os

CONFIG_PATH = "../configuration/config.yaml"
DATASET_PATH = "../data/"

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
        dataset_df (pandas.DataFrame): The dataset that was read through pandas,
            and assigned as a dataframe.
        label_name (str): The label column name derived from the configuration file
        
    Returns:
        pandas.Dataframe: Holds the data from Irisdata_raw.csv but,
            with an encoded label column
    """
    label_encoder = LabelEncoder
    dataset_df[label_name] = label_encoder.fit_transform(dataset_df[label_name])
    return dataset_df


def split_stratified_into_train_val_test(
    df_input: pd.DataFrame,
    stratify_colname: str,
    frac_train: float,
    frac_val: float,
    frac_test: float,
    random_state: int
    ) -> pd.DataFrame:
    '''
    Split a Dataframe into 3 sets
    
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Args:
        df_input (pd.DataFrame): Input dataframe to be split.
        stratify_colname (str): Name of the column used for stratification.
            Usually this column would be for the label.
        frac_train (float): fractional value to split the training set
        frac_val (float): fractional value to split the validate set
        frac_test (float): fractional value to split the test set
            The ratios with which the dataframe will be split into train, val, and
            test data. The values should be expressed as float fractions and should
            sum to 1.0.
        random_state (int): None, or RandomStateInstance

    Returns:
        df_train, df_val, df_test (pd.DataFrame):
            Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

@click.command()
@click.option("--dataset_run_id")
def task(dataset_run_id):
    with mlflow.start_run() as mlrun:
        with open(CONFIG_PATH, "r", encoding="UTF-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
        dataset_run = mlflow.tracking.MlflowClient().get_run(dataset_run_id)   
        dataset_path = os.path.join(dataset_run.info.artifact_uri, "irisdata_raw.csv")
        
        dataset_df = pd.read_csv(
            filepath_or_buffer=dataset_path,
            sep=","
            )
        
        encoded_df = label_encoding_method(
            dataset_df=dataset_df,
            label_name=cfg["label_name"]
            )
        
        train_df, validate_df, test_df = split_stratified_into_train_val_test(
            df_input=encoded_df,
            stratify_colname=cfg["label_name"],
            frac_train=0.6,
            frac_val=0.15,
            frac_test=0.25,
            random_state=None
            )
        
        print("Uploading train dataset: %s" % train_df)
        train_path = Path(DATASET_PATH, "train_data.csv")
        train_df.to_csv(train_path, index=False)
        mlflow.log_artifact(train_path)
        
        print("Uploading train dataset: %s" % test_df)
        test_path =  Path(DATASET_PATH, "test_data.csv")
        test_df.to_csv(test_path, index=False)
        mlflow.log_artifact(test_path)
        
        print("Uploading train dataset: %s" % validate_df)
        validate_path = Path(DATASET_PATH, "validate_data.csv")
        validate_df.to_csv(validate_path, index=False)
        mlflow.log_artifact(validate_path)

if __name__ == '__main__':
    task()