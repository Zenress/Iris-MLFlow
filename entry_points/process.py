"""
Here is where transformations to the dataframe go
Remember:
TODO: Docstring for functions
TODO: Function that prepares the data
TODO: Returns split datasets, and label_encoder.classes_
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import mlflow
import click
import pandas as pd
import yaml

CONFIG_PATH = "configuration/config.yaml"

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


def shuffle_method(
    encoded_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Shuffle dataframe using pandas
    
    Shuffles the dataframe using pandas sample function,
    then afterward we reset the indexes using pandas reset_index function
    
    Args:
        encoded_df (pandas.DataFrame): encoded dataframe that was returned in,
            the label_encoding_method function
    
    Returns:
        pandas.DataFrame: shuffled dataframe with new indexes
    """
    shuffled_df = encoded_df.sample(frac=1, random_state=123)
    shuffled_df.reset_index(drop=True,inplace=True)
    return shuffled_df
    

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


@click.option("--dataset_path")
def task(dataframe_path):
    with mlflow.start_run() as mlrun:
        with open(CONFIG_PATH, "r", encoding="UTF-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            
        dataset_df = pd.read_csv(
            filepath_or_buffer=dataframe_path,
            sep=","
            )
        
        encoded_df = label_encoding_method(
            dataset_df=dataset_df,
            label_name=cfg["label_name"]
            )
        
        shuffled_df = shuffle_method(encoded_df=encoded_df)
        
        X_train, y_train, X_test, y_test, X_val, y_val = split_dataset(
            dataset_df=shuffled_df,
            label_name=cfg["label_name"],
            feature_keys=cfg["features"].keys()
            )
        

if __name__ == '__main__':
    task()