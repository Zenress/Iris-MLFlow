"""
Here is where transformations to the dataframe go
Remember:
TODO: Docstring for functions
TODO: Function that prepares the data
TODO: Returns split datasets, and label_encoder.classes_
"""
from sklearn.preprocessing import LabelEncoder
from mlflow.tracking import MlflowClient
import click

def label_encoding_method():
    """
    Encoding the label
    """
    label_encoder = LabelEncoder
    dataset_df[label_name] = label_encoder.fit_transform(dataset_df[label_name])
    

def shuffle_method():
    """
    Shuffling dataset
    """
    
def split_dataset():
    """
    Split dataset into training, validation and testing set
    """

@click.option("--dataframe_path")
def task(dataframe_path):
    with mlflow.start_run() as mlrun:
        artifacts = [f.path for f in MlflowClient().list_artifacts()]

if __name__ == '__main__':
    task()