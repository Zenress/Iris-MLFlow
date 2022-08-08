"""
Here is where transformations to the dataframe go
Remember:
TODO: Docstring for functions
TODO: Function that prepares the data
TODO: Returns split datasets, and label_encoder.classes_
"""
from sklearn.preprocessing import LabelEncoder
def label_encoding_method():
    """
    Encoding the label
    """
    dataset_df[label_name] = label_encoder.fit_transform(dataset_df[label_name])

def shuffle_method():
    """
    Shuffling dataset
    """
    
def split_dataset():
    """
    Split dataset into training, validation and testing set
    """
    