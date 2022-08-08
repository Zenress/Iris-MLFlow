"""
What's needed in this file is:
TODO: redo function with focus on only outputting the dataframe
TODO: Docstring for functions
TODO: Function that takes and reads the data, then outputs it as a dataframe
TODO: Returns: Dataframe (pickled)
"""
from pathlib import Path


def load_file_as_dataframe(
    dataset_name: str,
    column_names: list,
    ):
    """
    Read data from dataset and encode the label column

    Reads the data from the dataset and assigns the header with column_names.
    Then it encodes the categorical label column into a numerical label column.

    Args:
        label_name (str): name of the label column
        dataset_name (str): name of the dataset to read from
        column_names (list): names of all the columns

    Returns:
        pandas.DataFrane: holds all the feature column records for each feature column
        pandas.Series: holds the label column records
    """
    dataset_full_path = Path(DATASET_PATH, dataset_name)

    # Assigning custom column headers while reading the csv file
    dataset_df = pd.read_csv(dataset_full_path, header=None, names=column_names)

    return dataset_df