import pandas_profiling as pp
import pandas as pd
from pathlib import Path
import yaml

REPORT_PATH = "reports/"
CSV_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
REPORT_TITLE = "Iris_DataReport_ver1-0.html"

def task():
    column_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    
    df = pd.read_csv(
        filepath_or_buffer=CSV_URL,
        header=None,
        names=column_names
        )
    
    report_path = Path(REPORT_PATH, REPORT_TITLE)
    data_profile_report = pp.ProfileReport(
        df=df,
        explorative=True,
        )
    
    data_profile_report.to_file(report_path)
    
if __name__ == "__main__":
    task()