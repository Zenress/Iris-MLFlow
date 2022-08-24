"""
Here we orchestrate the steps that should be run
"""
from concurrent.futures import process
import os
import mlflow

def workflow():
    with mlflow.start_run() as active_run:
        print('Launching Ingest Step')
        ingest_run = mlflow.run("steps/","ingest.py")
        ingest_run = mlflow.tracking.MlflowClient().get_run(ingest_run.run_id)
        dataset_path = os.path.join(ingest_run.info.artifact_uri, "dataset_path")
        
        #Current parameters: dataframe
        print('Launching Data Processing step')
        process_run = mlflow.run("steps/","process.py", parameters={"dataset_path": dataset_path})
        
        print("Launching Training step")
        training_run = mlflow.run("steps/","train.py",parameters={"process_run_id": process_run.run_id})
        
        print("Launching Validatiom step")
        
        
if __name__ == "__main__":
    workflow()
