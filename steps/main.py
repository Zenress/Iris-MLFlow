"""
Here we orchestrate the steps that should be run
"""
import os
import mlflow

def workflow():
    with mlflow.start_run() as active_run:
        print('Launching Ingest Step')
        ingest_run = mlflow.run("steps/","ingest.py", env_manager="local")
        ingest_run = mlflow.tracking.MlflowClient().get_run(ingest_run.run_id)
        dataset_path = os.path.join(ingest_run.info.artifact_uri, "irisdata_raw.csv")
        
        #Current parameters: dataframe
        print('Launching Data Processing step')
        process_run = mlflow.run("steps/","process.py", parameters={"dataset_path": dataset_path}, env_manager="local")
        
        print("Launching Training step")
        training_run = mlflow.run("steps/","train.py",parameters={"process_run_id": process_run.run_id}, env_manager="local")
        
        print("Launching Validatiom step")
        
        
if __name__ == "__main__":
    workflow()
