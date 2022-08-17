"""
Here we orchestrate the steps that should be run
"""
import os
import mlflow

def workflow():
    with mlflow.start_run() as active_run:
        print('Launching Ingest Step')
        ingest_run = mlflow.run("steps/","ingest_data")
        ingest_run = mlflow.tracking.MlflowClient().get_run(ingest_run.run_id)
        dataset_path = os.path.join(ingest_run.info.artifect_uri, "dataset_path")
        
        #Current parameters: dataframe
        print('Launching data processing')
        process_run = mlflow.run("steps/","data_processing", parameters={"data_path": dataset_path})
        process_run = mlflow.tracking.MlflowClient().get_run(process_run.run_id)
        
        
if __name__ == "__main__":
    workflow()
