"""
Here we orchestrate the steps that should be run
"""
import mlflow

def workflow():
    with mlflow.start_run() as active_run:
        print()

if __name__ == "__main__":
    workflow()
