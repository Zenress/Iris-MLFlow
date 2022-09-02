"""
Here we orchestrate the steps that should be run
"""
import mlflow
import click

CONFIG_PATH = "../configuration/config.yaml"


@click.command()
@click.option("--graphs")
def workflow(graphs):
    with mlflow.start_run() as active_run:
        print("Launching Ingest Step")
        ingest_run = mlflow.run(
            "steps/",
            "ingest.py",
            parameters={"config_path": CONFIG_PATH},
            env_manager="local"
        )

        print("Launching Data Processing step")
        process_run = mlflow.run(
            "steps/",
            "process.py",
            parameters={
                "dataset_run_id": ingest_run.run_id,
                "config_path": CONFIG_PATH
            },
            env_manager="local",
        )

        print("Launching Training step")
        training_run = mlflow.run(
            "steps/",
            "train.py",
            parameters={
                "process_run_id": process_run.run_id,
                "graphs": graphs,
                "config_path": CONFIG_PATH
            },
            env_manager="local",
        )

        print("Launching Validatiom step")
        validate_run = mlflow.run(
            "steps/",
            "validate.py",
            parameters={
                "process_run_id": process_run.run_id,
                "train_run_id": training_run.run_id,
                "config_path": CONFIG_PATH
            },
            env_manager="local",
        )


if __name__ == "__main__":
    workflow()
