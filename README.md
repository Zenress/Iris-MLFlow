# **Iris MLFlow Classification**

This project is a **Classification model** using the **Iris Dataset**. The model used is a **DecisionTree** and we use MLFlow to manage the Machine Learning life cycle.

The project is run locally and will therefore use storage on your computer

## Table of Contents

- [**Iris MLFlow Classification**](#iris-mlflow-classification)
  - [Table of Contents](#table-of-contents)
  - [Dependencies](#dependencies)
  - [The libraries used are](#the-libraries-used-are)
  - [Setup](#setup)
    - [Installation](#installation)
    - [Removing the Conda Environment](#removing-the-conda-environment)
  - [Project usage](#project-usage)
    - [Running the project](#running-the-project)
      - [Configuration](#configuration)
      - [MLFlow pipeline usage](#mlflow-pipeline-usage)
      - [Predicting](#predicting)
  - [Dataset](#dataset)
  - [Credits](#credits)
  - [Sources](#sources)

## Dependencies

- *Python 3.9.12*
- *Pip 21.0.1*

## The libraries used are

- *MLFlow 1.28.0*
- *Requests*
- *Pathlib*
- *Pandas 1.4.2*
- *SKlearn 1.0.2*
- *Yaml*
- *Click*

## Setup

### Installation

To run this project you should create a **Conda Environment** (<https://www.anaconda.com/products/distribution>) to run it on. This will help with making sure it can run in it's default configuration

The **Conda Environment** file `Iris-Pipeline/configuration/env.yaml` makes it easy to create the environment to run this project.

To create the environment, you should type the command in console:

```console
conda env create -f Iris-Pipeline/configuration/env.yaml
```

After having created the environment you also have to activate it:

```console
conda activate Iris_MLFLOW
```

### Removing the Conda Environment

To delete the conda environment you will first have to run a specific command:

´´´
conda env remove -n Iris_MLFLOW
´´´

Afterwards you have to locate where the environment folder is. By default it is under: `C:\Users\YourUserHere\Anaconda3\envs`

You then delete the folder that matches the name of the Environment files configured name, by default it's name would be: `Iris_MLFLOW`

## Project usage

This project is a Machine Learning Pipeline managed using MLFlow, the pipeline itself is also run using MLFlow

### Running the project

**Note:** *The commands below were run in an Anaconda prompt but can be run in a terminal / console just fine.*

#### Configuration

There is an included config.yaml file under the configuration folder. This file is where you would change different variables easily. Example:

```yaml
features: 
  sepal length: {min: 4.3, max: 7.9}
  sepal width: {min: 2, max: 4.4}
  petal length: {min: 1, max: 6.9}
  petal width: {min: 0.1, max: 2.5}
```

#### MLFlow pipeline usage

To use the pipeline you should run the following command:

```mlflow
mlflow run iris-pipeline
```

This will run the following steps:

- Ingest: Checks if the dataset already exists, if it doesn't then the dataset is downloaded
- Process: Data processing step that does a number of data transformations along with reading the dataset
- Training: Trains the DecisionTreeClassifier Model and makes sure it is training with the best configuration
- Validate: Validates that the training is done in a way that works well on new and unknown data

#### Predicting

This program is meant to  Train a model/Retrain a model using the Iris Dataset

To predict on the model used, you'll have to serve the model as a prediction service.

Insert model serving command here

## Dataset

The dataset used is the Iris Dataset (<https://archive.ics.uci.edu/ml/datasets/Iris>)

- 5 columns, headers added later on
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
  - Class
    - Iris Setosa
    - Iris Versicolor
    - Iris Virginica

The names we're derived from the documentation under **Attribute information** gathered here: <https://archive.ics.uci.edu/ml/datasets/Iris>

## Credits

- Credit to **UCI** for making the dataset widely accessible.
- Credit to **Michele Stawowy** for **Quality Assurance and Guidance**
- Credit to **Martin Riishøj Mathiasen** for the idea to **KFold crossvalidation**

## Sources

- Iris Dataset can be found here: <https://archive.ics.uci.edu/ml/datasets/Iris>
- DecisionTreeClassifier Inspiration: <https://www.datacamp.com/tutorial/decision-tree-classification-python>
- Anaconda: <https://www.anaconda.com/>

---
