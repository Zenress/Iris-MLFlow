import pandas as pd

df = pd.read_csv("data/irisdata_raw.csv",header=None, names= ["sepal length", "sepal width", "petal length", "petal width", "class"])
shuffled_df = df.sample(frac=1, random_state=123)
print(df.head())
print(shuffled_df.head())
shuffled_df.reset_index(drop=True,inplace=True)
print(shuffled_df.head())
shuffled_dataset = shuffled_df.to_csv(index=True)
print(shuffled_dataset)