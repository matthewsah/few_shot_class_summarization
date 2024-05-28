import pandas as pd

# Define the file path to the JSONL data file
PATH_TO_DATASET = './data/data.jsonl'

# Read the JSONL file into a DataFrame
df = pd.read_json(PATH_TO_DATASET, lines=True)

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the DataFrame
# 356 eval samples (sample size)
eval_samples = df.iloc[:356]
# 1000 training samples
initial_train_samples = df.iloc[356:1356]
# remaining samples
remaining_samples = df.iloc[1356:]

# Save the splits into seperate JSONL files
eval_samples.to_json('./data/eval_samples.jsonl', orient='records', lines=True)
initial_train_samples.to_json('./data/initial_train_samples.jsonl', orient='records', lines=True)
remaining_samples.to_json('./data/remaining_samples.jsonl', orient='records', lines=True)

