import pandas as pd
import tiktoken

# Define the file path to the JSONL data file
PATH_TO_DATASET = './data/data.jsonl'

# The mean tokens of content
MEAN_CONTENT_TOKEN_LEN = 865.3

# Tokenizer
tokenizer = tiktoken.encoding_for_model('gpt-4o')

# Read the JSONL file into a DataFrame
df = pd.read_json(PATH_TO_DATASET, lines=True)

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

def content_is_less_than_mean_len(content: str):
    return len(tokenizer.encode(content)) < MEAN_CONTENT_TOKEN_LEN

# Split the DataFrame
# 356 eval samples (sample size)
eval_samples = df.iloc[:356]

# 1000 training samples that are all under the length of mean token len
cur_sample = 356
initial_train_samples = []
remaining_samples = []
while len(initial_train_samples) < 1000:
    if content_is_less_than_mean_len(df.iloc[cur_sample]['content']):
        initial_train_samples.append(df.iloc[cur_sample])

    else:
        remaining_samples.append(df.iloc[cur_sample])
    cur_sample += 1

for i in range(cur_sample, len(df)):
    remaining_samples.append(df.iloc[i])

initial_train_samples = pd.DataFrame(data=initial_train_samples)
remaining_samples = pd.DataFrame(data=remaining_samples)

# Save the splits into seperate JSONL files
eval_samples.to_json('./data/eval_samples.jsonl', orient='records', lines=True)
initial_train_samples.to_json('./data/initial_train_samples.jsonl', orient='records', lines=True)
remaining_samples.to_json('./data/remaining_samples.jsonl', orient='records', lines=True)

