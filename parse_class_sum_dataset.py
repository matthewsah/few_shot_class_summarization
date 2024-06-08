import pandas as pd
import tiktoken
import json
import statistics

# Define the file path to the JSONL data file
PATH_TO_DATASET = './data/data.jsonl'

# Tokenizer
tokenizer = tiktoken.encoding_for_model('gpt-4o')

def get_mean_content_len(jsonl_file):
    token_lengths = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            entry = json.loads(line)
            content = entry['content']
            tokens = tokenizer.encode(content)
            token_lengths.append(len(tokens))
    
    return statistics.mean(token_lengths)

def content_is_less_than_mean_len(content: str):
    return len(tokenizer.encode(content)) < MEAN_CONTENT_TOKEN_LEN

# The mean tokens of content
MEAN_CONTENT_TOKEN_LEN = get_mean_content_len(PATH_TO_DATASET)
print(MEAN_CONTENT_TOKEN_LEN)


# Read the JSONL file into a DataFrame
df = pd.read_json(PATH_TO_DATASET, lines=True)

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the DataFrame
# eval_samples: 356 eval samples (sample size)
eval_samples = df.iloc[:356]

# initial_train_samples: 1000 samples under the length of mean token len which will be stored in the vectorstore
# remaining_samples: remaining samples
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

