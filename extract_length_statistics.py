import tiktoken
import json
import matplotlib.pyplot as plt
import statistics

tokenizer = tiktoken.encoding_for_model('gpt-4o')

def extract_token_lengths(jsonl_file):
    token_lengths = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            entry = json.loads(line)
            content = entry['content']
            tokens = tokenizer.encode(content)
            token_lengths.append(len(tokens))
    
    return token_lengths

jsonl_file = './data/data.jsonl'

token_lengths = extract_token_lengths(jsonl_file)
print (f"mean: {statistics.mean(token_lengths)}")
print (f"median: {statistics.median(token_lengths)}")

plt.figure(figsize=(10, 6))
plt.hist(token_lengths, bins=30, range=(0, 50000))
plt.title('Distribution of Token Lengths')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
