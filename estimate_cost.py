import tiktoken
import json

model_name = "gpt-4o"

encoding = tiktoken.encoding_for_model(model_name)

# Get average length of training samples
train_tokens = 0
output_tokens = 0
train_samples = 0
max_tokens_per_sample = 0
with open('./data/initial_train_samples.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        if line:
            sample = json.loads(line)
            sample['summary'] = ' '.join(sample['summary'])
            train_tokens += len(encoding.encode('summary: ' + str(sample)))
            max_tokens_per_sample = max(max_tokens_per_sample, len(encoding.encode('summary: ' + str(sample))))
            train_samples += 1
            output_tokens += len(encoding.encode(f'summary: {sample["summary"]}'))
avg_input_tokens_per_sample = train_tokens * 1.0 / train_samples
avg_output_tokens_per_sample = output_tokens * 1.0 / train_samples
print(f'avg input tokens per sample: {avg_input_tokens_per_sample}')
print(f'avg output tokens per sample: {avg_output_tokens_per_sample}')


total_input_tokens = 0
total_output_tokens = 0
with open('./data/eval_samples.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        if line:
            sample = json.loads(line)
            del sample['summary']
            total_input_tokens += len(encoding.encode(str(sample))) + 10 * avg_input_tokens_per_sample
            total_output_tokens += avg_output_tokens_per_sample
print('total input tokens: {total_input_tokens}')
print('total output tokens: {total_output_tokens}')

cost_per_one_mil_input_tokens = 5
cost_per_one_mil_output_tokens = 15
input_cost = total_input_tokens / 1000000 * cost_per_one_mil_input_tokens
output_cost = total_output_tokens / 1000000 * cost_per_one_mil_output_tokens
print(f'input cost: {input_cost}')
print(f'output cost: {output_cost}')

print(f'total cost: {input_cost + output_cost}')

print(max_tokens_per_sample)
print(max_tokens_per_sample * 11)