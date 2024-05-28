import os
import json
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.llms import OpenAI
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# Ensure the environment variable is set
assert 'OPENAI_API_KEY' in os.environ, "The OpenAI API key must be set in the environment."

# Define the persistent directory for Chroma
chroma_directory = "./chroma_persistent_directory"

# Initialize the embedding function
embeddings = OpenAIEmbeddings()

# Initialize the Chroma vector store with the embedding function
chroma = Chroma(persist_directory=chroma_directory, embedding_function=embeddings)

# Load examples from the JSONL file
examples = []
with open('./data/initial_train_samples.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        if line:
            data = json.loads(line)
            data['summary'] = ' '.join(data['summary'])
            examples.append(data.copy())

# Check if the directory is empty to determine if the Chroma vector store is new
is_chroma_new = not os.path.exists(chroma_directory) or not os.listdir(chroma_directory)

# Add examples to the Chroma vector store if it is new
if is_chroma_new:
    for example in examples:
        embedding = embeddings.embed_documents(example['content'])
        chroma.add_texts([example['content']], [embedding])

# Create the example selector
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=chroma,  # Corrected parameter name
    k=10
)

# Test on one sample first -- will need to change when collecting data
first_sample = None
with open('./data/eval_samples.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        if line:
            first_sample = json.loads(line)
            first_sample['summary'] = ' '.join(first_sample['summary'])
            break

project, content, summary = first_sample['project'], first_sample['content'], first_sample['summary']
selected_examples = example_selector.select_examples({"content": content})

# Convert selected examples to the format required by FewShotPromptTemplate
formatted_examples = [
    {"project": ex['project'], "content": ex['content'], "summary": ex['summary']}
    for ex in selected_examples
]

# Create an example prompt template
example_prompt = PromptTemplate(
    input_variables=['project', 'content', 'summary'],
    template="Project: {project}\nContent: {content}\nSummary: {summary}"
)

# Create the few-shot prompt template using the selected examples
few_shot_prompt = FewShotPromptTemplate(
    examples=formatted_examples,
    example_prompt=example_prompt,
    prefix="Summarize the following Java Class.",
    suffix="Project: {project}\nContent: {content}\nSummary:",
    input_variables=["project", "content"]
)

# Initialize the OpenAI LLM with the gpt-3.5-turbo-0125 model
llm = OpenAI(model="gpt-3.5-turbo-0125")

# Now you can use the few_shot_prompt with the LLM
input_text = few_shot_prompt.format(project=project, content=content)
print("Generated Prompt:")
print(input_text)
response = llm.generate([input_text])
print("Generated Response:")
print(response)
with open('output.txt', 'w') as output_file:
    output_file.write(response + '\n')

# When we are running on the entire dataset we will need to put a summary on each line