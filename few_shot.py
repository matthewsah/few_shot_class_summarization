import os
import json
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import tiktoken

# Ensure the environment variable is set
assert 'OPENAI_API_KEY' in os.environ, "The OpenAI API key must be set in the environment."

# Define the persistent directory for Chroma
CHROMA_DIRECTORY = "./chroma_persistent_directory"
MEAN_CONTENT_TOKEN_LEN = 865.3
tokenizer = tiktoken.encoding_for_model('gpt-4o')

if not os.path.isdir(CHROMA_DIRECTORY):
    # Load examples from the JSONL file
    documents = []
    with open('./data/initial_train_samples.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            if line:
                data = json.loads(line)
                id = str(data['id'])
                data['summary'] = ' '.join(data['summary'])
                del data['project']
                del data['id']
                page_content = json.dumps(data)
                document = Document(page_content=page_content, metadata={"id": id})
                documents.append(document)

    # Initialize the embedding function
    embeddings = OpenAIEmbeddings()

    # Initialize the Chroma vector store with the embedding function
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=CHROMA_DIRECTORY)
else:
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DIRECTORY)

# Create the example selector
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=10, #change to 10
    input_keys=['content']
)

# Test on one sample first -- will need to change when collecting data
first_sample = None
b = False
with open('./data/eval_samples.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        if line:
            first_sample = json.loads(line)
            first_sample['summary'] = ' '.join(first_sample['summary'])
            if b:
                break
            b = True

content, summary = first_sample['content'], first_sample['summary']
print(content, summary)
selected_example_id_dicts = example_selector.select_examples({"content": content})

selected_examples = [json.loads(vectorstore.get(where={'id': selected_example_id_dict['id']})['documents'][0]) for selected_example_id_dict in selected_example_id_dicts]

print(selected_examples[0], type(selected_examples[0]), len(selected_examples[0]))

# Convert selected examples to the format required by FewShotPromptTemplate
formatted_examples = [
    {"content": ex['content'], "summary": ex['summary']}
    for ex in selected_examples
]

print(formatted_examples)

# Create an example prompt template
example_prompt = PromptTemplate(
    input_variables=['content', 'summary'],
    template="content: {{ content }}\nsummary: {{ summary }}",
    template_format="jinja2"
)

# Create the few-shot prompt template using the selected examples
few_shot_prompt = FewShotPromptTemplate(
    examples=formatted_examples,
    example_prompt=example_prompt,
    prefix="Example summaries:\n", 
    suffix='Now, summarize the following Java class concisely, following the format of the provided examples\ncontent: {{ content }}\nsummary: ',
    input_variables=["content"],
    template_format="jinja2"
)

# Initialize the OpenAI LLM with the gpt-4o model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

input_text = few_shot_prompt.format(content=content)
print("Generated Prompt:")
print(input_text)
with open('input.txt', 'w') as input_file:
    input_file.write(input_text + '\n----------------\n')
messages = [
    ('system', 'You are a helpful assistant that summarizes Java classes.'),
    ('human', input_text)
]
response = llm.invoke(messages)
print(response)
with open('output.txt', 'w') as output_file:
    output_file.write(str(response) + '\n----------------\n')

# # When we are running on the entire dataset we will need to put a summary on each line