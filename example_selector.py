from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np
from datetime import datetime
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from typing import List

from langchain.vectorstores import FAISS

from typing_extensions import Annotated, TypedDict
from pydantic import Field

sys.path.append('.')

model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings_func = HuggingFaceEmbeddings(model_name=model_name)

llm = get_llm_model() #Set your own model configuration

examples = [
    {"input": "Show me the train with repair priority from 2024-08-04 to 2024-08-09.", "output": "relevant"},
    {"input": "If today is 2023-07-02, what time does train 800102 start and end today?", "output": "not_relevant"},
    {"input": "What time on 2024-08-06 is diagram CN811 spare?", "output": "time"},
    # Increase the sample size if needed 
]

def convert_input_to_string(input_data):
    if isinstance(input_data, dict):
        return ' | '.join(f"{key}: {value}" for key, value in input_data.items())
    return input_data  

example_embeddings = []
for example in examples:
    input_text = convert_input_to_string(example["input"])  # Now handles strings correctly
    embedding = embeddings_func.embed_documents([input_text])[0]  
    example_embeddings.append(embedding)

example_embeddings = np.array(example_embeddings).astype('float32')

dimension = example_embeddings.shape[1]  
faiss_index = faiss.IndexFlatL2(dimension)  
faiss_index.add(example_embeddings)  

def select_similar_examples(query: str, k=5):
    query_embedding = embeddings_func.embed_documents([query])[0]
    query_embedding = np.array([query_embedding]).astype('float32')

    distances, indices = faiss_index.search(query_embedding, k)

    selected_examples = [examples[idx] for idx in indices[0]]

    # Print selected examples for inspection
    print("Selected Examples Based on Query:")
    for ex in selected_examples:
        print(ex)
    print("="*50)

    return selected_examples

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

def create_few_shot_prompt(selected_examples):
    formatted_examples = []
    for ex in selected_examples:
        input_text = convert_input_to_string(ex["input"])  # Handles string input
        formatted_examples.append({
            'input': input_text, 
            'output': ex["output"]
        })
    
    return FewShotPromptTemplate(
        examples=formatted_examples, 
        example_prompt=example_prompt,
        prefix="You are a classifier that categorizes inputs based on their details. Below are several examples showing how each input corresponds to a specific category. Your task is to return the category listed in the 'output' field based on the given 'input'.",
        suffix="Input: {input}\nPlease return the category as listed in the 'output' field. The category is:",
        input_variables=["input"], 
    )

def get_llm_model_for_classification():
    return get_llm_model()  # Replace with your actual LLM initialization

def get_classification_from_llm(few_shot_prompt, input_text):
    formatted_prompt = few_shot_prompt.format(input=input_text)
    
    messages = [
        {"role": "system", "content": "You are a classifier that categorizes inputs based on their details. Below are several examples showing how each input corresponds to a specific category. Your task is to return the category listed in the 'output' field based on the given 'input'."},
        {"role": "user", "content": formatted_prompt}
    ]
    
    result = llm.invoke(messages)
    return result

# New function to process multiple test inputs
def classify_multiple_inputs(test_inputs, k=5):
    results = {}
    for test_input in test_inputs:
        print(f"\nProcessing input: {test_input}")
        similar_examples = select_similar_examples(test_input, k)
        few_shot_prompt = create_few_shot_prompt(similar_examples)
        final_output = get_classification_from_llm(few_shot_prompt, test_input)
        results[test_input] = final_output
        print(f"Final Classification for '{test_input}': {final_output}")
        print("-"*50)
    return results

# Example list of test inputs
test_inputs = [
    "What training process did trainer 221 train the trainee?"
]

# Run the batch classification
classifications = classify_multiple_inputs(test_inputs, k=5)

# # Optionally, print all results at the end
# print("\nSummary of Classifications:")
# for input_text, classification in classifications.items():
#     print(f"Input: {input_text} | Classification: {classification}")