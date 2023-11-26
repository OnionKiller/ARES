
import requests
import time
import pandas as pd
import ast
import json
import copy
import openai
from tqdm import tqdm
import csv
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import re
import random
import math

from rank_bm25 import BM25Okapi

######################################################################

def get_embedding(text, model="text-embedding-ada-002"):
    #text = text.replace("\n", " ")
    #if len(text) > 50:
    #    text = (" ").join(text.split(" ")[:50])
    for _ in range(5):
        try:
            return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
        except:
            print("Error generating embedding! Attempting again...")
            time.sleep(30)

#################################################

def generate_gpt_answer(query: str, documents: str, model_choice: str):
    #time.sleep(1)
    user_prompt += "Using the information in the following document, answer the given question:\n\n"
    user_prompt += "Question: " + query + "\n"
    user_prompt += "Document: " + (" ").join(documents) + "\n"
    user_prompt += "Answer: "

    messages = [
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = openai.ChatCompletion.create(
            #model="gpt-3.5-turbo-16k","gpt-4"
            model=model_choice,
            messages=messages,
            temperature=0.0
        )
                
    final_response = response["choices"][0]["message"]["content"]
    return final_response

#################################################

def evaluate_llm_generation(system_output: str, evaluation_set_answer: str):

    evaluation_set_terms = [evaluation_set_answer.lower()]
    evaluation_set_terms += evaluation_set_answer.lower().split(" ")

    system_output = system_output.lower()
    answer_found = False
    for term in evaluation_set_terms:
        if system_output.find(term) != -1:
            answer_found = True 
    return answer_found
    

#################################################

class RAG_System:
    def __init__(self, cfg=None, **kwargs):

        self.cfg = cfg

        #####################################

        self.generative_LLM_selection = cfg[0]
        self.retriever_selection = cfg[1]

        if self.generative_LLM_selection == "facebook/rag-sequence-nq":
            raise ValueError("Not implemented")
            self.model = ""
            self.device = torch.device("cuda:0")
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.get('pretrained_model_name'), model_max_length=cfg.get("evaluation_max_seq_len"))

        if self.retriever_selection == "bm25":
            document_set = cfg[2]['Document'].tolist()
            tokenized_documents = [doc.split() for doc in document_set]
            bm25_index = BM25Okapi(tokenized_documents)
            self.retriever = bm25_index
        elif "ada" in self.retriever_selection:
            dataframe = cfg[2].drop_duplicates(subset="Document")
            tqdm.pandas(desc="Generating document embeddings...", total=dataframe.shape[0])
            dataframe['embeddings'] = dataframe["document"].progress_apply(lambda x: get_embedding(x, model=self.retriever_selection))
            dataframe =  dataframe[dataframe['embeddings'].apply(lambda x: len(x)) == 1536]
            assert len(cfg[2]) == len(dataframe)
            dataframe = Dataset.from_pandas(dataframe)
            dataframe.add_faiss_index(column="embeddings")
            self.retriever = dataframe

    
    def retrieve_documents(self, query: str, documents: List[str], top_k=1) -> List[str]:
        if self.retriever_selection == "bm25":
            top_documents = self.retriever.get_top_n(query.split(), documents, n=top_k)
            assert type(top_documents) == list 
            assert type(top_documents[0]) == str
            return top_documents
        else:
            question_embedding = np.array(get_embedding(query)).astype(np.float32)
            scores, samples = self.retriever.get_nearest_examples("embeddings", question_embedding, k=top_k)
            top_documents = samples["document"][:]
            assert type(top_documents) == list 
            assert type(top_documents[0]) == str
            return top_documents

    
    def generate_output(self, query: str, retrieved_documents: List[str], documents_to_use=1) -> str:
        if "gpt" in self.generative_LLM_selection:
            return generate_gpt_answer(query, retrieved_documents[:documents_to_use], self.generative_LLM_selection)
        elif self.generative_LLM_selection == "facebook/rag-sequence-nq":
            raise ValueError("Not implemented")

        
        

######################################################################

datasets = ["nq", "fever", "record"]
top_k = 1

# LLM + Retriever tuples of each RAG system to be evaluated
RAG_systems = [["gpt-3.5-turbo", "bm25"], ["gpt-3.5-turbo", "text-embedding-ada-002"]]
RAG_systems_save_folder = "RAG_Systems_Comparison/"

######################################################################

for system in RAG_systems:

    for dataset in datasets:

        if dataset in ['nq', 'fever']:
            evaluation_dataset = pd.read_csv(f"../datasets_v2/{dataset}/ratio_1.0_reformatted_full_articles_False_validation_with_negatives.tsv", sep="\t")
        else:
            evaluation_dataset = pd.read_csv("../datasets_v2/record/record_validation_with_negatives.tsv", sep="\t")

        system.append(evaluation_dataset)

        evaluated_rag_system = RAG_System(cfg=system)

        context_relevance_labels = []
        answer_faithfulness_labels = []
        answer_relevance_labels = []
        system_outputs = []
        for row in tqdm(range(len(evaluation_dataset))):
            
            retrieved_documents = evaluated_rag_system.retrieve_documents(evaluation_dataset['Question'], evaluation_dataset['Document'])
            system_output = evaluated_rag_system.generate_output(evaluation_dataset['Question'], retrieved_documents)

            if evaluation_dataset['Document'] in retrieved_documents[:top_k]:
                context_relevance_label = 1
            else:
                context_relevance_label = 0
            
            answer_faithfulness_label, answer_relevance_label = evaluate_llm_generation(system_output, evaluation_dataset['Answer'])

            context_relevance_labels.append(context_relevance_label)
            answer_faithfulness_labels.append(answer_faithfulness_label)
            answer_relevance_labels.append(answer_relevance_label)
            system_outputs.append(system_output)

        evaluation_dataset_copy = evaluation_dataset.copy()
        evaluation_dataset_copy['Context_Relevance_Label'] = context_relevance_labels
        evaluation_dataset_copy['Answer_Faithfulness_Label'] = answer_faithfulness_labels
        evaluation_dataset_copy['Answer_Relevance_Label'] = answer_relevance_labels
        saved_filename = RAG_systems_save_folder + system[0] + "_" + system[1] + ".tsv"
        evaluation_dataset_copy.to_csv(saved_filename, sep="\t")
        print("Saved file: " + saved_filename)


        

        

    