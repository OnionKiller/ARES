
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
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, pipeline, AutoTokenizer, AutoConfig

from rank_bm25 import BM25Okapi
import os

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries

from colbert import Indexer, Searcher

######################################################################

datasets = ["nq"] #, "fever", "wow"]
top_k = 1
evaluation_cutoff = 100
max_new_tokens = 32
sampled_documents = 100000
correct_context_relevance_labels = True

RAG_systems_save_folder = "RAG_Systems_Comparison/"

# LLM + Retriever tuples of each RAG system to be evaluated
RAG_systems = [["mosaicml/mpt-7b-instruct", "text-embedding-ada-002"]]
#RAG_systems = [["mosaicml/mpt-7b-instruct", "text-embedding-ada-002"]]
#RAG_systems = [["facebook/rag-sequence-nq", "facebook/rag-sequence-nq"]]

"""RAG_systems = [["facebook/rag-sequence-nq", "facebook/rag-sequence-nq"],
                  ["mosaicml/mpt-7b-instruct", "bm25"], ["mosaicml/mpt-7b-instruct", "text-embedding-ada-002"], ["mosaicml/mpt-7b-instruct", "colbertv2"],
                  ["gpt-3.5-turbo", "bm25"], ["gpt-3.5-turbo", "text-embedding-ada-002"], ["gpt-3.5-turbo", "colbertv2"],
                  ["gpt-4", "bm25"], ["gpt-4", "text-embedding-ada-002"], ["mosaicml/mpt-7b-instruct", "colbertv2"]]"""

if __name__ == '__main__':
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):

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
            user_prompt = "Using the information in the following document, answer the given question:\n\n"
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
            answer_found = 0
            for term in evaluation_set_terms:
                if system_output.find(term) != -1:
                    answer_found = 1 
            
            return answer_found, answer_found
            

        #################################################

        class RAG_System:
            def __init__(self, cfg=None, **kwargs):

                self.cfg = cfg

                #####################################

                self.generative_LLM_selection = cfg[0]
                self.retriever_selection = cfg[1]

                ################################################

                if self.retriever_selection == "bm25":

                    frames = [cfg[4], cfg[2].sample(n=sampled_documents, random_state=42)]
                    dataframe = pd.concat(frames)
                    dataframe = dataframe.drop_duplicates(subset="Document")
                    print("Document Count: " + str(len(dataframe)))

                    document_set = dataframe['Document'].tolist()
                    tokenized_documents = [doc.split() for doc in document_set]
                    bm25_index = BM25Okapi(tokenized_documents)
                    self.retriever = bm25_index

                elif "ada" in self.retriever_selection:
                    if os.path.exists(cfg[3]):
                        def string_to_np_array(s):
                            return np.array(ast.literal_eval(s))

                        #############################

                        #dataframe_with_embeddings = load_dataset(cfg[3])
                        dataframe_with_embeddings = pd.read_csv(cfg[3], sep="\t")
                        dataframe_with_embeddings['embeddings'] = dataframe_with_embeddings['embeddings'].apply(string_to_np_array)
                        print("Loaded embeddings from previous run!")
                        dataframe_with_embeddings = Dataset.from_pandas(dataframe_with_embeddings)
                        breakpoint()
                        dataframe_with_embeddings.add_faiss_index(column="embeddings")
                        self.retriever = dataframe_with_embeddings
                        print("Document Count: " + str(len(dataframe_with_embeddings)))
                    else:
                        #dataframe = cfg[2].drop_duplicates(subset="Document")
                        #print("Generating embeddings from scratch!")
                        frames = [cfg[4], cfg[2].sample(n=sampled_documents, random_state=42)]
                        dataframe = pd.concat(frames)
                        dataframe = dataframe[:100]
                        dataframe = dataframe.drop_duplicates(subset="Document")
                        print("Document Count: " + str(len(dataframe)))
                        #breakpoint()
                        tqdm.pandas(desc="Generating document embeddings...", total=dataframe.shape[0])
                        dataframe['embeddings'] = dataframe["Document"].progress_apply(lambda x: get_embedding(x, model=self.retriever_selection))
                        dataframe =  dataframe[dataframe['embeddings'].apply(lambda x: len(x)) == 1536]
                        #assert len(cfg[2]) == len(dataframe)
                        dataframe = Dataset.from_pandas(dataframe)
                        dataframe.add_faiss_index(column="embeddings")
                        self.retriever = dataframe
                        #breakpoint()
                        #dataframe.save_to_disk(cfg[3])
                        dataframe.to_csv(cfg[3], sep="\t")
                        print("Saved dataframe to: " + cfg[3])
                        assert False
                elif self.retriever_selection == "colbertv2":
                    frames = [cfg[4], cfg[2].sample(n=sampled_documents, random_state=42)]
                    dataframe = pd.concat(frames)
                    dataframe = dataframe.drop_duplicates(subset="Document")

                    collection = dataframe['Document'].tolist()
                    print("Document Count: " + str(len(dataframe)))

                    #########################

                    doc_maxlen = 256
                    query_maxlen = 32
                    nbits = 2
                    kmeans_niters = 4
                    index_path = f"doc_maxlen={doc_maxlen}_query_maxlen={query_maxlen}_nbits={nbits}_kmeans_niters={kmeans_niters}.latest_index"

                    config = ColBERTConfig(
                        doc_maxlen=doc_maxlen,
                        query_maxlen=query_maxlen, 
                        nbits=nbits, 
                        kmeans_niters=kmeans_niters,
                        root="experiments",
                    )
                    indexer = Indexer(checkpoint="/future/u/jonsf/msmarco.psg.kldR2.nway64.ib__colbert-400000", config=config)
                    indexer.index(name=index_path, collection=collection, overwrite=True)
                    index_path = indexer.get_index()
                            
                    searcher = Searcher(index=index_path, collection=collection)
                    self.retriever = searcher

                """elif self.retriever_selection == "facebook/rag-sequence-nq":
                    dataframe = cfg[2].drop_duplicates(subset="Document")
                    tqdm.pandas(desc="Generating document embeddings...", total=dataframe.shape[0])
                    dataframe['embeddings'] = dataframe["Document"].progress_apply(lambda x: get_embedding(x, model="text-embedding-ada-002"))
                    dataframe =  dataframe[dataframe['embeddings'].apply(lambda x: len(x)) == 1536]
                    assert len(cfg[2]) == len(dataframe)
                    dataframe = Dataset.from_pandas(dataframe)
                    dataframe.add_faiss_index(column="embeddings")

                    self.faiss_path = "faiss_indexes/" + self.retriever_selection.replace("/", "-") + "_" + dataset + ".faiss"
                    self.dataset_path = "faiss_indexes/" + self.retriever_selection.replace("/", "-") + "_" + dataset + "_dataset"
                    dataframe.save_faiss_index('embeddings', self.faiss_path)
                    dataframe.save_to_disk("", self.dataset_path)"""

                ################################################

                if self.generative_LLM_selection == "facebook/rag-sequence-nq":
                    self.tokenizer = RagTokenizer.from_pretrained(self.generative_LLM_selection) 
                    self.retriever = RagRetriever.from_pretrained(self.generative_LLM_selection, 
                                                                index_name="exact", #"exact", 
                                                                use_dummy_dataset=False,
                                                                
                                                                #index_name="custom",
                                                                #passages_path=self.dataset_path,
                                                                #index_path=self.faiss_path,
                                                                ) 
                    self.model = RagSequenceForGeneration.from_pretrained(self.generative_LLM_selection, retriever=self.retriever) 
                    self.device = torch.device("cuda:0")
                    self.model.to(self.device)
                    self.model.eval()
                elif self.generative_LLM_selection == "mosaicml/mpt-7b-instruct":
                    config = AutoConfig.from_pretrained(self.generative_LLM_selection, trust_remote_code=True)
                    config.attn_config['attn_impl'] = 'triton'
                    config.init_device = 'cuda:0' # For fast initialization directly on GPU!

                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.generative_LLM_selection,
                        config=config,
                        torch_dtype=torch.bfloat16, # Load model weights in bfloat16
                        trust_remote_code=True
                    )
                    self.model.eval()
                    self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
                    self.pipe = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=config.init_device)

                    INSTRUCTION_KEY = "### Instruction:"
                    RESPONSE_KEY = "### Response:"
                    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                    self.PROMPT_FOR_GENERATION_FORMAT = """{intro}
                    {instruction_key}
                    {instruction}
                    {response_key}
                    """.format(
                        intro=INTRO_BLURB,
                        instruction_key=INSTRUCTION_KEY,
                        instruction="{instruction}",
                        response_key=RESPONSE_KEY,
                    )

                ################################################

            
            def retrieve_documents(self, query: str, documents, top_k=1):
                if self.retriever_selection == "bm25":
                    top_documents = self.retriever.get_top_n(query.split(), documents, n=top_k)
                    assert type(top_documents) == list 
                    assert type(top_documents[0]) == str
                    return top_documents
                elif self.retriever_selection == "facebook/rag-sequence-nq":
                    input_dict = self.tokenizer.prepare_seq2seq_batch(query, return_tensors="pt").to(self.device)
                    encoder_outputs = self.model(input_ids=input_dict["input_ids"]).question_encoder_last_hidden_state
                    outputs = self.retriever.retrieve(question_hidden_states=encoder_outputs.cpu().detach().numpy(), n_docs=top_k)
                    top_documents = outputs[2][0]['text']
                    assert type(top_documents) == list 
                    assert type(top_documents[0]) == str
                    return top_documents
                elif "ada" in self.retriever_selection:
                    question_embedding = np.array(get_embedding(query)).astype(np.float32)
                    scores, samples = self.retriever.get_nearest_examples("embeddings", question_embedding, k=top_k)
                    top_documents = samples["Document"][:]
                    assert type(top_documents) == list 
                    assert type(top_documents[0]) == str
                    return top_documents
                elif "colbertv2" == self.retriever_selection:
                    results = self.retriever.search(query, k=top_k)
                    top_documents = []
                    for passage_id, passage_rank, passage_score in zip(*results):
                        #print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {self.retriever.collection[passage_id]}")
                        top_documents.append(self.retriever.collection[passage_id])
                    assert type(top_documents) == list 
                    assert type(top_documents[0]) == str
                    return top_documents
                else:
                    raise ValueError("Not implemented")

            
            def generate_output(self, query: str, retrieved_documents, documents_to_use=1):
                if "gpt" in self.generative_LLM_selection:
                    llm_answer =  generate_gpt_answer(query, retrieved_documents[:documents_to_use], self.generative_LLM_selection)
                    assert type(llm_answer) == str 
                    return llm_answer
                elif self.generative_LLM_selection == "facebook/rag-sequence-nq":
                    input_dict = self.tokenizer.prepare_seq2seq_batch(query, return_tensors="pt").to(self.device)
                    generated = self.model.generate(input_ids=input_dict["input_ids"]) 
                    llm_answer = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                    return llm_answer
                elif self.generative_LLM_selection == "mosaicml/mpt-7b-instruct":
                    user_prompt = "Using the information in the following document, answer the given question:\n\n"
                    user_prompt += "Question: " + query + "\n"
                    user_prompt += "Document: " + (" ").join(retrieved_documents) + "\n"
                    formatted_example = self.PROMPT_FOR_GENERATION_FORMAT.format(instruction=user_prompt)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        generated_text = self.pipe(formatted_example, max_new_tokens=max_new_tokens, do_sample=True, use_cache=True)[0]['generated_text']
                        generated_text = generated_text.split("Response:")[1]
                        return generated_text
                else:
                    raise ValueError("Not implemented")

                
                

        ######################################################################

        for dataset in datasets:

            if dataset in ['nq', 'fever', "wow"]:
                evaluation_dataset = pd.read_csv(f"../datasets_v2/{dataset}/ratio_1.0_reformatted_full_articles_False_validation_with_negatives.tsv", sep="\t")
                documents_filepath = "../datasets_v2/decompressed_wikipedia_paragraphs.tsv"
                documents_filepath_with_embeddings = documents_filepath.replace(".tsv", "_with_embeddings.tsv")
                if not os.path.exists(documents_filepath_with_embeddings):
                    documents_dataset = pd.read_csv(documents_filepath, sep="\t")
                    documents_dataset['Document'] = documents_dataset['text']
                else:
                    documents_dataset = evaluation_dataset
            #else:
            #    evaluation_dataset = pd.read_csv("../datasets_v2/record/record_validation_with_negatives.tsv", sep="\t")

            RAG_evaluation_sets_collected = []
            for system in RAG_systems:
                
                evaluation_dataset = evaluation_dataset[:evaluation_cutoff]
                system.append(documents_dataset)
                system.append(documents_filepath_with_embeddings)
                system.append(evaluation_dataset)

                evaluated_rag_system = RAG_System(cfg=system)

                context_relevance_labels = []
                answer_faithfulness_labels = []
                answer_relevance_labels = []
                system_outputs = []
                for row in tqdm(range(len(evaluation_dataset))):
                    
                        retrieved_documents = evaluated_rag_system.retrieve_documents(evaluation_dataset.iloc[row]['Query'], evaluation_dataset['Document'].tolist())
                        system_output = evaluated_rag_system.generate_output(evaluation_dataset.iloc[row]['Query'], retrieved_documents)

                        if evaluation_dataset.iloc[row]['Document'] in retrieved_documents[:top_k]:
                            context_relevance_label = 1
                        else:
                            context_relevance_label = 0
                            for doc in retrieved_documents[:top_k]:
                                if doc in evaluation_dataset.iloc[row]['Document']:
                                    context_relevance_label = 1
                        
                        answer_faithfulness_label, answer_relevance_label = evaluate_llm_generation(system_output, evaluation_dataset.iloc[row]['Answer'])

                        print("Query: " + str(evaluation_dataset.iloc[row]['Query']))
                        print("retrieved_documents: " + str(retrieved_documents))
                        print("system_output: " + str(system_output))

                        print("Correct Document: " + str(evaluation_dataset.iloc[row]['Document']))
                        print("Correct Answer: " + str(evaluation_dataset.iloc[row]['Answer']))

                        print("context_relevance_label: " + str(context_relevance_label))
                        print("answer_faithfulness_label: " + str(answer_faithfulness_label))
                        print("answer_relevance_label: " + str(answer_relevance_label))
                        print("-------------------------------------------------")

                        if correct_context_relevance_labels and answer_relevance_label == 1:
                            context_relevance_label = 1

                        context_relevance_labels.append(context_relevance_label)
                        answer_faithfulness_labels.append(answer_faithfulness_label)
                        answer_relevance_labels.append(answer_relevance_label)
                        system_outputs.append(system_output)

                evaluation_dataset_copy = evaluation_dataset.copy()
                evaluation_dataset_copy['Context_Relevance_Label'] = context_relevance_labels
                evaluation_dataset_copy['Answer_Faithfulness_Label'] = answer_faithfulness_labels
                evaluation_dataset_copy['Answer_Relevance_Label'] = answer_relevance_labels

                print("Label Distributions:")
                print(context_relevance_labels.count(1))
                print(context_relevance_labels.count(0))
                print(answer_faithfulness_labels.count(1))
                print(answer_faithfulness_labels.count(0))
                print(answer_relevance_labels.count(1))
                print(answer_relevance_labels.count(0))

                saved_filename = RAG_systems_save_folder + system[0].replace("/","-") + "_" + system[1].replace("/","-") + "_" + dataset + ".tsv"
                evaluation_dataset_copy.to_csv(saved_filename, sep="\t")
                print("Saved file: " + saved_filename)
                RAG_evaluation_sets_collected.append(saved_filename)

            print("Dataset Finished: " + dataset)
            print(RAG_evaluation_sets_collected)
            print("----------------------------------------------------")


        

        

    