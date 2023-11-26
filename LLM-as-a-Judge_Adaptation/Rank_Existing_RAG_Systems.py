
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

######################################################################

datasets = ["nq", "fever", "record"]

# LLM + Retriever tuples of each RAG system to be evaluated
RAG_systems = [("gpt-3.5-turbo", "bm25"), ("gpt-3.5-turbo", "ada")]

######################################################################

for dataset in datasets:

    if dataset in ["nq", "fever"]:
        evaluation_dataset = load_dataset("kilt_tasks", dataset)['validation']
    else:
        evaluation_dataset = load_dataset("superglue", dataset)['validation']