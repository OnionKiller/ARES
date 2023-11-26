
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

dataset_chosen = "CNN_DM" #"WoW"

wow_saved_filename = "datasets/WoW/ann_wow_with_dialogue+retrieved_passages.tsv"
wow_documents_filename = "datasets/WoW/ais_wow_train_documents.tsv"

cnn_dm_saved_filename = "datasets/CNN_DM/ann_cnn_dm_retrieved_passages.tsv"
cnn_dm_documents_filename = "datasets/CNN_DM/cnn_dm_train_documents.tsv"


if dataset_chosen == "WoW":

    def gather_dialogue_and_retrieved_passage(ex_idx):
        convo_no, turn_no = ex_idx.split(":")
        convo_no = int(convo_no)
        turn_no = int(turn_no)
        assert len(ex_idx.split(":")) == 2

        dialogue = [dialogue_and_passages_json[convo_no]['dialog'][i]['text'] for i in range(turn_no) if i < turn_no]
        dialogue = (" ").join(dialogue)

        retrieved_passage = dialogue_and_passages_json[convo_no]['dialog'][turn_no]['checked_sentence']
        assert len(list(retrieved_passage)) == 1
        retrieved_passage_key = list(retrieved_passage)[0]
        retrieved_passage = retrieved_passage[retrieved_passage_key]
        return dialogue, retrieved_passage

    with open('datasets/WoW/test_random_split.json', 'r') as file:
        dialogue_and_passages_json = json.load(file)

    wow_testing_data = pd.read_csv("AIS/ann_wow.csv")
    print("Length of WoW Examples Overall: " + str(len(wow_testing_data)))
    wow_testing_data = wow_testing_data[wow_testing_data['Q1'] == "Yes, I understand it."]
    print("Length of WoW Examples with Understandable Summaries: " + str(len(wow_testing_data)))

    total_dialogues = []
    total_passages = []
    attribution_labels = []
    for row in tqdm(range(len(wow_testing_data))):
        dialogue, retrieved_passage = gather_dialogue_and_retrieved_passage(wow_testing_data.iloc[row]['ex-idx '])
        total_dialogues.append(dialogue)
        total_passages.append(retrieved_passage)

        if wow_testing_data.iloc[row]['Q2'] == "Yes, fully attributable.":
            attribution_labels.append(1)
        else:
            attribution_labels.append(0)

    #wow_testing_data['Dialogue'] = total_dialogues
    #wow_testing_data['Passage'] = total_passages

    print("attribution_labels")
    print(attribution_labels.count(1))
    print(attribution_labels.count(0))
    
    wow_testing_data['Query'] = total_dialogues
    wow_testing_data['Document'] = total_passages
    wow_testing_data['Answer'] = wow_testing_data['output']
    wow_testing_data['Context_Relevance_Label'] = ["1" for _ in range(len(wow_testing_data))]
    wow_testing_data['Answer_Faithfulness_Label'] = attribution_labels

    wow_saved_filename_train = wow_saved_filename.replace(".tsv", "_train.tsv")
    wow_saved_filename_test = wow_saved_filename.replace(".tsv", "_test.tsv")
    wow_testing_data_train, wow_testing_data_test = train_test_split(wow_testing_data, test_size= 200 / len(wow_testing_data), random_state=42)

    wow_testing_data_train.to_csv(wow_saved_filename_train, sep="\t")
    print("Saved file to: " + wow_saved_filename_train)

    wow_testing_data_test.to_csv(wow_saved_filename_test, sep="\t")
    print("Saved file to: " + wow_saved_filename_test)

    #wow_testing_data.to_csv(wow_saved_filename, sep="\t")
    #print("Saved file to: " + wow_saved_filename)

    #breakpoint()

    ##################################################

    with open('datasets/WoW/train.json', 'r') as file:
        train_passages_json = json.load(file)

    total_passages_retrieved = []
    for row in tqdm(range(len(train_passages_json))):
        for dialogue_passage in range(len(train_passages_json[row]['dialog'])):
            for retrieved_passage in train_passages_json[row]['dialog'][dialogue_passage]['retrieved_passages']:
                for current_passage_retrieved in list(retrieved_passage.values())[0]:
                    total_passages_retrieved.append(current_passage_retrieved)

    documents = pd.DataFrame(total_passages_retrieved, columns=["document"])
    documents.to_csv(wow_documents_filename, sep="\t")
    print("Saved file to: " + wow_documents_filename)

    #breakpoint()

elif dataset_chosen == "CNN_DM":

    from datasets import load_dataset
    dataset = load_dataset("cnn_dailymail", '3.0.0')['test']
    #dataset = dataset.set_index("id")

    cnn_dm_testing_data = pd.read_csv("AIS/ann_cnn_dm.csv")
    print("Length of CNN_DM Examples Overall: " + str(len(cnn_dm_testing_data)))
    cnn_dm_testing_data = cnn_dm_testing_data[cnn_dm_testing_data['Q1'] == "Yes, I understand it."]
    print("Length of CNN_DM Examples with Understandable Summaries: " + str(len(cnn_dm_testing_data)))

    articles = []
    attribution_labels = []

    for row in range(len(cnn_dm_testing_data)):
        #retrieved_article = dataset[cnn_dm_testing_data.iloc[row]['doc-url-hash']]
        hashed_doc_url = cnn_dm_testing_data.iloc[row]['doc-url-hash']
        retrieved_article = dataset.filter(lambda example: example['id'] == hashed_doc_url)
        try:
            assert len(retrieved_article) == 1
            articles.append(retrieved_article['article'])
        except:
            print("Could not find: " + hashed_doc_url)
            print("Articles Found: " + str(len(retrieved_article)))
            assert False

        if cnn_dm_testing_data.iloc[row]['Q2'] == "Yes, fully attributable.":
            attribution_labels.append(1)
        else:
            attribution_labels.append(0)

    print("attribution_labels")
    print(attribution_labels.count(1))
    print(attribution_labels.count(0))
    
    cnn_dm_testing_data['Query'] = [" " for _ in range(len(articles))]
    cnn_dm_testing_data['Document'] = articles
    cnn_dm_testing_data['Answer'] = cnn_dm_testing_data['output']
    cnn_dm_testing_data['Context_Relevance_Label'] = ["1" for _ in range(len(articles))]
    cnn_dm_testing_data['Answer_Faithfulness_Label'] = attribution_labels

    #######################################

    cnn_dm_saved_filename_train = cnn_dm_saved_filename.replace(".tsv", "_train.tsv")
    cnn_dm_saved_filename_test = cnn_dm_saved_filename.replace(".tsv", "_test.tsv")
    cnn_dm_testing_data_train, cnn_dm_testing_data_test = train_test_split(cnn_dm_testing_data, test_size= 200 / len(cnn_dm_testing_data), random_state=42)

    cnn_dm_testing_data_train.to_csv(cnn_dm_saved_filename_train, sep="\t")
    print("Saved file to: " + cnn_dm_saved_filename_train)

    cnn_dm_testing_data_test.to_csv(cnn_dm_saved_filename_test, sep="\t")
    print("Saved file to: " + cnn_dm_saved_filename_test)

    #cnn_dm_testing_data.to_csv(cnn_dm_saved_filename, sep="\t")
    #print("Saved file to: " + cnn_dm_saved_filename)

    #######################################

    documents = pd.DataFrame(articles, columns=["document"])
    documents.to_csv(cnn_dm_documents_filename, sep="\t")
    print("Saved file to: " + cnn_dm_documents_filename)



    
    """import json

    with open('datasets/finished_files/test.bin', 'rb') as file:
        binary_data = file.read()


    json_string = binary_data.decode('utf-8')

    import cPickle
    import cPickle as pickle
    import struct

    with open('datasets/finished_files/chunked/train_167.bin', 'rb') as file:
        #binary_data = file.readlines()
        binary_data = file.read()
        #binary_data = pickle.load(file.read())

    loaded_data = struct.unpack(binary_data)"""
    

    