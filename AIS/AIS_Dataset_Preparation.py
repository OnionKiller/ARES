
import json
import pandas as pd
from tqdm import tqdm

dataset_chosen = "WoW"
wow_saved_filename = "datasets/WoW/ann_wow_with_dialogue+retrieved_passages.csv"
wow_documents_filename = "datasets/WoW/ais_wow_train_documents.csv"

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

    total_dialogues = []
    total_passages = []
    for row in tqdm(range(len(wow_testing_data))):
        dialogue, retrieved_passage = gather_dialogue_and_retrieved_passage(wow_testing_data.iloc[row]['ex-idx '])
        total_dialogues.append(dialogue)
        total_passages.append(retrieved_passage)

    wow_testing_data['Dialogue'] = total_dialogues
    wow_testing_data['Passage'] = total_passages

    wow_testing_data.to_csv(wow_saved_filename, sep="\t")
    print("Saved file to: " + wow_saved_filename)

    #breakpoint()

    ##################################################

    with open('datasets/WoW/train.json', 'r') as file:
        train_passages_json = json.load(file)

    total_passages_retrieved = []
    for row in tqdm(range(len(train_passages_json))):
        for dialogue_passage in range(len(train_passages_json[row]['dialog'])):
            for retrieved_passage in train_passages_json[row]['dialog'][dialogue_passage]['retrieved_passages']:
                for current_passage_retrieved in list(retrieved_passage.values())[0]:
                    if current_passage_retrieved not in total_passages_retrieved:
                        total_passages_retrieved.append(current_passage_retrieved)

    documents = pd.DataFrame(total_passages_retrieved, columns=["document"])
    documents.to_csv(wow_documents_filename, sep="\t")
    print("Saved file to: " + wow_documents_filename)

    #breakpoint()
    

    