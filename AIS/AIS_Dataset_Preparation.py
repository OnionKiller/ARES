
import json
import pandas as pd

dataset_chosen = "WoW"
wow_saved_filename = "datasets/WoW/ann_wow_with_dialogue+retrieved_passages.csv"

if dataset_chosen == "WoW":

    def gather_dialogue_and_retrieved_passage(ex_idx):
        convo_no, turn_no = ex_idx.split(":")
        convo_no = int(convo_no)
        turn_no = int(turn_no)
        assert len(ex_idx.split(":")) == 2

        dialogue = [dialogue_and_passages_json[convo_no]['dialog'][i]['text'] for i in range(turn_no) if i < turn_no]
        dialogue = (" ").join(dialogue)

        retrieved_passage = dialogue_and_passages_json[convo_no]['dialog'][turn_no]['checked_sentence']
        assert len(list(retrieved_passage)) == 0
        retrieved_passage_key = list(retrieved_passage)[0]
        retrieved_passage = retrieved_passage[retrieved_passage_key]
        return dialogue, retrieved_passage

    with open('datasets/WoW/test_random_split.json', 'r') as file:
        dialogue_and_passages_json = json.load(file)

    wow_testing_data = pd.read_csv("AIS/ann_wow.csv")

    total_dialogues = []
    total_passages = []
    for row in range(len(wow_testing_data)):
        dialogue, retrieved_passage = gather_dialogue_and_retrieved_passage(wow_testing_data.iloc[row]['ex-idx '])
        total_dialogues.append(dialogue)
        total_passages.append(retrieved_passage)

    wow_testing_data['Dialogue'] = total_dialogues
    wow_testing_data['Passages'] = total_passages

    wow_testing_data.to_csv(wow_saved_filename, sep="\t")
    

    