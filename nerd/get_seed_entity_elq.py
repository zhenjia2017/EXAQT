"""Script to get seed entities from ELQ.

Output will be a seed entity file and a wikidata qid file:
Note that the program should run under directory of BLINK-master
after building the environment of ELQ (https://github.com/facebookresearch/BLINK/tree/master/elq)
conda activate el4qa
The format of the seed entities is:
ELQ:
{'predictions': predictions, 'timing': run time of predictions}
predictions is a list of ('question id', 'question text', 'entities')
entities is a list of (qid, score, mention)
"""

import pickle
import elq.main_dense as main_dense
import argparse
import json
import time
import truecase
import os
import globals

class EntityLinkELQMatch():
    def __init__(self, models_path):
        self.elq_config = {
            "interactive": False,
            "biencoder_model": elq_models_path + "elq_wiki_large.bin",
            "biencoder_config": elq_models_path + "elq_large_params.txt",
            "cand_token_ids_path": elq_models_path + "entity_token_ids_128.t7",
            "entity_catalogue": elq_models_path + "entity.jsonl",
            "entity_encoding": elq_models_path + "all_entities_large.t7",
            "output_path": "logs/",  # logging directory
            "faiss_index": "none",
            "index_path": elq_models_path + "faiss_hnsw_index.pkl",
            "num_cand_mentions": 10,
            "num_cand_entities": 10,
            "threshold_type": "joint",
            "threshold": -4.5,
        }
        self.models_path = models_path
        self.args = argparse.Namespace(**self.elq_config)
        print("\n\nPrepare data and start...")
        self.models = main_dense.load_models(self.args, logger=None)
        self.id2wikidata = json.load(open(models_path + "id2wikidata.json"))

    def get_entity_prediction(self, id, question):
        question = truecase.get_true_case(question)
        print (question)
        data_to_link = [{'id': id, 'text': question.strip() + '?'}]

        start = time.time()
        predictions = main_dense.run(self.args, None, *self.models, test_data=data_to_link)
        end = time.time() - start

        predictions = [{
            'id': prediction['id'],
            'text': prediction['text'],
            'entities': [(self.id2wikidata.get(prediction['pred_triples'][idx][0]), prediction['scores'][idx],
                          prediction['pred_tuples_string'][idx][1]) for idx in range(len(prediction['pred_triples']))],
        } for prediction in predictions]

        result = {'predictions': predictions, 'timing': end}
        return result

def get_seed_entities_elq(ELQ, path, id, question):
    elq_file = path + '/elq.pkl'
    wiki_ids_file = path + '/wiki_ids_elq.txt'
    wiki_ids = set()
    elq_result = ELQ.get_entity_prediction(id, question)
    elq_dic = {}
    for result in elq_result["predictions"]:
        elq_dic[str(result['id'])] = result["entities"]

    for (qid, score, text) in elq_dic[str(result['id'])]:
        if qid is not None:
            wiki_ids.add((qid, score, text))

    pickle.dump(elq_result, open(elq_file, 'wb'))
    f1 = open(wiki_ids_file, 'w', encoding='utf-8')
    for item in wiki_ids:
        f1.write(str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\n')
    f1.close()

if __name__ == "__main__":
    # prepare data...
    print("\n\nPrepare data and start...")
    cfg = globals.get_config(globals.config_file)
    test = cfg["benchmark_path"] + cfg["test_data"]
    dev = cfg["benchmark_path"] + cfg["dev_data"]
    train = cfg["benchmark_path"] + cfg["train_data"]
    #create the folder for saving required intermediate data for all questions
    os.makedirs(cfg["ques_path"], exist_ok=True)
    in_files = [train, dev, test]

    # pretrained models, indices, and entity embeddings, please download them from https://github.com/facebookresearch/BLINK/tree/master/elq
    elq_models_path = "path of elq models"

    ELQ = EntityLinkELQMatch(elq_models_path)
    for fil in in_files:
        data = json.load(open(fil))
        for question in data:
            QuestionId = str(question["Id"])
            QuestionText = question["Question"]
            #get truecase form of question
            QuestionText = truecase.get_true_case(QuestionText)
            path = cfg["ques_path"] + 'ques_' + str(QuestionId)
            #create the folder for saving required intermediate data for each questions
            os.makedirs(path, exist_ok = True)
            get_seed_entities_elq(ELQ, path, QuestionId, QuestionText)