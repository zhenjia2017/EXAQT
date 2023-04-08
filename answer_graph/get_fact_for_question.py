"""Script to read seed entities of each question and extract triples for the seed entities.

Output will be a txt SPO file for each question with the following format:
statement_id - ps:||subject_qid||label||ps_qid||ps_qid||object_qid||label
statement_id - pq:||subject_qid||label||ps_qid||pq_qid||object_qid||label
"""
import json
import os
import globals
import time

from get_CLOCQ_Wikidata_SPOs import get_wikidata_tuplesfromclocq, write_clocqspo_to_file

def get_spo(path):
    tagme_wiki_ids_file = path + "/wiki_ids_tagme_new.txt"
    elq_wiki_ids_file = path + "/wiki_ids_elq.txt"
    spo_file = path + "/SPO_new.txt"
    wiki_ids = set()
    if os.path.exists(tagme_wiki_ids_file):  # line A
        with open(tagme_wiki_ids_file) as f:
            for line in f:
                entity, score, text, label = line.strip().split('\t')
                wiki_ids.add((entity, score, text))
    if os.path.exists(elq_wiki_ids_file):  # line A
        #print('ELQ seed entity' + ' exists.')
        with open(elq_wiki_ids_file) as f:
            for line in f:
                entity, score, text = line.strip().split('\t')
                wiki_ids.add((entity, score, text))
    fact_dic = get_wikidata_tuplesfromclocq(wiki_ids)
    write_clocqspo_to_file(fact_dic, spo_file, wiki_ids)
    #print("\n\nSPOs generated...")


if __name__ == "__main__":
    # prepare data...
    print("\n\nPrepare data and start...")
    cfg = globals.get_config(globals.config_file)
    test = cfg["data_path"] + cfg["test_data"]
    dev = cfg["data_path"] + cfg["dev_data"]
    train = cfg["data_path"] + cfg["train_data"]
    in_files = [train,dev,test]
    run_time = []
    t1 = time.time()
    for fil in in_files:
        data = json.load(open(fil))
        for question in data:
            t2 = time.time()
            QuestionId = question["Id"]
            QuestionText = question["Question"]
            path = cfg["ques_path"] + "ques_" + str(QuestionId)
            get_spo(path)
            print(str(QuestionId) + ": " + QuestionText + " SPOs generated.")
            t3 = time.time()
            run_time.append(t3-t2)

    t4 = time.time()
    total_t = t4 - t1
    print("\n\ntotal_t-> ", str(total_t))
    print("\n\naverage extract fact time-> ", str(sum(run_time) / len(run_time)))