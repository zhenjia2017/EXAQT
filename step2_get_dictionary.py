"""Script to get dictionaries for training GCN model.
"""

import json
import pickle
import globals
from tqdm import tqdm

def write_to_file(word, fout):
    fp_entity = open(fout, 'w', encoding='utf-8')
    print (len(word))
    for item in word:
        fp_entity.write(item)
        fp_entity.write('\n')
    fp_entity.close()

def load_dict(filename):
    word2id = dict()
    with open(filename) as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    print(len(word2id))
    return word2id

def replace_symbols(s):
    #s = s.replace('<entity>', ' ')
    s = s.replace('(', ' ')
    s = s.replace(')', ' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    s = s.replace('{', ' ')
    s = s.replace('}', ' ')
    s = s.replace('|', ' ')
    s = s.replace('"', ' ')
    s = s.replace(':', ' ')
    s = s.replace('<', ' ')
    s = s.replace('>', ' ')
    s = s.replace('\'s',' ')
    s = s.replace('\'', ' ')
    s = s.replace('\n',' ')
    s = s.strip(',')
    s = s.strip('.')
    s = s.strip('#')
    s = s.strip('-')
    s = s.strip('\'')
    s = s.strip(';')
    s = s.strip('\"')
    s = s.strip('/')
    s = s.rstrip('?')
    s = s.rstrip('!')
    s = s.strip()
    return s

def get_dictionary(gcn_file_path, subgraph_files):
    fp_entity = gcn_file_path + '/entities.txt'
    fp_entity_name = gcn_file_path + '/entity_name_map.pkl'
    fp_date = gcn_file_path + '/dates.txt'
    fp_relation = gcn_file_path + '/'+ '/relations.txt'
    fp_trelation = gcn_file_path + '/'+ '/trelations.txt'
    fp_word = gcn_file_path + '/words.txt'
    fp_statement = gcn_file_path + '/statements.txt'
    fp_category = gcn_file_path + '/categories.txt'
    fp_signal = gcn_file_path + '/signals.txt'
    fp_tempfact_pkl = gcn_file_path + '/tempfacts.pkl'
    fp_tempfact = gcn_file_path + '/tempfacts.txt'

    entities = set()
    tempentities = list()
    relations = list()
    trelations = list()
    datas = list()
    categories = list()
    signals = list()
    entity_name = {}
    statements = set()
    words = list()
    tkgfacts = list()

    for subgraph_file in subgraph_files:
        with open(subgraph_file, "r") as f:
            for line in tqdm(f):
                line = json.loads(line.strip())
                datas.append(line)
    print ("Total number of questions: ", str(len(datas)))
    for data in datas:
        words += [replace_symbols(item) for item in data['question'].strip().split()]
        categories += data["type"]
        signals += data["signal"]
        tempentities += data["tempentities"]
        trelations += data["temprelations"]
        tkgfacts += data["tkg"]
        for entity in data["subgraph"]["entities"]:
            entities.add(entity["kb_id"].strip('"'))
        for entity in data["seed_entities"]:
            entities.add(entity["kb_id"])
        for entity in data["corner_entities"]:
            entities.add(entity["kb_id"])
        for tem in data["tempentities"]:
            entities.add(tem)
        for fact in data["tkg"]:
            entities.add(fact[0])
            entities.add(fact[2])
            entities.add(fact[4])
            trelations.append(fact[3])
            relations.append(fact[1])
            relations.append(fact[3])
            tempentities.append(fact[4])

        for tuples, dates in data['subgraph']['tuples']:
            for sbj, rel, obj in tuples:
                sbj_id  = sbj["kb_id"]
                sbj_txt = sbj["text"]
                rel_txt = rel["text"]
                obj_id = obj["kb_id"]
                obj_txt = obj["text"]
                relations.append(rel_txt)
                entities.add(sbj_id)
                entities.add(obj_id)
                if '-' in sbj_id and len(sbj_id.split('-')[1]) == 32:
                    statements.add(sbj_id)
                else:
                    if '-' in obj_id and len(obj_id.split('-')[1]) == 32:
                        statements.add(obj_id)
                if sbj_id not in entity_name:
                    if sbj["text"] == '':
                        entity_name[sbj_id] = sbj_id.replace("T00:00:00Z","")
                    else:
                        entity_name[sbj_id] = sbj_txt
                if obj_id not in entity_name:
                    if obj["text"] == '':
                        entity_name[obj_id] = obj_id.replace("T00:00:00Z", "")
                    else:
                        entity_name[obj_id] = obj_txt

    write_to_file(list(set(entities)), fp_entity)
    write_to_file(list(set(tempentities)), fp_date)
    write_to_file(list(set(relations)), fp_relation)
    write_to_file(list(set(trelations)), fp_trelation)
    write_to_file(list(set(words)), fp_word)
    write_to_file(list(set(statements)), fp_statement)
    write_to_file(list(set(categories)), fp_category)
    write_to_file(list(set(signals)), fp_signal)
    pickle.dump(entity_name, open(fp_entity_name, 'wb'))
    print (len(tkgfacts))

    tuple_tkgfacts = set()
    for item in tkgfacts:
        tuple_tkgfacts.add(tuple(item))
    sorted_tkgfacts = sorted(list(set(tuple_tkgfacts)), key=lambda x: x[5])
    pickle.dump(sorted_tkgfacts, open(fp_tempfact_pkl, 'wb'))

    entity2id = load_dict(fp_entity)
    relation2id = load_dict(fp_relation)
    date2id = load_dict(fp_date)
    tkgfacts_write_list = []

    for item in sorted_tkgfacts:
        tkgfacts_write_list.append(item[0] + " | " + item[1] + " | " +  item[2]+ " | " + item[3] + " | " +  item[4])
    write_to_file(tkgfacts_write_list, fp_tempfact)
    tkgfact2id = load_dict(fp_tempfact)

    print("#words = %s" % str(len(set(words))))
    print("#entities = %s" % str(len(entities)))
    print("#temporal relations = %s" % str(len(set(trelations))))
    print("#relations = %s" % str(len(set(relations))))
    print("#dates = %s" % str(len(set(tempentities))))
    print("#entity_names = %s" % str(len(entity_name.keys())))
    print("#statements = %s" % str(len(statements)))
    print("#tkgfacts = %s" % str(len(tkgfact2id)))

if __name__ == "__main__":

    cfg = globals.get_config(globals.config_file)
    pro_info = globals.ReadProperty.init_from_config().property
    topf = topg = topt = 25
    gcn_file_path = cfg['gcn_file_path']
    train_subgraph = gcn_file_path + "/train_subgraph.json"
    dev_subgraph = gcn_file_path + "/dev_subgraph.json"
    test_subgraph = gcn_file_path + "/test_subgraph.json"
    subgraphs = [train_subgraph, dev_subgraph, test_subgraph]
    get_dictionary(gcn_file_path, subgraphs)

