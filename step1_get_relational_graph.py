"""Script to get relational graph from answer graph (temporal enhanced completed GSTs).
"""
import os
import json
import pickle
import globals
import networkx as nx
from tqdm import tqdm
import time

def get_answer(question):
    """extract unique answers from dataset."""
    GT = set()
    # print (answers)
    for answer in question["Answer"]:
        if "AnswerType" in answer:
            if answer["AnswerType"] == "Entity":
                GT.add((answer["WikidataQid"], answer['WikidataLabel'].lower()))
            else:
                GT.add((answer["AnswerArgument"].replace("T00:00:00Z",""), answer["AnswerArgument"].replace("T00:00:00Z","")))
    return list(GT)

def get_triple_from_SPO(facts, enhance_facts, pro_info):
    #get spo triples
    tuples = []
    date_in_f = dict()
    state = dict()
    t_rels = set()
    t_ens = set()
    ens =set()
    rels = set()
    tempstate = dict()
    for line in enhance_facts:
        triple = line.strip().split('||')
        statement_id = triple[0].replace("-ps:", "").replace("-pq:", "").lower()
        if statement_id not in state:
            state[statement_id] = {'ps':[],'pq':[]}
        if statement_id not in tempstate:
            tempstate[statement_id] = {'ps_spo':[], 'pq_spo':set(), 'date':set()}
        sub_id = triple[1]
        obj_id = triple[5]
        sub_name = triple[2]
        obj_name = triple[6]
        rel = rel_name = triple[4]
        if 'corner#' in sub_id: sub_id = sub_id.replace('corner#', '').split('#')[0]
        if 'corner#' in obj_id: obj_id = obj_id.replace('corner#', '').split('#')[0]
        if 'T00:00:00Z' in sub_name:sub_name = sub_name.replace('T00:00:00Z', '')
        if 'T00:00:00Z' in obj_name:obj_name = obj_name.replace('T00:00:00Z', '')

        if rel in pro_info:
            rel_name = pro_info[rel]['label']
            if pro_info[rel]["type"] == "http://wikiba.se/ontology#Time":
                obj_id = obj_id.replace('T00:00:00Z', '')
                tempstate[statement_id]['date'].add((rel_name, obj_id))

            elif "T00:00:00Z" in obj_id:
                obj_id = obj_id.replace('T00:00:00Z', '')
                tempstate[statement_id]['date'].add((rel_name, obj_id))
        if "-ps:" in triple[0]:
            ps_spo = (sub_id, sub_name, rel_name, obj_id, obj_name)
            tempstate[statement_id]['ps_spo'].append(ps_spo)
            state[statement_id]['ps'].append(line)
        if "-pq:" in triple[0] and line not in state[statement_id]['pq']:
            state[statement_id]['pq'].append(line)
            pq_spo = (sub_id, sub_name, rel_name, obj_id, obj_name)
            tempstate[statement_id]['pq_spo'].add(pq_spo)

    tkgfacts = get_qtkg(tempstate)

    for line in facts:
        triple = line.strip().split('||')
        statement_id = triple[0].replace("-ps:", "").replace("-pq:", "").lower()
        if statement_id not in state:
            state[statement_id] = dict()
            state[statement_id]['ps'] = []
            state[statement_id]['pq'] = []
        if "-ps:" in triple[0]:
            state[statement_id]['ps'].append(line)
        if "-pq:" in triple[0] and line not in state[statement_id]['pq']:
            state[statement_id]['pq'].append(line)

    for statement_id in state:
        if statement_id not in date_in_f:
            date_in_f[statement_id] = dict()
            date_in_f[statement_id]['tuple'] = []
            date_in_f[statement_id]['date'] = []
        if len(state[statement_id]['ps']) > 0 and len(state[statement_id]['pq']) == 0:
            ps_lines = state[statement_id]['ps']
            for line in ps_lines:
                triple = line.strip().split("||")
                sub = triple[1]
                sub_name = triple[2]
                if "corner#" in sub: sub = sub.split("#")[1]
                obj = triple[5]
                obj_name = triple[6].replace("T00:00:00Z", "")
                if "corner#" in obj: obj = obj.split("#")[1]
                rel = triple[3]
                rel_name = rel
                if rel in pro_info:
                    rel_name = pro_info[rel]['label']
                    if pro_info[rel]["type"] == "http://wikiba.se/ontology#Time" and not obj.startswith("_:"):
                        t_rels.add(rel_name)
                        obj = obj.replace("T00:00:00Z", "")
                        date_in_f[statement_id]['date'].append({"date_id":obj, "date_rel":rel_name})
                        t_ens.add(obj)
                if "T00:00:00Z" in obj and not obj.startswith("_:"):
                    obj = obj.replace("T00:00:00Z", "")
                    date_in_f[statement_id]['date'].append({"date_id":obj, "date_rel":rel_name})
                    t_ens.add(obj)
                    t_rels.add(rel_name)

                date_in_f[statement_id]['tuple'].append([
                    {"kb_id": sub, "text": sub_name},
                    {"rel_id": rel, "text": rel_name},
                    {"kb_id": obj, "text": obj_name}])
                ens.add(sub)
                ens.add(obj)
                rels.add(rel_name)

        if len(state[statement_id]['ps']) > 0 and len(state[statement_id]['pq']) > 0:
            ps_lines = state[statement_id]['ps']
            pq_lines = state[statement_id]['pq']
            for line in ps_lines:
                triple = line.strip().split("||")
                sub = triple[1]
                sub_name = triple[2]
                if "corner#" in sub: sub = sub.split("#")[1]
                obj = triple[5]
                obj_name = triple[6].replace("T00:00:00Z", "")
                if "corner#" in obj: obj = obj.split("#")[1]
                rel = triple[3]
                rel_name = rel
                if rel in pro_info:
                    rel_name = pro_info[rel]['label']
                    if pro_info[rel]["type"] == "http://wikiba.se/ontology#Time" and not obj.startswith("_:"):
                        t_rels.add(rel_name)
                        obj = obj.replace("T00:00:00Z", "")
                        date_in_f[statement_id]['date'].append({"date_id":obj, "date_rel":rel_name})
                        t_ens.add(obj)

                if "T00:00:00Z" in obj and not obj.startswith("_:"):
                    obj = obj.replace("T00:00:00Z","")
                    date_in_f[statement_id]['date'].append({"date_id":obj, "date_rel":rel_name})
                    t_ens.add(obj)
                    t_rels.add(rel_name)

                date_in_f[statement_id]['tuple'].append([
                    {"kb_id": sub, "text": sub_name},
                    {"rel_id": rel, "text": rel_name},
                    {"kb_id": statement_id, "text": statement_id}])
                date_in_f[statement_id]['tuple'].append([
                    {"kb_id": statement_id, "text": statement_id},
                    {"rel_id": rel, "text": rel_name},
                    {"kb_id": obj, "text": obj_name}])
                ens.add(sub)
                ens.add(statement_id)
                ens.add(obj)
                rels.add(rel_name)

            for line in pq_lines:
                triple = line.strip().split("||")
                obj = triple[5]
                obj_name = triple[6].replace("T00:00:00Z", "")
                if "corner#" in obj: obj = obj.split("#")[1]
                rel_ps = triple[3]
                rel_pq = triple[4]
                rel_pq_name = rel_pq
                if rel_pq in pro_info:
                    rel_pq_name = pro_info[rel_pq]['label']
                    if pro_info[rel_pq]["type"] == "http://wikiba.se/ontology#Time" and not obj.startswith("_:"):
                        t_rels.add(rel_pq_name)
                        obj = obj.replace("T00:00:00Z", "")
                        date_in_f[statement_id]['date'].append({"date_id":obj, "date_rel":rel_pq_name})
                        t_ens.add(obj)
                if "T00:00:00Z" in obj:
                    obj = obj.replace("T00:00:00Z","")
                    t_ens.add(obj)
                    date_in_f[statement_id]['date'].append({"date_id":obj, "date_rel":rel_pq_name})
                    t_rels.add(rel_pq_name)
                date_in_f[statement_id]['tuple'].append([
                    {"kb_id": statement_id, "text": statement_id},
                    {"rel_id": rel_pq, "text": rel_pq_name},
                    {"kb_id": obj, "text": obj_name}])
                ens.add(statement_id)
                ens.add(obj)
                rels.add(rel_pq_name)

    for statement_id in date_in_f:
        tuples.append((date_in_f[statement_id]['tuple'], date_in_f[statement_id]['date']))
    return tuples, t_rels, t_ens, rels, ens, tkgfacts

def _read_seeds(tagme, elq):
    """Return map from question ids to seed entities."""
    seeds = set()

    with open(tagme) as f:
        for line in f:
            entity, score, text = line.strip().split('\t')
            seeds.add(entity)

    with open(elq) as f:
        for line in f:
            entity, score, text = line.strip().split('\t')
            seeds.add(entity)

    return list(seeds)

def _read_corners(cornerstone_file):
    """Return map from question ids to cornerstone entities."""
    corners = []
    if not os.path.exists(cornerstone_file):
        print("cornerstone file not found!")
    else:
        data = pickle.load(open(cornerstone_file, 'rb'))

        for key in data:
            if key.strip().split("::")[1] == "Entity":
                corners.append(key.strip().split("::")[2])
    return corners

def _readable_entities(entities, weight):
    readable_entities = []
    for ent in entities:
        sc = 0.0
        if ent in weight:
            sc = weight[ent]
        readable_entities.append(
                {"text": ent, "kb_id": ent,
                    "score": sc})

    return readable_entities

def _get_answer_coverage(GT, entities):
    found, total = 0., 0
    print("\n\nanswer:")
    for answer in GT:
        if answer[0] in entities:
            found += 1.
        elif answer[0] + "T00:00:00Z" in [item.lower() for item in entities]:
            found += 1.
        total += 1
    return found / total

def _read_weight(QKG_file):
    weight_map = dict()
    if not os.path.exists(QKG_file):
        print("QKG_file file not found!")
    else:
        QKG = nx.read_gpickle(QKG_file)
        for node in QKG.nodes:
            if node.strip().split("::")[1] == "Entity":
                weight_map[node.split("::")[2]] = QKG.node[node]["weight"]
    return weight_map

def get_qtkg(tempstate):
    tkgfacts = set()
    for statement_id in tempstate.keys():
        for item in tempstate[statement_id]['date']:
            if item[1].startswith('-'):
                int_date = int('-' + item[1].strip().replace('-', ''))
            else:
                int_date = int(item[1].strip().replace('-', ''))
            ps_spo = tempstate[statement_id]['ps_spo'][0]
            if (ps_spo[2], ps_spo[3]) in tempstate[statement_id]['date']:
                if ps_spo[3].startswith('-'):
                    int_date = int('-' + ps_spo[3].strip().replace('-', ''))
                else:
                    int_date = int(ps_spo[3].strip().replace('-', ''))
                tuple = (ps_spo[0], ps_spo[2], ps_spo[3], ps_spo[2], ps_spo[3], int_date)
                tkgfacts.add(tuple)
            else:
                tuple = (ps_spo[0], ps_spo[2], ps_spo[3], item[0], item[1], int_date)
                tkgfacts.add(tuple)
            for pq_spo in tempstate[statement_id]['pq_spo']:
                if (pq_spo[2], pq_spo[3]) not in tempstate[statement_id]['date']:
                    tuple = (pq_spo[0], pq_spo[2], pq_spo[3], item[0], item[1], int_date)
                    tkgfacts.add(tuple)
    tkgfacts = sorted(list(tkgfacts),key=lambda x:x[5])

    return tkgfacts

def _read_weight_from_QKG(graph_file):
    G = nx.read_gpickle(graph_file)
    return {G.nodes[n]['id']: G.nodes[n]['weight'] for n in G}

def get_subgraph(dataset, pro_info, cfg, topf = 25, topg = 25, topt = 25):
    t1 = time.time()
    gcn_file_path = cfg['gcn_file_path']
    os.makedirs(gcn_file_path, exist_ok=True)
    analysis_file = gcn_file_path + '/' + dataset + "_relation_subg_analysis.txt"
    fa = open(analysis_file, "w", encoding='utf-8')
    subgraph_file = gcn_file_path + "/" + dataset +  "_subgraph.json"
    fo = open(subgraph_file, "wb")

    test = cfg["data_path"] + cfg["test_data"]
    dev = cfg["data_path"] + cfg["dev_data"]
    train = cfg["data_path"] + cfg["train_data"]

    if dataset == 'test':
        in_file = test
    elif dataset == 'dev':
        in_file = dev
    elif dataset == 'train':
        in_file = train

    questions = json.load(open(in_file))
    graphspo_file = cfg["data_path"] + dataset + '_' + str(topf) + '_' + str(topg) + ".json"
    ques_temprank = {}
    ques_enhance = {}
    if topt > 0 :
        tempspo_file = cfg["data_path"] + dataset + '_' + str(topf) + '_' + str(
                topg) + '_temp.json'
        tempspo_rank_file = cfg["data_path"] + dataset + '_' + str(topf) + '_' + str(topg) + '_temp_rank'
        tempspo_rank = pickle.load(open(tempspo_rank_file, 'rb'))
        for item in tempspo_rank:
            hit_sta = []
            for rank in item['rank']:
                k = rank.split('\t')[0]
                if int(k) <= topt:
                    hit_sta.append(rank.split('\t')[1])
            ques_temprank[item['id']] = hit_sta

        with open(tempspo_file, encoding = 'utf-8') as f_in:
            for line in tqdm(f_in):
                enhance_facts = []
                line = json.loads(line)
                hit_sta = ques_temprank[line['id']]
                for temp_line in line['tempfact']:
                    triple = temp_line.strip().split('||')
                    if len(triple) < 7 or len(triple) > 7: continue
                    statement_id = triple[0].replace("-ps:", "").replace("-pq:", "").lower()
                    if statement_id in hit_sta:
                        enhance_facts.append(temp_line)
                ques_enhance[line['id']] = enhance_facts

    ques_spos = {}
    with open(graphspo_file, encoding='utf-8') as f_in:
        for line in tqdm(f_in):
            line = json.loads(line)
            ques_spos[line['id']] = line['subgraph']

    seed_map = {}
    corner_map = {}
    answer_recall, total = 0.0, 0
    bad_questions = []
    good_questions = []
    ok_questions = []
    num_empty_tuples = 0
    SPONOTFOUND = 0
    total_entities = 0
    connect_subgraphs = []
    for question in questions:
        QuestionId = question["Id"]
        QuestionText = question["Question"]
        print("\n\nQuestion Id-> ", QuestionId)
        print("Question -> ", QuestionText)
        path = cfg['ques_path'] + 'ques_' + str(QuestionId)
        graph_file = path + '/QKG_' + str(topf) + '.gpickle'
        corner_file = path + '/cornerstone_' + str(topf)
        tagme_file = path + '/wiki_ids_tagme.txt'
        elq_file = path + '/wiki_ids_elq.txt'
        lines = ques_spos[QuestionId]
        enhance_facts = []
        if topt > 0:
            enhance_facts = ques_enhance[QuestionId]
        corner_map[QuestionId] = _read_corners(corner_file)
        seed_map[QuestionId] = _read_seeds(tagme_file, elq_file)
        tuples, t_rels, t_ents, rels, ents, tkgfacts = get_triple_from_SPO(lines, enhance_facts, pro_info)

        seed_entities = []
        corner_entities = []
        ans_entities = []
        GT = get_answer(question)
        for ee in corner_map[QuestionId]:
            if ee in ents:
                corner_entities.append(ee)
        for ee in seed_map[QuestionId]:
            if ee in ents:
                seed_entities.append(ee)

        entities_weight = _read_weight_from_QKG(graph_file)

        if corner_entities:
            for answer in GT:
                if answer[0] in ents:
                    ans_entities.append(answer[0])

        if not question["Answer"] or len(ans_entities) == 0:
            curr_recall = 0.
        else:
            curr_recall = _get_answer_coverage(GT, ents)

        if curr_recall == 0.:
            bad_questions.append(QuestionId)

        if curr_recall < 1. and curr_recall > 0.:
            ok_questions.append(QuestionId)

        if curr_recall == 1.:
            good_questions.append(QuestionId)

        answer_recall += curr_recall

        total += 1

        answers = [{"kb_id": answer[0], "text": answer[1]} for answer in GT]

        ques_entities = _readable_entities(ents, entities_weight)
        total_entities += len(ques_entities)
        data = {
                "question": QuestionText,
                "question_seed_entities": seed_map[QuestionId],
                "seed_entities": _readable_entities(seed_entities, entities_weight),
                "corner_entities": _readable_entities(corner_entities, entities_weight),
                "answers": answers,
                "id": QuestionId,
                "subgraph": {
                    "entities": ques_entities,
                    "tuples": tuples
                },
                "signal": question["Temporal signal"],
                "type": question["Type"],
                "tkg": tkgfacts,
                "tempentities": list(t_ents),
                "temprelations": list(t_rels)
                }

        if dataset == 'train':
            if data['id'] in good_questions or data['id'] in ok_questions:
                fo.write(json.dumps(data).encode("utf-8"))
                fo.write("\n".encode("utf-8"))
        else:
            fo.write(json.dumps(data).encode("utf-8"))
            fo.write("\n".encode("utf-8"))

        result = "{0}|{1}|{2}|{3}|{4}|{5}".format(
                str(QuestionId),
                QuestionText,
                str(len(ents)),
                str(len(corner_entities)),
                str(len(seed_entities)),
                str(curr_recall)
                )
        fa.write(result)
        fa.write('\n')

    t2 = time.time()

    fa.write("total number of questions: " + str(total) + '\n')
    fa.write("total number of entities: " + str(total_entities)  + '\n')
    fa.write("average number of entities: " + str(total_entities * 1.0 / (total - SPONOTFOUND))  + '\n')
    fa.write("questions with empty subgraphs: " + str(SPONOTFOUND)  + '\n')
    fa.write("Good questions =  " + str(len(good_questions) * 1.0 / total)  + '\n')
    fa.write("Number of good questions =  " + str(len(good_questions))  + '\n')
    fa.write("Number of ok questions =  " + str(len(ok_questions))  + '\n')
    fa.write("Answer recall =  " + str(answer_recall / total)  + '\n')
    fa.write("total time: " + str(t2-t1) + '\n')
    fa.write("average time: " + str((t2-t1) * 1.0 / total) + '\n')
    fo.close()
    fa.close()

    print("total number of questions.")
    print(str(total))
    print("total number of entities.")
    print(str(total_entities))
    print("questions with empty subgraphs.")
    print(str(num_empty_tuples))
    print("Good questions = ")
    print(str(len(good_questions) * 1.0 / total))
    print("Number of good questions = ")
    print(str(len(good_questions)))
    print("Number of OK questions = ")
    print(str(len(ok_questions)))
    print("Answer recall = ")
    print(str(answer_recall / total))
    print("SPO files not found = ")
    print(str(SPONOTFOUND))
    print("average number of entities.")
    print(str(total_entities * 1.0 / (total - SPONOTFOUND)))

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--dataset', type=str, default='dev')
    cfg = globals.get_config(globals.config_file)
    pro_info = globals.ReadProperty.init_from_config().property
    args = argparser.parse_args()
    dataset = args.dataset
    topf = topg = topt = 25

    get_subgraph(dataset, pro_info, cfg, topf, topg, topt)


