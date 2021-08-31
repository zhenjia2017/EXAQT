"""Script to get top-g GSTs compact subgraph for questions from top-f scored facts .
   Default: top-f = top-g = 25
"""
import copy
import os
import json
import networkx as nx
import globals
import time
import signal
import pickle
from get_GST import call_main_rawGST

from nltk.stem import PorterStemmer
PS = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
LC = LancasterStemmer()
from nltk.stem import SnowballStemmer
SB = SnowballStemmer("english")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
MAX_TIME = 300

#remove symbols from relation (predicate, property) labels
def replace_symbols_in_relation(s):
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
    s = s.replace('\'s', ' ')
    s = s.replace('\'', ' ')
    s = s.replace('\n', ' ')
    s = s.replace('/', ' ')
    s = s.replace('\\', ' ')
    s = s.replace(',', ' ')
    s = s.replace('-', ' ')
    s = s.replace('.', ' ')
    s = s.replace(',', ' ')
    s = s.replace('\"', ' ')
    s = s.strip()
    return s

#remove symbols from question texts
def replace_symbols_in_question(s):
    s = s.replace('(', ' ')
    s = s.replace(')', ' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    s = s.replace('{', ' ')
    s = s.replace('}', ' ')
    s = s.replace('|', ' ')
    s = s.replace('"', ' ')
    s = s.replace('<', ' ')
    s = s.replace('>', ' ')
    s = s.replace('\'',' ')
    s = s.replace('\n',' ')
    s = s.replace('?', ' ')
    s = s.strip(',')
    s = s.rstrip('.')
    s = s.strip()
    return s

#get label, alias and type of relations
def property_type_label_alias(pre, pro_info):
    type = ""
    label = ""
    alias = []
    if pre in pro_info.keys():
        type = pro_info[pre]['type']
        label = pro_info[pre]['label']
        altLabel = pro_info[pre]['altLabel']
        if altLabel.find(", ") >= 0:
            alias = altLabel.split(", ")
        elif len(altLabel) > 0:
            alias = [altLabel]
    return type, label, alias

#create quasi question graph from spo triples
def build_graph_from_triple_edges(unique_SPO_dict, proinfo, sta_score):
    G = nx.DiGraph()
    pred_count = {}
    for (n1, n2, n3) in unique_SPO_dict:
        n2_id = n2.split("#")[0]
        n2_sta = n2.split("#")[1]
        sta_weight = sta_score[n2_sta]
        #get type, label, and alias of relations n2_id
        n22t, n22l, n22a = property_type_label_alias(n2_id, proinfo)
        n1_name = unique_SPO_dict[(n1, n2, n3)]['name_n1']
        n3_name = unique_SPO_dict[(n1, n2, n3)]['name_n3'].replace('T00:00:00Z','')
        n11 = n1_name + "::Entity::" + n1
        if n22l not in pred_count:
            pred_count[n22l] = 1
        else:
            pred_count[n22l] = pred_count[n22l] + 1
        n22 = n22l + "::Predicate::" + n2_sta + "::" + str (pred_count[n22l])
        n33 = n3_name + "::Entity::" + n3
        if n11 not in G.nodes():
            n1_alias = []
            G.add_node(n11, id=n1, alias=n1_alias, weight=unique_SPO_dict[(n1, n2, n3)]['score_n1'],
                       matched=unique_SPO_dict[(n1, n2, n3)]['matched_n1'])

        if n22 not in G.nodes():
            G.add_node(n22, id=n2_id, alias=n22a, type=n22t, weight=0.0, matched='')

        if n33 not in G.nodes():
            n3_alias = []
            G.add_node(n33, id=n3, alias=n3_alias, weight=unique_SPO_dict[(n1, n2, n3)]['score_n3'],
                       matched=unique_SPO_dict[(n1, n2, n3)]['matched_n3'])

        G.add_edge(n11, n22, weight=sta_weight, wlist=[sta_weight], etype='Triple')
        G.add_edge(n22, n33, weight=sta_weight, wlist=[sta_weight], etype='Triple')

        #add qualifier nodes edges
        if 'qualifier' in unique_SPO_dict[(n1, n2, n3)]:
            for qualct in range (0, len (unique_SPO_dict[(n1, n2, n3)]['qualifier'])):
                qual = unique_SPO_dict[(n1, n2, n3)]['qualifier'][qualct]
                qn2_id = qual[0].split("#")[0]
                qn3_id = qual[1]
                qn3_name = qual[2].replace('T00:00:00Z','')
                qn22t, qn22l, qn22a = property_type_label_alias (qn2_id, proinfo)
                if qn22l not in pred_count:
                    pred_count[qn22l] = 1
                else:
                    pred_count[qn22l] = pred_count[qn22l] + 1
                qn22 = qn22l + "::Predicate::" + qual[0].split("#")[1] + "::" + str(pred_count[qn22l])

                qn33 = qn3_name + "::Entity::" + qn3_id

                if qn22 not in G.nodes ():
                    G.add_node (qn22, id=qn2_id, alias=qn22a, type=qn22t, weight=0.0, matched='')

                if qn33 not in G.nodes ():
                    qn33_alias = []
                    G.add_node(qn33, id=qn3_id ,alias=qn33_alias, weight=unique_SPO_dict[(n1, n2, n3)]['score_qn3'][qualct],
                                matched=unique_SPO_dict[(n1, n2, n3)]['matched_qn3'][qualct])

                G.add_edge (n22, qn22, weight=sta_weight, wlist=[sta_weight], etype='Triple')
                G.add_edge (qn22, qn33, weight=sta_weight, wlist=[sta_weight], etype='Triple')

    return G

#convert quasi question graph from directed to undirected graph
def directed_to_undirected(G1):
    G = nx.Graph()
    for n in G1:
        if 'id' in G1.nodes[n]:
            G.add_node(n, id = G1.nodes[n]['id'], alias = G1.nodes[n]['alias'], weight=G1.nodes[n]['weight'], matched=G1.nodes[n]['matched'])
        else:
            print ("\n\nThis node has no id")
            print (n)
            print (G1.nodes[n])
            break
    done = set()
    elist = []
    for (n1, n2) in G1.edges():
        if (n1, n2) not in done:
            done.add((n1, n2))
            done.add((n2, n1))
            data = G1.get_edge_data(n1, n2)

            d = data['weight']
            wlist1 = copy.deepcopy(data['wlist'])
            etype1 = data['etype']

            if (n2, n1) in G1.edges():
                data1 = G1.get_edge_data(n2, n1)
                if data1['etype'] == 'Triple':
                    if data1['weight'] > d:  # Keeping maximum weight edge
                        d = data1['weight']

                    for w in data1['wlist']:

                        wlist1.append(w)


            for i in range(0, len(wlist1)):
                if wlist1[i] > 1.0 and wlist1[i] <= 1.0001:
                    wlist1[i] = 1.0
            if d > 1.0 and d <= 1.0001:
                d = 1.0
            G.add_edge(n1, n2, weight=d, wlist=wlist1, etype=etype1)

    flag = 0
    elist = sorted(elist, key=lambda x: x[2], reverse=True)

    for (n1, n2) in G.edges():
        data = G.get_edge_data(n1, n2)
        d = data['weight']
        wlist1 = data['wlist']

        if d > 1:
            flag += 1
        for ww in wlist1:
            if ww > 1:
                flag += 1
    return G

#get cornerstones of property
def pred_match(labels, q_ent):
    matched = ''
    lab_stem = set()
    for label in labels:
        label = replace_symbols_in_relation(label)
        label_li = set(label.split())
        for lab in label_li:
            if lab not in stop_words:
                lab_stem |= set([lab, PS.stem(lab), LC.stem(lab), SB.stem(lab)])

    for terms in q_ent:
        if terms in stop_words: continue
        ter_stem = set()
        qterms = set(terms.split())
        for ter in qterms:
            if ter not in stop_words:
                ter_stem |= set([ter, PS.stem(ter), LC.stem(ter), SB.stem(ter)])

        if len(lab_stem.intersection(ter_stem)) > 0:
            matched = terms
            return matched
    return matched

def get_spo_for_build_graph(spo_lines,q_ent,hit_sta,sta_score,ent_sta):
    corner_ent = {}
    unique_SPO_dict = {}
    spo_fact = {}
    qkgspo = []
    for line in spo_lines:
        triple = line.strip().split('||')
        if len(triple) < 7 or len(triple) > 7: continue
        sta_id = triple[0]
        statement_id = sta_id.replace("-ps:", "").replace("-pq:", "").lower()
        if statement_id in hit_sta: qkgspo.append(line)
        if statement_id not in spo_fact:
            spo_fact[statement_id] = {}
        n1_id = triple[1]
        n1_name = triple[2]
        n2 = triple[4]
        n3_id = triple[5]
        n3_name = triple[6]

        if n1_id not in ent_sta:
            ent_sta[n1_id] = []
        ent_sta[n1_id].append(statement_id)

        if n3_id not in ent_sta: ent_sta[n3_id] = []
        ent_sta[n3_id].append(statement_id)

        if sta_id.endswith("-ps:"):
            spo_fact[statement_id]["ps"] = (n1_id, n1_name, n2, n3_id, n3_name)

        if sta_id.endswith("-pq:"):
            if 'pq' not in spo_fact[statement_id]:
                spo_fact[statement_id]['pq'] = []
            spo_fact[statement_id]['pq'].append((n1_id, n1_name, n2, n3_id, n3_name))

    for sta in spo_fact:
        if 'ps' not in spo_fact[sta]: continue
        if sta not in hit_sta: continue
        ps = spo_fact[sta]['ps']
        n1_id = ps[0]
        n1_name = ps[1]
        n2 = ps[2] + "#" + sta
        n3_id = ps[3]
        n3_name = ps[4]
        score_n1 = 0.0
        score_n3 = 0.0
        for score in [sta_score[item] for item in ent_sta[n1_id]]:
            score_n1 += score
        for score in [sta_score[item] for item in ent_sta[n3_id]]:
            score_n3 += score
        matched_n1 = ''
        matched_n3 = ''
        if n1_id.startswith('corner#'):
            n1_id = n1_id.replace('corner#', '')
            n11 = n1_id.split('#')
            n1_id = n11[0]
            term = n11[2]

            term_words = set(term.split())
            for terms in q_ent:
                qterm = set(terms.split())
                if len(term_words.intersection(qterm)) > 0:
                    matched_n1 = terms
                    break

            if matched_n1 not in corner_ent:
                corner_ent[matched_n1] = set()
            corner_ent[matched_n1].add(n1_name.lower() + '::Entity::' + n1_id)

        if n3_id.startswith('corner#'):
            n3_id = n3_id.replace('corner#', '')
            n33 = n3_id.split('#')
            n3_id = n33[0]

            term = n33[2]
            term_words = set(term.split())

            for terms in q_ent:
                qterm = set(terms.split())
                if len(term_words.intersection(qterm)) > 0:
                    matched_n3 = terms
                    break

            if matched_n3 not in corner_ent:
                corner_ent[matched_n3] = set()
            corner_ent[matched_n3].add(n3_name.lower() + '::Entity::' + n3_id)

        if (n1_id, n2, n3_id) not in unique_SPO_dict:
            unique_SPO_dict[(n1_id, n2, n3_id)] = {}
        unique_SPO_dict[(n1_id, n2, n3_id)]['score_n1'] = score_n1
        unique_SPO_dict[(n1_id, n2, n3_id)]['score_n3'] = score_n3
        unique_SPO_dict[(n1_id, n2, n3_id)]['matched_n1'] = matched_n1
        unique_SPO_dict[(n1_id, n2, n3_id)]['matched_n3'] = matched_n3
        unique_SPO_dict[(n1_id, n2, n3_id)]['name_n1'] = n1_name
        unique_SPO_dict[(n1_id, n2, n3_id)]['name_n3'] = n3_name

        if 'pq' in spo_fact[sta]:
            pqstat = spo_fact[sta]['pq']
            for pq in pqstat:
                qn2 = pq[2] + "#" + sta
                qn3_id = pq[3]
                qn3_name = pq[4]
                score_qn3 = 0.0
                for score in [sta_score[item] for item in ent_sta[qn3_id]]:
                    score_qn3 += score
                matched_qn3 = ''

                if qn3_id.startswith('corner#'):
                    qn3_id = qn3_id.replace('corner#', '')
                    qn33 = qn3_id.split('#')
                    qn3_id = qn33[0]
                    term = qn33[2]
                    term_words = set(term.split())

                    for terms in q_ent:
                        qterm = set(terms.split())
                        if len(term_words.intersection(qterm)) > 0:
                            matched_qn3 = terms
                            break

                    if matched_qn3 not in corner_ent:
                        corner_ent[matched_qn3] = set()
                    corner_ent[matched_qn3].add(qn3_name.lower() + '::Entity::' + qn3_id)

                if 'qualifier' not in unique_SPO_dict[(n1_id, n2, n3_id)]:
                    unique_SPO_dict[(n1_id, n2, n3_id)]['qualifier'] = []
                unique_SPO_dict[(n1_id, n2, n3_id)]['qualifier'].append((qn2, qn3_id, qn3_name))

                if 'score_qn3' not in unique_SPO_dict[(n1_id, n2, n3_id)]:
                    unique_SPO_dict[(n1_id, n2, n3_id)]['score_qn3'] = []
                unique_SPO_dict[(n1_id, n2, n3_id)]['score_qn3'].append(score_qn3)

                if 'matched_qn3' not in unique_SPO_dict[(n1_id, n2, n3_id)]:
                    unique_SPO_dict[(n1_id, n2, n3_id)]['matched_qn3'] = []
                unique_SPO_dict[(n1_id, n2, n3_id)]['matched_qn3'].append(matched_qn3)

    return unique_SPO_dict, corner_ent, spo_fact, qkgspo

def call_main_GRAPH(spo_file, spo_rank_file, rank, q_ent, graph_file, corner_file, pro_info, seeds_paths, add_spo_line, add_spo_score):
    spo_rank = pickle.load(open(spo_rank_file, 'rb'))
    hit_sta = []
    sta_score = {}
    for key, value in spo_rank.items():
        sta_id = value.split('\t')[0]
        score =  float(value.split('\t')[1])
        if int(key) <= rank:
            hit_sta.append(sta_id)
        sta_score[sta_id] = score

    remove_duplicated_line = []
    with open(spo_file, 'r', encoding='utf-8') as f11:
        for line in f11:
            if line not in remove_duplicated_line:
                remove_duplicated_line.append(line)
    #construct graph first time to check if the graph is connected
    ent_sta = {}
    unique_SPO_dict, corner_ent, spo_fact, qkgspo = get_spo_for_build_graph(remove_duplicated_line, q_ent, hit_sta, sta_score, ent_sta)

    #print("\nAdding SPO triple edges\n")

    G = build_graph_from_triple_edges(unique_SPO_dict, pro_info, sta_score)
    # change G from direct to undirect graph
    G = directed_to_undirected(G)

    del unique_SPO_dict
    seed_ids = set()
    for item in corner_ent:
        for e in corner_ent[item]:
            seed_ids.add(e.split("::")[2])

    number_of_components_original = nx.number_connected_components(G)
    number_of_components_connect = number_of_components_original
    if number_of_components_original > 1:
        S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        subg_group_seeds = []
        seed_pairs = []
        connect_sta = []
        for subg in S:
            subg_seed = []
            for node in subg:
                if node.split("::")[1] == "Entity":
                    qid = node.split("::")[2]
                    if qid in seed_ids:
                        subg_seed.append(qid)
            if len(subg_seed) > 0:
                subg_group_seeds.append(subg_seed)
            else:
                print("\n\nError !!! This subgraph has no seeds!!")

        if len(subg_group_seeds) > 1:
            for i in range(0, len(subg_group_seeds) - 1):
                for s1 in subg_group_seeds[i]:
                    for j in range(i + 1, len(subg_group_seeds)):
                        for s2 in subg_group_seeds[j]:
                            seed_pairs.append((s1, s2))

        for (s1, s2) in seed_pairs:
            if (s1 + '||' + s2) in seeds_paths:
                for item in seeds_paths[(s1 + '||' + s2)]:
                    for sta in item:
                        if len(sta) > 0 and sta not in connect_sta:
                            connect_sta.append(sta.lower())
            elif (s2 + '||' + s1) in seeds_paths:
                for item in seeds_paths[(s2 + '||' + s1)]:
                    for sta in item:
                        if len(sta) > 0 and sta not in connect_sta:
                            connect_sta.append(sta.lower())

        ent_seed_score_map = {}
        for n1_id in ent_sta:
            if n1_id.startswith('corner#'):
                n1_id = n1_id.replace('corner#', '')
                n11 = n1_id.split('#')
                n1_id = n11[0]
                n1_score = n11[1]
                term = n11[2]
                ent_seed_score_map[n1_id] = {'score':n1_score,'term':term}

        adding_line_count = 0
        adding_lines = []
        for line in add_spo_line:

            triple = line.strip().split('||')
            if len(triple) < 7 or len(triple) > 7: continue
            statement_id = triple[0].replace("-ps:", "").replace("-pq:", "").lower()
            if statement_id in connect_sta:
                n1_id = triple[1]
                n3_id = triple[5]
                if n1_id.startswith('corner#'):
                    n1_id = n1_id.replace('corner#', '')
                    n1_id = 'corner#'+n1_id+'#'+ent_seed_score_map[n1_id]['score']+'#'+ent_seed_score_map[n1_id]['term']
                    line = line.replace(triple[1],n1_id)
                if n3_id.startswith('corner#'):
                    n3_id = n3_id.replace('corner#', '')
                    n3_id = 'corner#'+n3_id+'#'+ent_seed_score_map[n3_id]['score']+'#'+ent_seed_score_map[n3_id]['term']
                    line = line.replace(triple[5],n3_id)

                if line not in remove_duplicated_line:
                    remove_duplicated_line.append(line)
                    adding_lines.append(line)
                    adding_line_count += 1
        print("\n\nadding lines: " + str(adding_line_count))

        if len(connect_sta) > 0:
            for sta in connect_sta:
                if sta not in sta_score:
                    if sta not in add_spo_score:
                        print(add_spo_score)
                    sta_score[sta] = float(add_spo_score[sta])
            unique_SPO_dict, corner_ent, spo_fact, qkgspo = get_spo_for_build_graph(remove_duplicated_line, q_ent, hit_sta + connect_sta, sta_score, ent_sta)
            G = build_graph_from_triple_edges(unique_SPO_dict, pro_info, sta_score)
            # change G from direct to undirect graph
            G = directed_to_undirected(G)
            del spo_fact
            del unique_SPO_dict
            seed_ids = set()
            for item in corner_ent:
                for e in corner_ent[item]:
                    seed_ids.add(e.split("::")[2])

            number_of_components_connect = nx.number_connected_components(G)
            print("\n\nnumber_of_components", number_of_components_original)
            print("\n\nnumber_of_components_connect", number_of_components_connect)


    for n1 in G:
        nn1 = n1.split('::')
        id = G.nodes[n1]['id']
        label = nn1[0]
        alias = G.nodes[n1]['alias'].copy()
        alias.append(label)
        if nn1[1] == 'Predicate':
            G.nodes[n1]['weight'] = sta_score[nn1[2]]
            G.nodes[n1]['matched'] = pred_match(alias, q_ent)

    print("\n\nGetting cornerstones \n\n")
    corner2 = {}
    for n in G:
        id = G.nodes[n]['id']
        if G.nodes[n]['matched'] != '':
            if G.nodes[n]['matched'] not in corner2:
                corner2[G.nodes[n]['matched']] = []
            corner2[G.nodes[n]['matched']].append(n)
        elif id in seed_ids:
            print (n)
            if G.nodes[n]['matched'] not in corner2:
                corner2[G.nodes[n]['matched']] = []
            corner2[G.nodes[n]['matched']].append(n)

    cornerstone = {}
    for v in corner2:
        for e in corner2[v]:
            cornerstone[e] = v
    try:
        f77 = open(corner_file, 'wb')
        pickle.dump(cornerstone, f77)
        f77.close()
    except:
        print("fail to save cornerstone into file\n\n")
    nx.write_gpickle(G, graph_file)

    return G, cornerstone, qkgspo, number_of_components_original, number_of_components_connect

def get_groundtruth(corpus):
    ques_ans = {}
    with open(corpus) as json_data:
        datalist = json.load(json_data)
    json_data.close()
    for item in datalist:
        GT = []
        ques_id = str(item['Id'])
        text = item['Question']
        answers = item['Answer']
        for ans in answers:
            if 'WikidataQid' in ans:
                Qid = ans['WikidataQid'].lower()
                GT.append(Qid)
            if "AnswerArgument" in ans:
                GT.append(ans['AnswerArgument'].replace('T00:00:00Z', '').lower())

        ques_ans[ques_id] = {}
        ques_ans[ques_id]['text'] = text
        ques_ans[ques_id]['GT'] = GT
    return ques_ans

def _get_answer_coverage(GT, all_entities):
    if len(GT) == 0:
        return 0.
    if len(all_entities) == 0:
        return 0.
    else:
        found, total = 0., 0
        for answer in GT:
            if answer.lower() in all_entities:
                found += 1.
            total += 1
        return found / total

def myHandler(signum, frame):
    print("time out!!!")
    exit()

def get_extractentities_spo(spos):
    spo_entity = list()
    for line in spos:
        triple = line.strip().split('||')
        if len(triple) < 7 or len(triple) > 7: continue
        sub = triple[1]
        obj = triple[5]
        if 'T00:00:00Z' in sub:
            sub = sub.replace('T00:00:00Z', '')
        if 'T00:00:00Z' in obj:
            obj = obj.replace('T00:00:00Z', '')
        if 'corner#' in sub:
            sub = sub.replace('corner#', '').split('#')[0]
        if 'corner#' in obj:
            obj = obj.replace('corner#', '').split('#')[0]
        spo_entity.append(sub.lower())
        spo_entity.append(obj.lower())

    return list(set(spo_entity))

class QuestionUnionGST():
    def __init__(self, cfg, pro_info, topf, topg):
        self.cfg = cfg
        self.pro_info = pro_info
        self.topf = topf
        self.topg = topg

    def get_GST_spo_file(self, spo_file, GST_file, GST_entities, union_predicates, union_cornerpredicates):
        gstspo = []
        spolines = []
        state = {}
        uniongstspolines = []
        GST = nx.read_gpickle(GST_file)

        remove_duplicated_line = []
        with open(spo_file, 'r', encoding='utf-8') as f11:
            for line in f11:
                if line not in remove_duplicated_line:
                    remove_duplicated_line.append(line)

        for line in remove_duplicated_line:
            triple = line.strip().split('||')
            if len(triple) < 7 or len(triple) > 7: continue
            statement_id = line.strip().split("||")[0].replace("-ps:", "").replace("-pq:", "").lower()
            if statement_id not in state:
                state[statement_id] = dict()
                state[statement_id]['ps'] = []
                state[statement_id]['pq'] = []
            if "-ps:" in line.split("||")[0]:
                state[statement_id]['ps'].append(line)
            if "-pq:" in line.split("||")[0]:
                state[statement_id]['pq'].append(line)
            spolines.append(line)

        for line in spolines:
            statement_id = line.strip().split("||")[0].replace("-ps:", "").replace("-pq:", "").lower()
            if statement_id in union_cornerpredicates:
                uniongstspolines.append(line)
                continue
            sub = line.split("||")[1].replace("T00:00:00Z", "").lower()
            if "corner#" in sub: sub = sub.split("#")[1]
            obj = line.split("||")[5].replace("T00:00:00Z", "").lower()
            if "corner#" in obj: obj = obj.split("#")[1]
            pqrel = line.split("||")[4]
            pqrel_name = pqrel
            if pqrel in self.pro_info: pqrel_name = self.pro_info[pqrel]['label'].lower()

            if sub in GST_entities and obj in GST_entities:
                if pqrel_name + "::" + statement_id in union_predicates:
                    for item in state[statement_id]['ps']:
                        if item not in uniongstspolines:
                            uniongstspolines.append(item)

                    if line not in uniongstspolines:
                        uniongstspolines.append(line)

        if len(uniongstspolines) == 0:
            add_predicates = [node.split("::")[2].lower() for node in GST.nodes if
                     node.split("::")[1] == "Predicate"]
            for line in spolines:
                statement_id = line.strip().split("||")[0].replace("-ps:", "").replace("-pq:", "").lower()
                if statement_id in add_predicates:
                    uniongstspolines.append(line)

        for line in uniongstspolines:
            gstspo.append(line)
        return gstspo

    def get_QKG_GST_from_spo(self, id, question, gt, seeds_paths, add_spo_line, add_spo_score):
        path = self.cfg['ques_path'] + 'ques_' + str(id)
        spo_file = path + '/SPO.txt'
        spo_rank_file = path + '/SPO_rank.pkl'

        graph_file = path + '/QKG_' + str(self.topf) + '.gpickle'
        corner_file = path + '/cornerstone_' +  str(self.topf)+ '.pkl'
        unionGST_file = path + '/unionGST_' +  str(self.topf) + '_' + str(self.topg) + '.gpickle'

        result = "{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}|{10}|{11}|{12}|{13}|{14}|{15}|{16}|{17}".format(
            str(id),
            question,
            '0',
            '0',
            '0',

            '0',
            '0',
            '0',
            '0',
            '0',
            '0',

            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0')
        qterms = replace_symbols_in_question(question)
        q_ent = set()
        for term in qterms.split():
            if term not in stop_words:
                q_ent.add(term.lower())
        t1 = time.time()

        if not (os.path.exists(spo_file)):
            return result, [], []

        print("\n\nGenerating QKG from SPOS...\n")
        QKG, cornerstone, qkgspo, number_of_components, number_of_components_connect = call_main_GRAPH(spo_file, spo_rank_file, self.topf, q_ent, graph_file, corner_file, pro_info, seeds_paths, add_spo_line, add_spo_score)
        if number_of_components_connect < 1: return result, [], []
        S = [QKG.subgraph(c).copy() for c in nx.connected_components(QKG)]
        GST_list = []
        ques_gstac = 0.
        gst_len_entities = 0
        GST_node_len = 0
        GST_edge_len = 0
        complete_gstspo_ac = 0.
        complete_len_gst_entities = 0
        qkg_can_entities = [node.split("::")[2].replace('T00:00:00Z', '').lower() for node in QKG if
                            node.split("::")[1] == "Entity"]
        ques_qkgac = _get_answer_coverage(gt, qkg_can_entities)
        comgstspo = []
        gst_number_of_components_connect = 0
        for subg in S:
            try:
                signal.signal(signal.SIGALRM, myHandler)
                signal.alarm(MAX_TIME)
                GST = call_main_rawGST(subg, cornerstone, self.topg)
                print("DONE GST Algorithm...", str(id))
                if GST:
                    GST_list.append(GST)
            except:
                print(str(id) + " time out!")
                continue

            t2 = time.time()

        if len(GST_list) > 0:
            unionGST = nx.compose_all(GST_list)
            nx.write_gpickle(unionGST, unionGST_file)
            gst_number_of_components_connect = nx.number_connected_components(unionGST)
            gst_can_entities = [node.split("::")[2].replace('T00:00:00Z', '').lower() for node in unionGST.nodes() if
                                node.split("::")[1] == "Entity"]
            union_predicates = [node.split("::")[0] + "::" + node.split("::")[2].lower() for node in unionGST.nodes
                                 if node.split("::")[1] == "Predicate"]
            union_cornerpredicates = [node.split("::")[2].lower() for node in unionGST.nodes if
                                       node.split("::")[1] == "Predicate" and node in cornerstone]
            comgstspo = self.get_GST_spo_file(spo_file, unionGST_file, gst_can_entities, union_predicates, union_cornerpredicates)

            ques_gstac = _get_answer_coverage(gt, gst_can_entities)
            gst_len_entities = len(gst_can_entities)
            complete_gstspo_entities = get_extractentities_spo(comgstspo)
            complete_gstspo_ac = _get_answer_coverage(gt, complete_gstspo_entities)
            complete_len_gst_entities = len(complete_gstspo_entities)
            GST_node_len = len(unionGST.nodes())
            GST_edge_len = len(unionGST.edges())

        t3 = time.time()
        total_t = t3 - t1
        result = "{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}|{10}|{11}|{12}|{13}|{14}|{15}|{16}|{17}".format(
            str(id),
            question,
            str(len(q_ent)),
            str(len(cornerstone)),

            str(len(QKG.nodes())),
            str(len(QKG.edges())),
            str(len(qkg_can_entities)),

            str(GST_node_len),
            str(GST_edge_len),
            str(gst_len_entities),

            str(complete_len_gst_entities),

            str(ques_qkgac),
            str(ques_gstac),
            str(complete_gstspo_ac),
            str(total_t),
            str(number_of_components),
            str(number_of_components_connect),
            str(gst_number_of_components_connect))
        return result, qkgspo, comgstspo

if __name__ == "__main__":
    import argparse
    cfg = globals.get_config(globals.config_file)
    pro_info = globals.ReadProperty.init_from_config().property
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--dataset', type=str, default='test')
    argparser.add_argument('-f', '--topf', type=int, default=25)
    argparser.add_argument('-g', '--topg', type=int, default=25)

    args = argparser.parse_args()
    dataset = args.dataset
    topf = args.topf
    topg = args.topg

    test = cfg["benchmark_path"] + cfg["test_data"]
    dev = cfg["benchmark_path"] + cfg["dev_data"]
    train = cfg["benchmark_path"] + cfg["train_data"]
    if dataset == 'test': in_file = test
    elif dataset == 'dev': in_file = dev
    elif dataset == 'train': in_file = train
    ques_seeds_paths = {}
    #connect path for seed pairs
    path_file = cfg["connect_path"] + "seedpair_question_best_connectivity_paths_score.pkl"
    ques_seeds_paths = pickle.load(open(path_file, 'rb'))
    gst_GST = QuestionUnionGST(cfg, pro_info, topf, topg)
    result_number = {'qent': [], 'corner': [], 'qkgentity': [], 'qkgnode': [], 'gstnode': [], 'gstentity': [],'qkgac': [],
                     'gstac': [], 'completegstentity':[], 'complete_gstac': [], 'number_of_components':[], 'number_of_components_connect':[], 'time': []}

    groundtruth = get_groundtruth(in_file)
    data = json.load(open(in_file))
    gst_path = cfg["compactsubg_path"]
    os.makedirs(gst_path, exist_ok = True)
    #compact gst subgraph file
    gstspo_file = gst_path + dataset + '_' + str(topf) + '_'  + str(topg) + ".json"

    f1 = open(gstspo_file, "wb")
    for question in data:
        QuestionId = question["Id"]
        QuestionText = question["Question"]
        seeds_paths = {}
        add_spo_line = []
        score = {}
        add_spo_score = {}
        if QuestionId in ques_seeds_paths:
            seeds_paths.update(ques_seeds_paths[QuestionId]['ques_paths'])
            add_spo_line += ques_seeds_paths[QuestionId]['spo_line']
            score.update(ques_seeds_paths[QuestionId]['score'])
            add_spo_score.update({sta: float(score[sta]) for sta in score})
        GT = groundtruth[str(QuestionId)]['GT']
        result, qkgspo, comgstspo = gst_GST.get_QKG_GST_from_spo(QuestionId, QuestionText, GT, seeds_paths, add_spo_line, add_spo_score)
        gst = {
                "question": QuestionText,
                "id": QuestionId,
                "subgraph": comgstspo
            }
        f1.write(json.dumps(gst).encode("utf-8"))
        f1.write("\n".encode("utf-8"))
        print("\n\nQuestion Id-> ", QuestionId)
        print("Question -> ", QuestionText)
        print(result)

        result_number['qent'].append(float(result.split('|')[2]))
        result_number['corner'].append(float(result.split('|')[3]))
        result_number['qkgnode'].append(float(result.split('|')[4]))
        result_number['qkgentity'].append(float(result.split('|')[6]))
        result_number['gstnode'].append(float(result.split('|')[7]))
        result_number['gstentity'].append(float(result.split('|')[9]))
        result_number['completegstentity'].append(float(result.split('|')[10]))
        result_number['qkgac'].append(float(result.split('|')[11]))
        result_number['gstac'].append(float(result.split('|')[12]))
        result_number['complete_gstac'].append(float(result.split('|')[13]))
        result_number['number_of_components'].append(float(result.split('|')[15]))
        result_number['number_of_components_connect'].append(float(result.split('|')[16]))
        result_number['time'].append(float(result.split('|')[14]))

    print('Average  qent: ' + str(sum(result_number['qent']) / len(result_number['qent'])))
    print('Average  corner: ' + str(sum(result_number['corner']) / len(result_number['corner'])))
    print('Average  qkgnode: ' + str(sum(result_number['qkgnode']) / len(result_number['qkgnode'])))
    print('Average  gstnode: ' + str(sum(result_number['gstnode']) / len(result_number['gstnode'])))
    print('Average  qkgentity: ' + str(sum(result_number['qkgentity']) / len(result_number['qkgentity'])))
    print('Average  completegstentity: ' + str(sum(result_number['completegstentity']) / len(result_number['completegstentity'])))
    print('Average  qkgac: ' + str(sum(result_number['qkgac']) / len(result_number['qkgac'])))
    print('number of questions with answer in qkg: ' + str(len([item for item in result_number['qkgac'] if item > 0])))
    print('Average  gstac: ' + str(sum(result_number['gstac']) / len(result_number['gstac'])))
    print('number of questions with answer in gst: ' + str(len([item for item in result_number['gstac'] if item > 0])))
    print('Average  complete_gstac: ' + str(sum(result_number['complete_gstac']) / len(result_number['complete_gstac'])))
    print('number of questions with answer in complete_gstac: ' + str(len([item for item in result_number['complete_gstac'] if item > 0])))
    print('Average  number_of_components: ' + str(
        sum(result_number['number_of_components']) / len(result_number['number_of_components'])))
    print('Average  number_of_components_connect: ' + str(
        sum(result_number['number_of_components_connect']) / len(result_number['number_of_components_connect'])))
    print('Average  time: ' + str(sum(result_number['time']) / len(result_number['time'])))

    f1.close()