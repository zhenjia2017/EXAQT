import re
import truecase
import hashlib
from transformers import BertTokenizer, BertModel
import torch.nn
from exaqt.library.utils import get_config, get_logger
from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient

class SeedPathExtractor:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.nerd = self.config["nerd"]
        # initialize clocq
        if config["clocq_use_api"]:
            self.clocq = CLOCQInterfaceClient(host=config["clocq_host"], port=config["clocq_port"])
        else:
            self.clocq = CLOCQ(dev=True)

        self.max_fact_number = self.config["max_fact_number"]
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        self.model = BertModel.from_pretrained("bert-base-cased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def write_fact_to_lines(self, fact_dic, ids):
        templines = []
        # for entities
        ENT_PATTERN = re.compile('^Q[0-9]+$')
        # for predicates
        PRE_PATTERN = re.compile('^P[0-9]+$')
        for sta in fact_dic:
            t = fact_dic[sta]['ps'][0]
            sub = t[0]['id'].strip('"')
            pre = t[1]['id'].strip('"')
            obj = t[2]['id'].strip('"')
            subname = sub
            objname = obj
            if isinstance(t[0]['label'], list):
                for item in t[0]['label']:
                    if ENT_PATTERN.match(item) == None:
                        subname = item
                        break
            else:
                subname = t[0]['label'].strip('"')
            if isinstance(t[2]['label'], list):
                for item in t[2]['label']:
                    if ENT_PATTERN.match(item) == None:
                        objname = item
                        break
            else:
                objname = t[2]['label'].strip('"')
            for id in ids:
                if sub == id:
                    sub = "corner#" + sub
                if obj == id:
                    obj = "corner#" + obj

            p = sub + "||" + subname + "||" + pre
            ps_line = sta + "-ps:" + "||" + p + "||" + pre + "||" + obj + "||" + str(objname)
            templines.append(ps_line)
            for pqt in fact_dic[sta]['pq']:
                pre = pqt[0]['id'].strip('"')
                obj = pqt[1]['id'].strip('"')
                objname = obj
                if isinstance(pqt[1]['label'], list):
                    for item in pqt[1]['label']:
                        if ENT_PATTERN.match(item) == None:
                            objname = item
                            break
                else:
                    objname = pqt[1]['label'].strip('"')
                for id in ids:
                    if obj == id:
                        obj = "corner#" + obj
                pq_line = sta + "-pq:" + "||" + p + "||" + pre + "||" + obj + "||" + objname
                if pq_line not in templines:
                    templines.append(pq_line)
        return templines

    def get_fact_dic(self, fact):
        fact_dic = {}
        ps_context = []
        pq_context = []
        # one fact including (sub, rel, obj, qualifiers)
        str2hash = ''
        for it in fact:
            str2hash += it
        md5hash = hashlib.md5(str2hash.encode()).hexdigest()
        # first entity sub in the fact as the start of a statement id
        statementid = fact[0] + '-' + md5hash
        if statementid not in fact_dic:
            fact_dic[statementid] = {}
            fact_dic[statementid]['ps'] = []
            fact_dic[statementid]['pq'] = []
        sub = fact[0].strip('"')
        rel = fact[1].strip('"')
        obj = fact[2].strip('"')

        sub_name = self.clocq.get_label(sub)
        rel_name = self.clocq.get_label(rel)
        obj_label = self.clocq.get_label(obj)

        if obj_label == 'None':
            obj_name = obj
        else:
            obj_name = obj_label
        if "T00:00:00Z" in obj_name:
            obj_name = obj_name.replace("T00:00:00Z", "")

        ps_context.append(sub_name + ' ' + rel_name + ' ' + obj_name)
        fact_dic[statementid]['ps'].append(
            [{'id': sub, 'label': sub_name}, {'id': rel, 'label': rel_name},
             {'id': obj, 'label': obj_name}])
        pqn = int((len(fact) - 3) / 2)
        if pqn > 0:
            for i in range(pqn):
                pq_rel = fact[3 + i * 2].strip('"')
                pq_obj = fact[4 + i * 2].strip('"')
                pq_rel_name = self.clocq.get_label(pq_rel)
                pq_obj_label = self.clocq.get_label(pq_obj)
                if pq_obj_label == 'None':
                    pq_obj_name = pq_obj
                else:
                    pq_obj_name = pq_obj_label
                if "T00:00:00Z" in pq_obj_name:
                    pq_obj_name = pq_obj_name.replace("T00:00:00Z", "")
                pq_fact = [{'id': pq_rel, 'label': pq_rel_name},
                           {'id': pq_obj, 'label': pq_obj_name}]
                if pq_fact not in fact_dic[statementid]['pq']:
                    fact_dic[statementid]['pq'].append(pq_fact)
                    pq_context.append(pq_rel_name + ' ' + pq_obj_name)
        fact_context = " ".join(ps_context) + " " + " and ".join(pq_context) + " "
        return fact_dic, fact_context

    def get_bert_emb(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs.to(self.device)
        outputs = self.model(**inputs)
        text_emb = outputs[0][0]
        text_emb = text_emb[0]
        return text_emb

    def get_best_path(self, pair_all_path, pair_connectivity, question):
        fact_dic = {}
        # seed_pairs = pair_connectivity.keys()
        seed_pairs = pair_all_path.keys()
        seeds = set()
        for pair in seed_pairs:
            seeds.add(pair.split('||')[0])
            seeds.add(pair.split('||')[1])
        all_paths = {seed_pair: [] for seed_pair in seed_pairs}
        ques_paths_similarity = {seed_pair: [] for seed_pair in seed_pairs}
        best_paths = {seed_pair: [] for seed_pair in seed_pairs}
        pair_best_path_score = {seed_pair: 0.0 for seed_pair in seed_pairs}
        ques_emb = self.get_bert_emb(truecase.get_true_case(question))
        count = 0
        for seed_pair, connectivity in pair_connectivity.items():
            if connectivity == 1:
                if seed_pair not in pair_all_path:
                    print(seed_pair)
                    continue
                paths = pair_all_path[seed_pair]
                one_hop_paths = []
                for fact in paths:
                    count += 1
                    path_simi = {'index': count, 'sta': [], 'context': '', 'path': fact}
                    one_fact_dic, one_fact_context = self.get_fact_dic(fact)
                    fact_dic.update(one_fact_dic)
                    path_simi['sta'].append(list(one_fact_dic.keys())[0])
                    path_simi['context'] = one_fact_context
                    one_hop_paths.append(path_simi)
                ques_paths_similarity[seed_pair] += one_hop_paths
                all_paths[seed_pair] += [item['sta'] for item in one_hop_paths]
            elif connectivity == 0.5:
                if seed_pair not in pair_all_path:
                    print(seed_pair)
                    continue
                paths = pair_all_path[seed_pair]
                two_hop_paths = []
                for path in paths:
                    # check if path is multiple facts
                    firsts = path[0]
                    seconds = path[1]
                    if len(firsts) < 1 or len(seconds) < 1:
                        continue
                    if len(firsts) > self.max_fact_number:
                        firsts = firsts[:self.max_fact_number]
                    if len(seconds) > self.max_fact_number:
                        seconds = seconds[:self.max_fact_number]
                    for facts_fir in firsts:
                        fact_fir_dic, fact_fir_context = self.get_fact_dic(facts_fir)
                        fact_dic.update(fact_fir_dic)
                        for facts_sec in seconds:
                            count += 1
                            fact_sec_dic, fact_sec_context = self.get_fact_dic(facts_sec)
                            fact_dic.update(fact_sec_dic)
                            path_context = fact_fir_context + '; ' + fact_sec_context
                            path_simi = {
                                'index': count,
                                'sta': [list(fact_fir_dic.keys())[0], list(fact_sec_dic.keys())[0]],
                                'context': path_context,
                                'path': [facts_fir, facts_sec]
                            }
                            two_hop_paths.append(path_simi)
                ques_paths_similarity[seed_pair] += two_hop_paths
                all_paths[seed_pair] += [item['sta'] for item in two_hop_paths]

        statements_in_bestpath = list()
        for pair, path_list in ques_paths_similarity.items():
            best_simi = 0.
            count = 0
            best_path = []
            for item in path_list:
                try:
                    path_emb = self.get_bert_emb(item['context'])
                    simi = float(self.cos(ques_emb, path_emb).data.cpu().numpy())
                    item.update({'sim': simi})
                    if simi > best_simi:
                        best_simi = simi
                        best_path = item['sta'].copy()
                except:
                    print('\nfail to get bert emb!!')
                    print(item['context'])
                count += 1
                if count > self.max_fact_number:
                    break

            best_paths[pair] += [best_path]
            pair_best_path_score[pair] = best_simi
            statements_in_bestpath += best_path

        # total_question += 1
        best_spo_line = []
        spo_line = self.write_fact_to_lines(fact_dic, list(seeds))
        for line in spo_line:
            triple = line.strip().split('||')
            if len(triple) < 7 or len(triple) > 7: continue
            statement_id = line.strip().split("||")[0].replace("-ps:", "").replace("-pq:", "")
            if statement_id in statements_in_bestpath:
                best_spo_line.append(line)

        return best_paths, ques_paths_similarity, best_spo_line, pair_best_path_score

    def _seed_connecitivity(self, seed_pairs):
        pair_all_path = {}
        pair_connectivity = {}
        for pair in seed_pairs:
            item1 = pair.split('||')[0]
            item2 = pair.split('||')[1]
            connectivity = self.clocq.connectivity_check(item1, item2)
            pair_connectivity[pair] = connectivity
            if connectivity > 0:
                try:
                    paths = self.clocq.connect(item1, item2)
                    pair_all_path[pair] = paths
                except:
                    self.logger.info(f"{item1} and {item2} is connected but have no path between them")
        return pair_connectivity, pair_all_path

    def seed_pairs_best_path(self, instance):
        self.logger.debug(f"Running Connectivity Checking")
        question = instance["Question"]
        seed_pairs = []
        ques_seed_qid = set()
        ques_seed_qid |= set([item[0] for item in instance["elq"]])
        if self.nerd == "elq-wat":
            ques_seed_qid |= set([item[0][0] for item in instance["wat"]])
        elif self.nerd == "elq-tagme":
            ques_seed_qid |= set([item[0] for item in instance["tagme"]])
        elif self.nerd == "elq-tagme-wat":
            ques_seed_qid |= set([item[0][0] for item in instance["wat"]])
            ques_seed_qid |= set([item[0] for item in instance["tagme"]])

        ques_seed_qid = list(ques_seed_qid)
        wikidata_entities = [evidence["wikidata_entities"] for evidence in instance["candidate_evidences"]]
        evidence_seed_qid = set()
        for items in wikidata_entities:
            evidence_seed_qid |= set([item["id"] for item in items if item["id"] in ques_seed_qid])

        evidence_seed_qid = list(evidence_seed_qid)

        if len(evidence_seed_qid) > 1:
            for i in range(0, len(evidence_seed_qid) - 1):
                item1 = evidence_seed_qid[i]
                for j in range(i + 1, len(evidence_seed_qid)):
                    item2 = evidence_seed_qid[j]
                    str_li = [item1, item2]
                    str_li.sort()
                    str_pair = '||'.join(str_li)
                    if str_pair not in seed_pairs:
                        seed_pairs.append(str_pair)
        self.logger.info(f"evidence seed pairs): {seed_pairs} ")
        pair_connectivity, pair_all_path = self._seed_connecitivity(seed_pairs)
        best_paths, ques_paths_similarity, best_spo_line, pair_best_path_score = self.get_best_path(pair_all_path, pair_connectivity, question)

        result = {
            "corner_pairs": seed_pairs,
            "pair_connectivity": pair_connectivity,
            "best_paths": best_paths,
            "all_paths": pair_all_path,
            "ques_paths_detail": ques_paths_similarity,
            "spo_line": best_spo_line,
            "score": pair_best_path_score
        }


        return result

