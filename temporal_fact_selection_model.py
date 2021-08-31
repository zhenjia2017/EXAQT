"""Script to get score and rank of each temporal fact to generate question compact subgraph with temporal facts.
"""
import truecase
import pickle
import json
import globals
import torch.nn as nn
import transformers
import torch
from tqdm import tqdm

DEVICE = "cuda"
MAX_LEN =  512
TRAIN_BATCH_SIZE = 50
VALID_BATCH_SIZE = 50
EPOCHS = 2
ACCUMULATION = 4

cfg = globals.get_config(globals.config_file)
BERT_PATH = cfg["model_path"] + "bert_base_cased/"
MODEL_PATH = cfg["model_path"] + "phase2_model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case = False)

class BERTDataset:
    def __init__(self, q1, q2, target):
        self.q1 = q1
        self.q2 = q2
        self.target = target
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.q1)

    def __getitem__(self, item):
        q1 = str(self.q1[item])
        q2 = str(self.q2[item])

        q1 = " ".join(q1.split())
        q2 = " ".join(q2.split())

        inputs = self.tokenizer.encode_plus(q1, q2, add_special_tokens=True, max_length=self.max_len, padding='longest',
                                            truncation=True)

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {'ids': torch.tensor(ids, dtype=torch.long), 'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.target[item], dtype=torch.long)}

# BERT BertMode, this works for both the cased and uncased version, change the directorty in the config path /config.BERT_PATH/  and it will work with both CASED and UNCASED version.
class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        outs = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(outs.pooler_output)
        output = self.out(bo)
        return output

def ques_tuple_prediction(ques,tuple,model):
    tokenizer = TOKENIZER
    max_len = MAX_LEN
    device = torch.device("cuda")
    inputs = tokenizer.encode_plus(
        ques,
        tuple,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt')

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    ids = ids.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)

    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

def get_statement_fact(spos, pro_info):
    spo_sta = {}
    for line in spos:
        triple = line.strip().split('||')
        if len(triple) < 7 or len(triple) > 7: continue
        statement_id = line.strip().split("||")[0].replace("-ps:", "").replace("-pq:", "").lower()
        sub_id = triple[1]
        obj_id = triple[5]
        if 'corner#' in sub_id: sub_id = sub_id.replace('corner#', '').split('#')[0]
        if 'corner#' in obj_id: obj_id = obj_id.replace('corner#', '').split('#')[0]
        sub_name = triple[2].replace('T00:00:00Z', '')
        obj_name = triple[6].replace('T00:00:00Z', '')
        rel_name = triple[4]
        if triple[4] in pro_info:
            rel_name = pro_info[triple[4]]['label']
        if statement_id not in spo_sta:
            spo_sta[statement_id] = dict()
            spo_sta[statement_id]['ps'] = []
            spo_sta[statement_id]['pq'] = []
            spo_sta[statement_id]['qid'] = []
        spo_sta[statement_id]['qid'].append(sub_id)
        spo_sta[statement_id]['qid'].append(obj_id)
        if "-ps:" in line.split("||")[0]:
            spo_sta[statement_id]['ps'].append(sub_name + ' ' + rel_name + ' ' + obj_name)
        if "-pq:" in line.split("||")[0]:
            spo_sta[statement_id]['pq'].append(' ' + rel_name + ' ' + obj_name)

    return spo_sta

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--dataset', type=str, default='test')
    argparser.add_argument('-f', '--topf', type=int, default=25)
    argparser.add_argument('-g', '--topg', type=int, default=25)

    cfg = globals.get_config(globals.config_file)
    pro_info = globals.ReadProperty.init_from_config().property
    args = argparser.parse_args()
    print("Predicting Running")
    dataset = args.dataset
    topf = args.topf
    topg = args.topg
    test = cfg["benchmark_path"] + cfg["test_data"]
    dev = cfg["benchmark_path"] + cfg["dev_data"]
    train = cfg["benchmark_path"] + cfg["train_data"]

    if dataset == 'test':
        in_file = test
    elif dataset == 'dev':
        in_file = dev
    elif dataset == 'train':
        in_file = train

    #input files
    spo_file = cfg["compactgst"] +  dataset + '_' + str(topf) + '_' + str(topg) + ".json"
    tempspo_file = cfg["temcompactsubg_path"] + dataset + '_' + str(topf) + '_' + str(topg) + '_temp.json'

    #output file
    tempspo_rank = cfg["temcompactsubg_path"] + dataset + '_' + str(topf) + '_' + str(topg) + '_temp_rank'

    device = torch.device("cuda")
    MODEL = BERTBaseUncased()
    model = torch.load(MODEL_PATH)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in model.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    MODEL.load_state_dict(new_state_dict)
    MODEL.to(device)
    MODEL.eval()
    print("load model successfully!")
    datas = []
    ranks = []
    with open(tempspo_file, encoding='utf-8') as f_in:
        for line in tqdm(f_in):
            line = json.loads(line)
            datas.append(line)

    for data in datas:
        sta_score = []
        ques_id = data["id"]
        ques_text = data["question"]
        tempspo_lines = data["tempfact"]
        tempspo_scores = []
        ques_text = truecase.get_true_case(ques_text)
        if len(tempspo_lines) > 0:
            spo_sta = get_statement_fact(tempspo_lines, pro_info)
            for statement_id in spo_sta:
                context = " ".join(spo_sta[statement_id]['ps']) + " and".join(spo_sta[statement_id]['pq'])
                score = ques_tuple_prediction(ques_text, context, MODEL)
                sta_score.append((statement_id, float(score), context))
            sta_score = sorted(sta_score, key=lambda tup: tup[1], reverse=True)

            for i, item in enumerate(sta_score):
                tempspo_scores.append(str(i + 1) + '\t' + str(item[0]) + '\t' + str(item[1]) + '\n')

        rank_fact_dic = {}
        rank_fact_dic['question'] = data["question"]
        rank_fact_dic['id'] = data["id"]
        rank_fact_dic['rank'] = []
        for i, item in enumerate(tempspo_scores):
            if int(item.split('\t')[0]) <= 100:
                rank_fact_dic['rank'].append(str(item.split('\t')[0]) + '\t' + str(item.split('\t')[1]) + '\t' + str(item.split('\t')[2]))
        ranks.append(rank_fact_dic)

    pickle.dump(ranks, open(tempspo_rank, 'wb'))



