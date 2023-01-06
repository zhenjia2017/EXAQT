import torch
from tqdm import tqdm
import json
from model import Exact
from script_listscore import compare_pr
from data_loader import DataLoader
from util import use_cuda, sparse_bmm, get_config, load_dict, cal_accuracy, load_map
import globals

def output_pred_dist(pred_dist, answer_dist, id2entity, start_id, data_loader, f_pred):
    for i, p_dist in enumerate(pred_dist):
        data_id = start_id + i
        l2g = {l:g for g, l in data_loader.global2local_entity_maps[data_id].items()}
        output_dist = {id2entity[l2g[j]]: float(prob) for j, prob in enumerate(p_dist.data.cpu().numpy()) if j < len(l2g)}
        answers = [answer['text'] if type(answer['kb_id']) == int else answer['kb_id'] for answer in data_loader.data[data_id]['answers']]
        f_pred.write(json.dumps({'dist': output_dist, 'id': data_loader.data[data_id]['id'], 'answers':answers, 'seeds': data_loader.data[data_id]['seed_entities'], 'tuples': data_loader.data[data_id]['subgraph']['tuples']}) + '\n')

def inference(my_model, valid_data, entity2id, cfg, log_info=False):
    # Evaluation
    my_model.eval()
    eval_loss, eval_acc, eval_max_acc = [], [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    valid_data.reset_batches(is_sequential = True)
    test_batch_size = 1
    if log_info:
        f_pred = open(cfg['model_folder'] + cfg['pred_file'], 'w')
    for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
        batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
        loss, pred, pred_dist = my_model(batch)
        pred = pred.data.cpu().numpy()
        acc, max_acc = cal_accuracy(pred, batch[-1])
        if log_info:
            output_pred_dist(pred_dist, batch[-1], id2entity, iteration * test_batch_size, valid_data, f_pred)
        eval_loss.append(loss.data)
        eval_acc.append(acc)
        eval_max_acc.append(max_acc)

    print('avg_loss', sum(eval_loss) / len(eval_loss))
    print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
    print('avg_acc', sum(eval_acc) / len(eval_acc))

    return sum(eval_acc) / len(eval_acc)

def test(cfg):
    print("testing ...")
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    trelation2id = load_dict(cfg['data_folder'] + cfg['trelation2id'])
    date2id = load_dict(cfg['data_folder'] + cfg['date2id'])
    tempf2id = load_map(cfg['data_folder'] + cfg['tempfact2id'])
    category2id = load_dict(cfg['data_folder'] + cfg['category2id'])
    signal2id = load_dict(cfg['data_folder'] + cfg['signal2id'])
    test_data = DataLoader(cfg['data_folder'] + cfg['test_data'], word2id, relation2id, trelation2id, entity2id,
                           date2id, tempf2id, category2id, signal2id, cfg['max_query_word'], cfg['max_temp_fact'])

    # Test set evaluation
    print("evaluating on test")
    print('loading model from ...', cfg['model_folder'] + cfg['save_model_file'])

    my_model = get_model(cfg, test_data.num_kb_relation, len(entity2id), len(word2id), len(category2id), len(signal2id), len(date2id), len(tempf2id))
    my_model.load_state_dict(torch.load(cfg['model_folder'] + cfg['save_model_file']))

    test_acc = inference(my_model, test_data, entity2id, cfg, log_info=True)

    return test_acc

def get_model(cfg, num_kb_relation, num_entities, num_vocab, num_categories, num_signals, num_dates, num_tempf):
    pretrained_word_emb_file = cfg['data_folder'] + cfg['word_emb_file']
    pretrained_entity_emb_file = cfg['data_folder'] + cfg['entity_emb_file']
    pretrained_relation_emb_file = cfg['data_folder'] + cfg['relation_emb_file']
    pretrained_date_tem_file = cfg['data_folder'] + cfg['date_emb_file']
    pretrained_tempfact_te_emb_file = cfg['data_folder'] + cfg['tempfact_te_emb_file']
    pretrained_tempfact_emb_file = cfg['data_folder'] + cfg['tempfact_emb_file']

    type_dim = num_categories #multi-hot encoding dimension
    sig_dim = num_signals #multi-hot encoding dimension

    my_model = use_cuda(Exact(pretrained_word_emb_file, pretrained_relation_emb_file, pretrained_entity_emb_file, pretrained_date_tem_file, pretrained_tempfact_te_emb_file, pretrained_tempfact_emb_file, cfg['num_layer'], num_kb_relation, num_entities, num_vocab, num_tempf,  cfg['entity_dim'], cfg['word_dim'], cfg['tem_dim'], cfg['fact_dim'],
                 type_dim, sig_dim, cfg['pagerank_lambda'], cfg['fact_scale'], cfg['lstm_dropout'], cfg['linear_dropout'], cfg['TCE'], cfg['TSE'], cfg['TEE'], cfg['TE'], cfg['ATR']))

    return my_model

def evaluate_result_for_category(benchmark_file, test_re_fp, out_file):
    fo = open(out_file, 'w', encoding='utf-8')
    result = open(test_re_fp, 'r', encoding='utf-8')
    res_lines = result.readlines()
    result_number = {}

    with open(benchmark_file, encoding='utf-8') as json_data:
        list = json.load(json_data)
        json_data.close()

    type_dic = {'Explicit': [], 'Implicit': [], 'Temp.Ans': [], 'Ordinal': []}
    p1_res_dic = {'Explicit': [], 'Implicit': [], 'Temp.Ans': [], 'Ordinal': []}
    h5_res_dic = {'Explicit': [], 'Implicit': [], 'Temp.Ans': [], 'Ordinal': []}
    mrr_res_dic = {'Explicit': [], 'Implicit': [], 'Temp.Ans': [], 'Ordinal': []}

    count = 0
    for item in list:
        source = item["Data source"]
        types = item["Type"]
        id = str(item["Id"])
        for type in types:
            type_dic[type].append(id)
        count += 1

    for line in res_lines:
        if '|' in line and len(line.split('|')) > 3:
            # print (line)
            id = line.split('|')[0]
            p1 = float(line.split('|')[1])
            h5 = float(line.split('|')[2])
            mrr = float(line.split('|')[3])

            for key, value in type_dic.items():
                if id in value:
                    p1_res_dic[key].append(p1)
                    h5_res_dic[key].append(h5)
                    mrr_res_dic[key].append(mrr)

    for key in p1_res_dic.keys():
        if key not in result_number:
            result_number[key] = {}
        result_number[key]['p1'] = str(round(sum(p1_res_dic[key]) / len(p1_res_dic[key]), 3))
        print('Average  hits1: ', str(sum(p1_res_dic[key]) / len(p1_res_dic[key])))

    for key in mrr_res_dic.keys():
        print(key)
        result_number[key]['mrr'] = str(round(sum(mrr_res_dic[key]) / len(mrr_res_dic[key]), 3))
        print('Average  mrr: ', str(sum(mrr_res_dic[key]) / len(mrr_res_dic[key])))

    for key in h5_res_dic.keys():
        print(key)
        result_number[key]['hits5'] = str(round(sum(h5_res_dic[key]) / len(h5_res_dic[key]), 3))

        print('Average  hits5: ', str(sum(h5_res_dic[key]) / len(h5_res_dic[key])))

    for key in result_number:
        fo.write(key + '|')
        for item in result_number[key]:
            fo.write(result_number[key][item])
            fo.write('|')
    fo.write('\n')
    fo.close()


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-p', '--parameter', type=str, default='exaqt')
    args = argparser.parse_args()
    parameter = args.parameter

    cfg = globals.get_config(globals.config_file)
    answer_predict_path = cfg['answer_predict_path']

    test_subgraph = "test_subgraph.json"

    config_path = answer_predict_path + 'config'

    config_file = config_path +  '/gcn_config_' + parameter + '.yml'
    test_re_fp = 'evaluate_test_' + parameter + '.txt'
    GCN_CFG = get_config(config_file)
    # result on test set
    threshold = 0.2
    test_acc = test(GCN_CFG)
    pred_kb_file = GCN_CFG['model_folder'] + GCN_CFG['pred_file']
    compare_pr(pred_kb_file, threshold, open(test_re_fp, 'w', encoding='utf-8'))
    # result for different categories
    benchmark_file = cfg["benchmark_path"] + cfg["test_data"]
    test_re_category_file = 'evaluate_test_' + parameter + '_category.txt'
    evaluate_result_for_category(benchmark_file, test_re_fp, test_re_category_file)



