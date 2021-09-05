import torch
from tqdm import tqdm
import json
import yaml
import os
from model import Exact
from script_listscore import compare_pr
from data_loader import DataLoader
from util import use_cuda, sparse_bmm, get_config, load_dict, cal_accuracy, load_map
import globals

def train(cfg):
    print("training ...")
    # prepare data
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    trelation2id = load_dict(cfg['data_folder'] + cfg['trelation2id'])
    date2id = load_dict(cfg['data_folder'] + cfg['date2id'])
    tempf2id = load_map(cfg['data_folder'] + cfg['tempfact2id'])
    category2id = load_dict(cfg['data_folder'] + cfg['category2id'])
    signal2id = load_dict(cfg['data_folder'] + cfg['signal2id'])

    train_data = DataLoader(cfg['data_folder'] + cfg['train_data'], word2id, relation2id, trelation2id, entity2id, date2id, tempf2id, category2id, signal2id, cfg['max_query_word'], cfg['max_temp_fact'])
    valid_data = DataLoader(cfg['data_folder'] + cfg['dev_data'], word2id, relation2id, trelation2id, entity2id, date2id, tempf2id, category2id, signal2id, cfg['max_query_word'], cfg['max_temp_fact'])

    # create model & set parameters
    my_model = get_model(cfg, train_data.num_kb_relation, len(entity2id), len(word2id), len(category2id), len(signal2id), len(date2id), len(tempf2id))
    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg['learning_rate'])

    best_dev_acc = 0.0

    for epoch in range(cfg['num_epoch']):
        try:
            print('epoch', epoch)
            train_data.reset_batches(is_sequential = cfg['is_debug'])
            # Train
            my_model.train()
            train_loss, train_acc, train_max_acc = [], [], []
            for iteration in tqdm(range(train_data.num_data // cfg['batch_size'])):
                batch = train_data.get_batch(iteration, cfg['batch_size'], cfg['fact_dropout'])
                loss, pred, pred_dist = my_model(batch)
                #break
                pred = pred.data.cpu().numpy()
                acc, max_acc = cal_accuracy(pred, batch[-1])
                train_loss.append(loss.data)
                train_acc.append(acc)
                train_max_acc.append(max_acc)
                # back propogate
                my_model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg['gradient_clip'])
                optimizer.step()
            print('avg_training_loss', sum(train_loss) / len(train_loss))
            print('max_training_acc', sum(train_max_acc) / len(train_max_acc))
            print('avg_training_acc', sum(train_acc) / len(train_acc))
            print("validating ...")
            eval_acc = inference_best_acc(my_model, valid_data, entity2id, cfg)
            if eval_acc > best_dev_acc and cfg['to_save_model']:
               print("saving model to", cfg['model_folder'] + cfg['save_model_file'])
               torch.save(my_model.state_dict(), cfg['model_folder'] + cfg['save_model_file'])
               best_dev_acc = eval_acc

        except KeyboardInterrupt:
            break

    return

def output_pred_dist(pred_dist, answer_dist, id2entity, start_id, data_loader, f_pred):
    for i, p_dist in enumerate(pred_dist):
        data_id = start_id + i
        l2g = {l:g for g, l in data_loader.global2local_entity_maps[data_id].items()}
        output_dist = {id2entity[l2g[j]]: float(prob) for j, prob in enumerate(p_dist.data.cpu().numpy()) if j < len(l2g)}
        answers = [answer['text'] if type(answer['kb_id']) == int else answer['kb_id'] for answer in data_loader.data[data_id]['answers']]
        f_pred.write(json.dumps({'dist': output_dist, 'id': data_loader.data[data_id]['id'], 'answers':answers, 'seeds': data_loader.data[data_id]['seed_entities'], 'tuples': data_loader.data[data_id]['subgraph']['tuples']}) + '\n')

def inference_best_acc(my_model, valid_data, entity2id, cfg, log_info=False):
    # Evaluation
    my_model.eval()
    eval_loss, eval_acc, eval_max_acc = [], [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    valid_data.reset_batches(is_sequential = True)
    test_batch_size = 32
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

def dev(cfg):
    print("testing ...")
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    trelation2id = load_dict(cfg['data_folder'] + cfg['trelation2id'])
    date2id = load_dict(cfg['data_folder'] + cfg['date2id'])
    tempf2id = load_map(cfg['data_folder'] + cfg['tempfact2id'])
    category2id = load_dict(cfg['data_folder'] + cfg['category2id'])
    signal2id = load_dict(cfg['data_folder'] + cfg['signal2id'])
    dev_data = DataLoader(cfg['data_folder'] + cfg['dev_data'], word2id, relation2id, trelation2id, entity2id,
                           date2id, tempf2id, category2id, signal2id, cfg['max_query_word'], cfg['max_temp_fact'])

    # Test set evaluation
    print("evaluating on test")
    print('loading model from ...', cfg['model_folder'] + cfg['save_model_file'])

    my_model = get_model(cfg, dev_data.num_kb_relation, len(entity2id), len(word2id), len(category2id), len(signal2id), len(date2id), len(tempf2id))
    my_model.load_state_dict(torch.load(cfg['model_folder'] + cfg['save_model_file']))

    dev_acc = inference(my_model, dev_data, entity2id, cfg, log_info=True)

    return dev_acc

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

def generate_config_file(answer_predict_path, parameter, config_file, train_subg, dev_subg, test_subg):

    if parameter == 'exaqt':
        TCE = True
        TSE = True
        TE = True
        ATR = True
        TEE = True
    if parameter == 'exaqt-tce':
        TCE = False
        TSE = True
        TE = True
        ATR = True
        TEE = True
    if parameter == 'exaqt-tse':
        TCE = True
        TSE = False
        TE = True
        ATR = True
        TEE = True
    if parameter == 'exaqt-tee':
        TCE = True
        TSE = True
        TE = True
        ATR = True
        TEE = False
    if parameter == 'exaqt-te':
        TCE = True
        TSE = True
        TE = False
        ATR = True
        TEE = True
    if parameter == 'exaqt-tpa':
        TCE = True
        TSE = True
        TE = True
        ATR = False
        TEE = True

    config_dic = {}
    config_dic['name'] = 'tempq'
    config_dic['data_folder'] =  answer_predict_path
    config_dic['model_folder'] =  answer_predict_path + 'model/'
    config_dic['train_data'] =  train_subg
    config_dic['dev_data'] =  dev_subg
    config_dic['test_data'] =  test_subg

    config_dic['entity2id'] = 'entities.txt'
    config_dic['trelation2id'] = 'trelations.txt'
    config_dic['date2id'] = 'dates.txt'
    config_dic['relation2id'] = 'relations.txt'
    config_dic['word2id'] = 'words.txt'
    config_dic['category2id'] = 'categories.txt'
    config_dic['signal2id'] = 'signals.txt'
    config_dic['tempfact2id'] = 'tempfacts2id.pkl'

    config_dic['relation_emb_file'] = 'relation_emb_100d.npy'
    config_dic['word_emb_file'] = 'word_emb_100d.npy'
    config_dic['entity_emb_file'] =  'entity_emb_100d.npy'
    config_dic['date_emb_file'] = 'date_te_emb_100d.npy'
    config_dic['tempfact_te_emb_file'] = 'tempfact_te_emb_500d.npy'
    config_dic['tempfact_emb_file'] = 'tempfact_emb_400d.npy'

    config_dic['to_save_model'] =  True
    config_dic['save_model_file'] =  'best_model_' + parameter
    config_dic['pred_file'] =  'pred_'  + parameter
    config_dic['load_model_file'] =  'best_model_' + parameter

    config_dic['TCE'] =  TCE
    config_dic['TSE'] =  TSE
    config_dic['TE'] =  TE
    config_dic['TEE'] = TEE
    config_dic['ATR'] = ATR

    # graph options
    config_dic['fact_dropout'] =  0.1

    config_dic['num_layer'] =  3
    config_dic['max_query_word'] =  40
    config_dic['max_temp_fact'] = 40
    config_dic['entity_dim'] =  100
    config_dic['word_dim'] =  100
    config_dic['tem_dim'] =  100
    config_dic['fact_dim'] = 400
    config_dic['pagerank_lambda'] =  0.8
    config_dic['corner_lambda'] =  1
    config_dic['fact_scale'] =  3

    # optimization
    config_dic['num_epoch'] =  100
    config_dic['batch_size'] =  25
    config_dic['gradient_clip'] =  1
    config_dic['learning_rate'] =  0.001
    config_dic['lstm_dropout'] =  0.3
    config_dic['linear_dropout'] = 0.2
    config_dic['is_debug'] =  True

    with open(config_file, 'w') as file:
        yaml.dump(config_dic, file, default_flow_style=False)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-p', '--parameter', type=str, default='exaqt')
    args = argparser.parse_args()
    parameter = args.parameter

    topf = topg = topt = 25

    cfg = globals.get_config(globals.config_file)
    answer_predict_path = cfg['answer_predict_path']

    train_subgraph = "train_subgraph.json"
    dev_subgraph = "dev_subgraph.json"
    test_subgraph = "test_subgraph.json"

    config_path = answer_predict_path + 'config'
    result_path = answer_predict_path + 'result'
    model_path = answer_predict_path + 'model'
    os.makedirs(config_path, exist_ok = True)
    os.makedirs(result_path, exist_ok = True)
    os.makedirs(model_path, exist_ok = True)

    config_file = config_path +  '/gcn_config_' + parameter + '.yml'
    dev_re_fp = result_path + '/gcn_result_dev_' + parameter + '.txt'
    test_re_fp = result_path + '/gcn_result_test_' + parameter + '.txt'

    generate_config_file(answer_predict_path, parameter, config_file, train_subgraph, dev_subgraph, test_subgraph)
    GCN_CFG = get_config(config_file)
    train(GCN_CFG)
    # result on dev set
    dev_acc = dev(GCN_CFG)
    pred_kb_file = GCN_CFG['model_folder'] + GCN_CFG['pred_file']
    threshold = 0.2
    compare_pr(pred_kb_file, threshold, open(dev_re_fp, 'w', encoding='utf-8'))
    # result on test set
    test_acc = test(GCN_CFG)
    pred_kb_file = GCN_CFG['model_folder'] + GCN_CFG['pred_file']
    compare_pr(pred_kb_file, threshold, open(test_re_fp, 'w', encoding='utf-8'))



