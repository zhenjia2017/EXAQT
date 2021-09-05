import json
from tqdm import tqdm

def combine_dist(dist1, dist2, w1):
    ensemble_dist = dist2.copy()
    for gid, prob in dist1.items():
        if gid in ensemble_dist:
            ensemble_dist[gid] = (1 - w1) * ensemble_dist[gid] + w1 * prob
        else:
            ensemble_dist[gid] = prob
    return ensemble_dist

#hits@5
def get_hitsmetric(kb_entities, dist_kb, topk, answers):
    pred_list = []
    for entity in kb_entities:
        pred_list.append((entity, dist_kb[entity]))
    sorted_l = sorted(pred_list, reverse=True, key=lambda t: t[1])
    hits5 = 0.0
    best5_pred = []
    if len(answers) == 0:
        if len(kb_entities) == 0:
            hits5 = 1.0  # hits@5
        else:
            hits5 = 1.0  # hits@5
    else:
        for j in range(0, len(sorted_l)):
            if j < topk:
                best5_pred.append(sorted_l[j][0] + '#' + str(sorted_l[j][1]))
                if sorted_l[j][0] in answers:
                    hits5 = 1.0

    return hits5, best5_pred

# presicion@1 recall@1 f1@1
def get_prfmetric(kb_pred_file, topk):
    pred_list = []
    with open(kb_pred_file) as f_kb:
        line_id = 0
        for line_kb in tqdm(zip(f_kb)):
            line_id += 1
            line_kb = json.loads(line_kb[0])
            answers = set([answer.lower() for answer in line_kb['answers']])
            dist_kb = line_kb['dist']
            kb_entities = set(dist_kb.keys())
    for entity in kb_entities:
        pred_list.append((entity, dist_kb[entity]))
    rank = 0
    pred_list1 = []
    for j in range(0, len(pred_list)):
        if j > 0:
            if pred_list[j][2] < pred_list[j - 1][2]:
                rank += 1
        pred_list1.append((pred_list[j][0], rank))
    answers_lower = []
    for item in answers:
        answers_lower.append(item.lower())
    correct, total = 0.0, 0.0
    pred_list = pred_list1
    for i in range(0, len(pred_list)):
        if pred_list[i][1] <= 1:  # rank 1
            ans1 = pred_list[i][0]
            if 'T00:00:00Z' in ans1: ans1 = ans1.replace('T00:00:00Z', '')
            if ans1.lower() in answers_lower:
                correct += 1
            total += 1
    if len(answers_lower) == 0:
        if total == 0:
            return 1.0, 1.0, 1.0  # precision, recall, f1
        else:
            return 0.0, 1.0, 0.0  # precision, recall, f1
    else:
        if total == 0:
            return 1.0, 0.0, 0.0  # precision, recall, f1
        else:
            precision, recall = correct / total, correct / len(answers_lower)
            f1 = 2.0 / (1.0 / precision + 1.0 / recall) if precision != 0 and recall != 0 else 0.0
            return precision, recall, f1

def get_mmr_metric(kb_entities, dist_kb, topk, answers):
    pred_list = []
    for entity in kb_entities:
        pred_list.append((entity, dist_kb[entity]))
    sorted_l = sorted(pred_list, reverse=True, key=lambda t: t[1])

    mrr = 0.0
    flag = 0
    if topk == 5:
        topk = len(pred_list)

    for i in range(0, len(sorted_l)):
        for answer in answers:
            if answer.lower() == sorted_l[i][0].lower():  # Order preserving
                mrr += 1.0 / float(i+1)
                flag = 1
                break
        if flag == 1:
            break

    return mrr

def get_one_f1(entities, dist, threshold, answers):
    best_entity = -1
    max_prob = 0.0
    preds = []
    for entity in entities:
        if dist[entity] > max_prob:
            max_prob = dist[entity]
            best_entity = entity
        if dist[entity] > threshold:
            preds.append(entity)
    precision, recall, f1, hits = cal_eval_metric(best_entity, preds, answers)
    return  precision, recall, f1, hits, best_entity

def cal_eval_metric(best_pred, preds, answers):
    correct, total = 0.0, 0.0
    for entity in preds:
        if entity in answers:
            correct += 1
        total += 1
    if len(answers) == 0:
        if total == 0:
            return 1.0, 1.0, 1.0, 1.0  # precision, recall, f1, hits
        else:
            return 0.0, 1.0, 0.0, 1.0  # precision, recall, f1, hits
    else:
        hits = float(best_pred in answers)
        if total == 0:
            return 1.0, 0.0, 0.0, hits  # precision, recall, f1, hits
        else:
            precision, recall = correct / total, correct / len(answers)
            f1 = 2.0 / (1.0 / precision + 1.0 / recall) if precision != 0 and recall != 0 else 0.0
            return precision, recall, f1, hits

def compare_pr(kb_pred_file, threshold, fp):
    kb_only_recall, kb_only_precision, kb_only_f1, kb_only_hits = [], [], [], []
    kb_only_mrr = []
    kb_only_hits5 = []

    with open(kb_pred_file) as f_kb:
        line_id = 0
        for line_kb in tqdm(zip(f_kb)):
            line_id += 1

            line_kb = json.loads(line_kb[0])

            answers = set([answer for answer in line_kb['answers']])

            id = line_kb['id']
            dist_kb = line_kb['dist']
            kb_entities = set(dist_kb.keys())

            p, r, f1, hits1, best_entity = get_one_f1(kb_entities, dist_kb, threshold, answers)
            kb_only_precision.append(p)
            kb_only_recall.append(r)
            kb_only_f1.append(f1)
            kb_only_hits.append(hits1)

            mmr = get_mmr_metric(kb_entities, dist_kb, 5, answers)
            kb_only_mrr.append(mmr)

            hits5, best_entities = get_hitsmetric(kb_entities, dist_kb, 5, answers)
            kb_only_hits5.append(hits5)

            result = "{0}|{1}|{2}|{3}|{4}".format(
                str(id), str(hits1), str(hits5), str(mmr), ";".join(best_entities))

            fp.write(result)
            fp.write("\n")
    fp.write ("line count |" + str(line_id))
    fp.write('\n')
    fp.write('Average hits1 |' + str(sum(kb_only_hits) / len(kb_only_hits)))
    fp.write('\n')
    fp.write('Average hits5 |' + str(sum(kb_only_hits5) / len(kb_only_hits5)))
    fp.write('\n')
    fp.write('Average mmr |'+ str(sum(kb_only_mrr) / len(kb_only_mrr)))
    fp.write('\n')
    fp.write('precision |' + str(sum(kb_only_precision) / len(kb_only_precision)))
    fp.write('\n')
    fp.write('recall |' + str(sum(kb_only_recall) / len(kb_only_recall)))
    fp.write('\n')
    fp.write('f1 |' + str(sum(kb_only_f1) / len(kb_only_f1)))
    fp.write('\n')

    print('Average hits1: ' , str(sum(kb_only_hits) / len(kb_only_hits)))
    print('Average hits5: ' , str(sum(kb_only_hits5) / len(kb_only_hits5)))
    print('Average mmr: ' , str(sum(kb_only_mrr) / len(kb_only_mrr)))
    print('precision: ' , str(sum(kb_only_precision) / len(kb_only_precision)))
    print('recall: ' , str(sum(kb_only_recall) / len(kb_only_recall)))
    print('f1: ' , str(sum(kb_only_f1) / len(kb_only_f1)))
