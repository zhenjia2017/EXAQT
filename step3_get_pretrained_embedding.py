"""Script to compute
   relation embeddings for each relation in given list.
   entity embeddings for each entity in given list.
   word embeddings for each word in given word list.
   temporal fact embeddings for each fact in given fact list.
"""

import globals
from time_encoder import TimeEncoder
from util import load_dict
from wikipedia2vec import Wikipedia2Vec
import numpy as np
import pickle
from nltk.corpus import stopwords as SW
import json

from nltk.stem import PorterStemmer
PS = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
LC = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
WL = WordNetLemmatizer()
from nltk.stem import SnowballStemmer
SB = SnowballStemmer("english")
stopwords = set(SW.words("english"))
stopwords.add("'s")

word_to_relation = {}
relation_lens = {}
word_dim = 100
tem_dim = 100
min_date = {"year": -2000, "month": 1, "day": 1}
max_date = {"year": 7000, "month": 12, "day": 31}

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

def replace_symbols_in_entity(s):
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
    s = s.replace("'", ' ')
    s = s.replace(';', ' ')
    s = s.replace('/', ' ')
    s = s.replace(',', ' ')
    s = s.replace('-', ' ')
    s = s.replace('+', ' ')
    s = s.replace('.', ' ')
    s = s.strip('"@fr')
    s = s.strip('"@en')
    s = s.strip('"@cs')
    s = s.strip('"@de')
    s = s.strip()
    return s

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def relationids(relationids_file, relation2id):
    relationid2id = {}
    with open(relationids_file) as json_data:
        relids = json.load(json_data)
        for id in relids:
            relationid2id[id] = relation2id[relids[id]]

def _add_rel_word(word, rid):
    if word not in word_to_relation: word_to_relation[word] = []
    word_to_relation[word].append(rid)
    if rid not in relation_lens: relation_lens[rid] = 0
    relation_lens[rid] += 1

def relation_emb(relation2id, wiki2vec):
    reverse_relation2id = {}
    embedding_matrix = []
    no_in_word = []
    for key, value in relation2id.items():
        reverse_relation2id[str(value)] = key
    relation_emb = {r: np.zeros((word_dim,)) for r in reverse_relation2id}
    for i in range(len(reverse_relation2id)):
        i_str = str(i)
        relation = reverse_relation2id[i_str]
        relation = replace_symbols_in_relation(relation)
        relation_li = relation.split()
        c = 0
        for word in relation_li:
            key_li = [word, word.lower(), WL.lemmatize(word), PS.stem(word), LC.stem(word), SB.stem(word)]
            flag = 0
            for item in key_li:
                try:
                    value = wiki2vec.get_word_vector(item)
                    flag = 1
                    break
                except KeyError:
                    continue
                    # try:
                    #     value = wiki2vec.get_entity_vector(item)
                    # except KeyError:
                    #     continue
            if flag == 1:
                relation_emb[i_str] += value
                c += 1

        if c > 0:
            relation_emb[i_str] = relation_emb[i_str]/c
        else:
            no_in_word.append(word)
        embedding_matrix.append(relation_emb[i_str])

    embedding_matrix = np.asarray(embedding_matrix)
    print("\nword not in dictionary: ", str(len(set(no_in_word))))
    print("\nword not in dictionary: ", set(no_in_word))
    print("\nlen of relations: ", str(len(relation2id)))
    print("\nlen of relation_emb: ", str(len(relation_emb)))
    print("\nlen of embedding_matrix: ", str(len(embedding_matrix)))
    return relation_emb, embedding_matrix

def average_word_emb(word, wordli, wiki2vec, no_in_word):
    vocab_emb = np.zeros((word_dim,))
    flag = 0
    c = 0.0
    for item in wordli:
        try:
            value = wiki2vec.get_word_vector(item)
            flag = 1
            c += 1
        except KeyError:
            continue
        vocab_emb += value
    if flag == 0:
        no_in_word.append(word)
        return np.random.uniform(low=-0.5, high=0.5, size=(word_dim,))
    else:
        return vocab_emb/c

def get_entity_emb_not_in_embdic(wiki2vec, entity):
    ent_li = replace_symbols_in_entity(entity).split()
    ent_emb = np.zeros((word_dim,))
    c = 0.0
    flag = 0
    for item in ent_li:
        item_forms = [item, item.capitalize(), item.lower()]
        for it in item_forms:
            try:
                ent_emb += wiki2vec.get_entity_vector(item)
                c += 1.0
            except KeyError:
                try:
                    ent_emb += wiki2vec.get_word_vector(item)
                    c += 1.0
                except KeyError:
                    continue
            else:
                break
    if c > 0:
        ent_emb = ent_emb/c
        flag = 1
    if flag == 0:
        ent_emb = np.random.uniform(low=-0.5, high=0.5, size=(word_dim,))
    return flag, ent_emb

def get_upper(words):
    newwords = []
    for i in range(0,len(words)):
        if i ==0 :
            newwords.append(words[i].upper())
        elif words[i - 1] == ' ':
            newwords.append(words[i].upper())
        else:
            newwords.append(words[i])
    upper = ("").join(newwords)
    return upper

def average_ent_emb(ent, wiki2vec, no_in_entity):
    ent_forms = [ent, get_upper(ent), ent.capitalize(), ent.lower()]

    flag = 0
    for item in ent_forms:
        try:
            value = wiki2vec.get_entity_vector(item)
            flag = 1
        except KeyError:
            try:
                value = wiki2vec.get_word_vector(item)
                flag = 1
            except KeyError:
                continue
    if flag == 1:
        return value
    if flag == 0:
        ent_li = ent.split()
        c = 0
        value_word = np.zeros((word_dim,))
        for word in ent_li:
            key_li = [word, word.capitalize(), word.lower(), WL.lemmatize(word), PS.stem(word), LC.stem(word), SB.stem(word)]
            for item in key_li:
                try:
                    value_word += wiki2vec.get_word_vector(item)
                    c += 1
                except KeyError:
                    continue
        if c > 0:
            value_word = value_word / c
            return value_word
        else:
            no_in_entity.append(ent)
            return np.random.uniform(low=-0.5, high=0.5, size=(word_dim,))

def word_emb(word2id, wiki2vec):
    reverse_word2id = {}
    embedding_matrix = []
    no_in_word = []
    for key, value in word2id.items():
        reverse_word2id[str(value)] = key
    vocab_emb = {r: np.random.uniform(low=-0.5, high=0.5, size=(word_dim,)) for r in reverse_word2id}
    for i in range(len(reverse_word2id)):
        i_str = str(i)
        key = reverse_word2id[i_str]
        key_li = [key, key.capitalize(), key.lower(), WL.lemmatize(key), PS.stem(key), LC.stem(key), SB.stem(key)]
        for item in key_li:
            try:
                vocab_emb[i_str] = wiki2vec.get_word_vector(item)
            except KeyError:
                continue
            else:
                break
        else:
            if '-' in key or '/' in key:
                if '-' in key:
                    vocab_emb[i_str] = average_word_emb(key, key.split('-'), wiki2vec, no_in_word)
                else:
                    vocab_emb[i_str] = average_word_emb(key, key.split('/'), wiki2vec, no_in_word)
            else:
                no_in_word.append(key)

        embedding_matrix.append(vocab_emb[i_str])

    embedding_matrix = np.asarray(embedding_matrix)
    print ("\nword not in dictionary: ", str(len(no_in_word)))
    print ("\nlen of words: ", str(len(word2id)))
    print("\nlen of vocab_emb: ", str(len(vocab_emb)))
    print("\nlen of embedding_matrix: ", str(len(embedding_matrix)))
    return vocab_emb, embedding_matrix

def entity_emb(entity2id, entity_name_map, wiki2vec):
    print ("length of entity", str(len(entity2id)))
    print ("length of entity name", str(len(entity_name_map)))
    reverse_entity2id = {}
    embedding_matrix = []
    no_in_entity = []
    for key, value in entity2id.items():
        reverse_entity2id[str(value)] = entity_name_map[key]
    print ("length of reverse_entity2id", str(len(reverse_entity2id)))
    entity_emb = {r: np.zeros((word_dim,)) for r in reverse_entity2id}
    for i in range(len(reverse_entity2id)):
        i_str = str(i)
        entity_label = reverse_entity2id[i_str]
        entity_emb[i_str] = average_ent_emb(entity_label,wiki2vec,no_in_entity)
        embedding_matrix.append(entity_emb[i_str])
    embedding_matrix = np.asarray(embedding_matrix)
    print("length of entity not in embedding dic: ", str(len(set(no_in_entity))))
    print("len of entity_emb: ", str(len(entity_emb)))
    print("len of embedding_matrix: ", str(len(embedding_matrix)))
    return entity_emb, embedding_matrix

def _is_date(date, max_date, min_date):
    is_date = False
    date_range = dict()
    first = ''
    second = ''
    third = ''
    if date.startswith("-") and len(date.split("-")) == 4:
        first = '-' + date.split("-")[1]
        second = date.split("-")[2]
        third = date.split("-")[3]
    if not date.startswith("-") and len(date.split("-")) == 3:
        first = date.split("-")[0]
        second = date.split("-")[1]
        third = date.split("-")[2]
    if is_number(first) and is_number(second) and is_number(third):
        if int(first) <= max_date['year'] and int(first) >= min_date['year'] and int(second) <= 12 and int(
                    second) >= 1 and int(third) <= 31 and int(third) >= 1:
            is_date = True
            date_range['year'] = int(first)
            date_range['month'] = int(second)
            date_range['day'] = int(third)
        if int(third) <= max_date['year'] and int(third) >= min_date['year'] and int(first) <= 12 and int(
                    first) >= 1 and int(second) <= 31 and int(second) >= 1:
            is_date = True
            date_range['year'] = int(third)
            date_range['month'] = int(first)
            date_range['day'] = int(second)
        if int(third) <= max_date['year'] and int(third) >= min_date['year'] and int(second) <= 12 and int(
                    second) >= 1 and int(first) <= 31 and int(first) >= 1:
            is_date = True
            date_range['year'] = int(third)
            date_range['month'] = int(second)
            date_range['day'] = int(first)
    return is_date, date_range

def _get_tem_emb(date, time_encoder_mgr, max_date, min_date):
    date_flag, date_range = _is_date(date, max_date, min_date)
    if date_flag:
        e = time_encoder_mgr.get_time_encoding(date_range)
        if e is not None:
            return e.cpu().numpy()

def date_emb(entity2id, date2id, time_encoder_mgr, max_date, min_date):
    reverse_entity2id = {}
    embedding_matrix = []

    in_date = []
    for key, value in entity2id.items():
        # print (key, value)
        reverse_entity2id[str(value)] = key
    entity_tememb = {r: np.zeros((tem_dim,)) for r in reverse_entity2id}
    for i in range(len(reverse_entity2id)):
        i_str = str(i)
        entity_qid = reverse_entity2id[i_str]
        if entity_qid in date2id:
            try:
                tem_emb = _get_tem_emb(entity_qid, time_encoder_mgr, max_date, min_date)
                if tem_emb is not None:
                    entity_tememb[i_str] = tem_emb
                    in_date.append(entity_qid)
            except:
                print(entity_qid)
        else:
            entity_tememb[i_str] = np.random.uniform(low=-0.5, high=0.5, size=(tem_dim,))

        embedding_matrix.append(entity_tememb[i_str])
    embedding_matrix = np.asarray(embedding_matrix)
    print("entity in date: ", str(len(in_date)))
    print("len of entity_tememb: ", str(len(entity_tememb)))
    print("len of embedding_matrix: ", str(len(embedding_matrix)))
    return entity_tememb, embedding_matrix

def tempfact_emb(sorted_tkgfacts, entity2id, relation2id, entity_embeddings, relation_embeddings, date_embeddings):
    tempfact2id = {}
    for item in sorted_tkgfacts:
        tempfact2id[item] = len(tempfact2id)
    reverse_tempfact2id = {}
    te_embedding_matrix = []
    embedding_matrix = []
    for key, value in tempfact2id.items():
        reverse_tempfact2id[str(value)] = key
    relation_aver_emb_dim = 100
    entity_emb_dim = 100
    tempf_emb = {r: np.zeros((3 * entity_emb_dim + relation_aver_emb_dim,)) for r in reverse_tempfact2id}
    tempf_te_emb = {r: np.zeros((3 * entity_emb_dim + relation_aver_emb_dim + tem_dim,)) for r in reverse_tempfact2id}

    for i in range(len(reverse_tempfact2id)):
        i_str = str(i)
        tuple = reverse_tempfact2id[i_str]
        sub_emb = entity_embeddings[str(entity2id[tuple[0]])]
        rel_emb = relation_embeddings[str(relation2id[tuple[1]])]
        obj_emb = entity_embeddings[str(entity2id[tuple[2]])]
        trel_emb = relation_embeddings[str(relation2id[tuple[3]])]
        date_te_emb = date_embeddings[str(entity2id[tuple[4]])]
        date_ent_emb = entity_embeddings[str(entity2id[tuple[4]])]

        tempf_te_emb[i_str] = np.concatenate((sub_emb, (rel_emb + trel_emb) / 2 , obj_emb, date_ent_emb, date_te_emb), axis=0)
        te_embedding_matrix.append(tempf_te_emb[i_str])

        tempf_emb[i_str] = np.concatenate((sub_emb, (rel_emb + trel_emb) / 2 , obj_emb, date_ent_emb), axis=0)
        embedding_matrix.append(tempf_emb[i_str])

    te_embedding_matrix = np.asarray(te_embedding_matrix)
    embedding_matrix = np.asarray(embedding_matrix)
    print("len of te embedding_matrix: ", str(len(te_embedding_matrix)))
    print("len of non-te embedding_matrix: ", str(len(embedding_matrix)))
    return tempf_te_emb, te_embedding_matrix, tempf_emb, embedding_matrix, tempfact2id

def load_dict(filename):
    word2id = dict()
    with open(filename) as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    print(len(word2id))
    return word2id

def get_pretrained_embedding_from_wiki2vec(MODEL_FILE, answer_predict_path):
    wiki2vec = Wikipedia2Vec.load(MODEL_FILE)
    fp_entity = answer_predict_path + 'entities.txt'
    fp_entity_name_map = answer_predict_path + 'entity_name_map.pkl'
    fp_relation = answer_predict_path + 'relations.txt'
    fp_word = answer_predict_path + 'words.txt'
    fp_date = answer_predict_path + 'dates.txt'
    fp_tempfact_pkl = answer_predict_path + 'tempfacts.pkl'
    fp_tempfact2id_pkl =  answer_predict_path + 'tempfacts2id.pkl'
    relation_emb_file = answer_predict_path + 'relation_emb_100d.npy'
    vocab_emb_file = answer_predict_path + 'word_emb_100d.npy'
    entity_emb_file = answer_predict_path + 'entity_emb_100d.npy'
    entity_date_emb_file = answer_predict_path + 'date_te_emb_100d.npy'
    tempfact_te_emb_file = answer_predict_path + 'tempfact_te_emb_500d.npy'
    tempfact_emb_file = answer_predict_path + 'tempfact_emb_400d.npy'

    sorted_tkgfacts = pickle.load(open(fp_tempfact_pkl, 'rb'))
    min = sorted_tkgfacts[0][4]
    max = sorted_tkgfacts[len(sorted_tkgfacts)-1][4]
    print ("\nmin_date: ", min)
    print ("\nmax_date: ", max)

    time_encoder_mgr = TimeEncoder(tem_dim, 0.1, span=1, min_date=min_date, max_date=max_date)

    word2id = load_dict(fp_word)
    entity2id = load_dict(fp_entity)
    date2id = load_dict(fp_date)
    relation2id = load_dict(fp_relation)
    entity_name_map = pickle.load(open(fp_entity_name_map, 'rb'))
    # relation encoding, get pretrained_relation_emb_file
    print('Embedding Relations....')
    relation_embeddings, relation_embedding_matrix = relation_emb(relation2id, wiki2vec)
    print('Embedding Words....')
    vocab_embeddings, vocab_embedding_matrix = word_emb(word2id, wiki2vec)
    print('Embedding Entities...')
    entity_embeddings, entity_embedding_matrix = entity_emb(entity2id, entity_name_map, wiki2vec)
    print('Time encoding for Dates...')
    date_embeddings, date_embedding_matrix = date_emb(entity2id, date2id, time_encoder_mgr, max_date, min_date)
    print('Embedding Tempfacts...')
    tempf_te_emb, tempf_te_embedding_matrix, tempf_emb, tempf_embedding_matrix, tempfact2id = tempfact_emb(sorted_tkgfacts, entity2id, relation2id, entity_embeddings, relation_embeddings, date_embeddings)

    pickle.dump(tempfact2id, open(fp_tempfact2id_pkl, 'wb'))
    print('Saving Relations....')
    np.save(relation_emb_file, relation_embedding_matrix)
    print('Saving Vocabs....')
    np.save(vocab_emb_file, vocab_embedding_matrix)
    print('Saving Entities....')
    np.save(entity_emb_file, entity_embedding_matrix)
    print('Saving Time encoding for Dates....')
    np.save(entity_date_emb_file, date_embedding_matrix)
    print('Saving Time encoding for Tempfacts....')
    np.save(tempfact_te_emb_file, tempf_te_embedding_matrix)
    print('Saving Tempfacts....')
    np.save(tempfact_emb_file, tempf_embedding_matrix)

if __name__ == "__main__":
    cfg = globals.get_config(globals.config_file)
    answer_predict_path = cfg['answer_predict_path']
    MODEL_FILE = cfg['model_path'] + cfg['wikipedia2vec']
    get_pretrained_embedding_from_wiki2vec(MODEL_FILE, answer_predict_path)
