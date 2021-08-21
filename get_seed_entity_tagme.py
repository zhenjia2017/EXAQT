"""Script to get seed entities from TagMe.
Output will be a seed entity file and a wikidata qid file:
The format of the seed entities is:
Tagme:
tagme_ent['spot']: list of ('spot', 'wiki_title', 'wiki_id', 'rho', 'start', 'end')
tagme_ent['wikidata']: list of (wikidata_id, link[1])
"""

import pickle
import json
import requests
import truecase
import os
import globals

MY_GCUBE_TOKEN = 'f08bb655-6465-4cdb-a5b2-7b7195cea1d7-843339462'

def get_wikipedialink(pageid):
    info_url = "https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids=" + pageid + "&inprop=url&format=json"
    try:
        response = requests.get(info_url)
        result = response.json()["query"]["pages"]
        if result:
            link = result[pageid]['fullurl']
            return link
    except:
        print("get_wikipedialink problem", pageid)

def get_qid(wikipedia_link):
	url = "https://tools.wmflabs.org/openrefine-wikidata/en/api?query=" + wikipedia_link
	try:
		response = requests.get (url)
		results = response.json ()["result"]
		if results:
			qid = results[0]['id']
			return qid
	except:
		print ("get_qid problem", wikipedia_link)

def wat_entity_linking(text):
    # Main method, text annotation with WAT entity linking system
    wat_url = 'https://wat.d4science.org/wat/tag/tag'
    payload = [("gcube-token", MY_GCUBE_TOKEN),
               ("text", text),
               ("lang", 'en'),
               ("tokenizer", "nlp4j"),
               ('debug', 9),
               ("method",
                "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,voting:relatedness=lm,ranker:model=0046.model,confidence:model=pruner-wiki.linear")]
    try:
        response = requests.get(wat_url, params=payload)
        wat_annotations = [WATAnnotation(a) for a in response.json()['annotations']]
        return [w.json_dict() for w in wat_annotations]
    except:
        print("here is a timeout error!")
        return None

class WATAnnotation:
    # An entity annotated by WAT
    def __init__(self, d):

        # char offset (included)
        self.start = d['start']
        # char offset (not included)
        self.end = d['end']

        # annotation accuracy
        self.rho = d['rho']
        # spot-entity probability
        self.prior_prob = d['explanation']['prior_explanation']['entity_mention_probability']
        # annotated text
        self.spot = d['spot']
        # Wikpedia entity info
        self.wiki_id = d['id']
        self.wiki_title = d['title']

    def json_dict(self):
        # Simple dictionary representation
        return {'wiki_title': self.wiki_title,
                'wiki_id': self.wiki_id,
                'start': self.start,
                'end': self.end,
                'rho': self.rho,
                'prior_prob': self.prior_prob
                }

class EntityLinkTagMeMatch():
    def __init__(self, tagme_threshold=0):
        self.tagme_threshold = tagme_threshold

    def get_seed_entities_tagme(self, ques_truecase):
        tagme_ent = self.get_response_wat(ques_truecase)
        tagme_ent['wikidata'] = []
        for link in tagme_ent['spot']:
            pageid = link[2]
            wikipedia_link = get_wikipedialink(pageid)
            if wikipedia_link:
                wikidata_id = get_qid(wikipedia_link)
                if wikidata_id:
                    tagme_ent['wikidata'].append((wikidata_id, link[1]))
        return tagme_ent

    def get_response_wat(self, ques):
        tagme_ent = {}
        tagme_ent['spot'] = []
        try:
            annotations = wat_entity_linking(ques)
            # print (annotations)
            if annotations:
                for doc in annotations:
                    if doc['rho'] >= self.tagme_threshold:
                        doc['spot'] = ques[doc["start"]:doc["end"]]
                        tagme_ent['spot'].append(
                            (doc['spot'], doc['wiki_title'], str(doc['wiki_id']), doc['rho'], doc['start'], doc['end']))
        except:
            print("TAGME Problem \n", ques)
        return tagme_ent

def get_seed_entities_tagme(TAGME, path, id, question):
    tagme_file = path + '/tagme'
    wiki_ids_file = path + '/wiki_ids_tagme.txt'
    wiki_ids = set()
    tagme_ent = TAGME.get_seed_entities_tagme(question)

    if 'wikidata' in tagme_ent and 'spot' in tagme_ent:
        for id1 in tagme_ent['wikidata']:
            index = tagme_ent['wikidata'].index(id1)
            text = tagme_ent['spot'][index][0].lower()
            score = float(tagme_ent['spot'][index][3])
            wiki_ids.add((id1[0], score, text))

    f1 = open(tagme_file, 'wb')
    f3 = open(wiki_ids_file, 'w', encoding='utf-8')
    pickle.dump(tagme_ent, f1)
    for item in wiki_ids:
        f3.write(str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\n')
    f1.close()
    f3.close()

if __name__ == "__main__":
    # prepare data...
    print("\n\nPrepare data and start...")
    cfg = globals.get_config(globals.config_file)
    test = cfg["data_path"] + cfg["test_data"]
    dev = cfg["data_path"] + cfg["dev_data"]
    train = cfg["data_path"] + cfg["train_data"]
    os.makedirs(cfg["ques_path"], exist_ok=True)
    tagme_threshold = 0
    TAGME = EntityLinkTagMeMatch(tagme_threshold)
    in_files = [train, dev, test]
    for fil in in_files:
        data = json.load(open(fil))
        for question in data:
            QuestionId = str(question["Id"])
            QuestionText = question["Question"]
            QuestionText = truecase.get_true_case(QuestionText)
            path = cfg["ques_path"] + 'ques_' + str(QuestionId)
            os.makedirs(path, exist_ok=True)
            get_seed_entities_tagme(TAGME, path, QuestionId, QuestionText)
