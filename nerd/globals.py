import yaml
import json

"""
Entity (item/property)
 wd:Q* 
       --> rdfs:label, skos:altLabel, schema:description "*"@*
       --> schema:dateModified, schema:version
       --> wdt:P* "*", URI, _:blank
       --> p:P* Statement

Item
 wd:Q* <-- schema:about <http://*.wikipedia.org/wiki/*>
                          --> schema:inLanguage, wikibase:badge

Property
 wd:P* --> wikibase:propertyType PropertyType
       --> wkibase:directClaim        wdt:P*
       --> wikibase:claim             p:P*
       --> wikibase:statementProperty ps:P*
       --> wikibase:statementValue    psv:P*
       --> wikibase:qualifier         pq:P*
       --> wikibase:qualifierValue    pqv:P*
       --> wikibase:reference         pr:P*
       --> wikibase:referenceValue    prv:P*
       --> wikibase:novalue           wdno:P*

PropertyType
 wikibase: String, Url, WikibaseItem, WikibaseProperty, CommonsMedia, Math,
           Monolingualtext, GlobeCoordinate, Quantity, Time, ExternalId


Statement
 wds:* --> wikibase:rank Rank
       --> a wdno:P*
       --> ps:P* "*", URI, _:blank
       --> psv:P* Value
       --> pq:P* "*", URI, _:blank
       --> pqv:P* Value
       --> prov:wasDerivedFrom Reference
"""
URL = "https://query.wikidata.org/sparql"
WDT = "http://www.wikidata.org/prop/direct/"
P = "http://www.wikidata.org/prop/"
WD = "http://www.wikidata.org/entity/"
PS = "http://www.wikidata.org/prop/statement/"
PQ = "http://www.wikidata.org/prop/qualifier/"
WDS = "http://www.wikidata.org/entity/statement/"
schema = "http://schema.org/"
rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

WIKIPEDIA_EN = "https://en.wikipedia.org/wiki/"

PROI = "http://wikiba.se/ontology#WikibaseItem"
PROQ = "http://wikiba.se/ontology#Quantity"
PROT = "http://wikiba.se/ontology#Time"

config_file = '../config/config.yml'

def get_config(config_path):
    """Read configuration and set variables.

        :return:
        """
    global config
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config

def read_property(property_file):
    property = {}
    with open(property_file) as json_data:
        datalist = json.load(json_data)
    json_data.close()
    for item in datalist:
        pro = {}
        qid = item['property']['value'].replace("http://www.wikidata.org/entity/","")
        pro["label"] = item['propertyLabel']['value']
        pro["type"] = item['propertyType']['value']
        if "propertyAltLabel" in item:
            pro["altLabel"] = item['propertyAltLabel']['value']
        else:
            pro["altLabel"] = ""
        property[qid] = pro
    return property

class ReadProperty():
    def __init__(self, property):
        self.property = property

    @staticmethod
    def init_from_config():
        property_file = config['property_file']
        property = read_property(property_file)
        print("Property Dictionary loaded, length ", len(property))
        return ReadProperty(property)

