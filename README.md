EXAQT + TimeQuestions
============

This is the implementation of EXAQT described in CIKM 2021 paper Complex Temporal Question Answering on Knowledge Graphs.

Setup 
------

The following software is required:

* Python 3.8

* Networkx 2.5

* Numpy 1.19.2

* Wikipedia2vec 1.0.5

* Nltk 3.5

* Transformers 4.4.2

* Torch 1.8.1

* Pandas 1.1.3

* Tqdm 4.50.2

* Requests 2.24.0

* PyYAML 5.4.1

To install the required libraries, it is recommended to create a virtual environment:

    python3 -m venv ENV_exaqt
    source ENV_exaqt/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt



Data
------
The benchmark, preprocessed Wikidata facts for each question, all required intermediate data and our main results can be downloaded from [here](https://exaqt.mpi-inf.mpg.de/static/data.zip) (unzip and put it in the root folder of the cloned github repo; total data size around 40 GB).

The data folder structure is as follows:


```
data
├── compactgst
    ├── train_25_25.json
    ├── dev_25_25.json
    └── test_25_25.json  
├── connectivity
    └── seedpair_question_best_connectivity_paths_score.pkl    
├── dictionaries
├── files
    ├── ques_1
    ├── ques_2
    ├── ...
    └── ques_16181
├── model
├── temcompactgst
    ├── train_25_25_temp.json
    ├── train_25_25_temp_rank
    ├── dev_25_25_temp.json
    ├── dev_25_25_temp_rank
    ├── test_25_25_temp.json
    └── test_25_25_temp_rank
├── TimeQuestions
    ├── train.json
    ├── dev.json
    └── test.json
└── wikidata_property_dictionary.json
```

 - ./compactgst: completed GST subgraph for each question
 - ./connectivity: preprocessed connectivity data including shortest connect path of seed pairs for each question
 - ./dictionaries: relational graphs, dictionaries and pretrained embedding files used in answer prediction
 - ./files: preprocessed Wikidata facts and intermediate data for each question including seed entity, scored facts, quasi question graph, cornerstones and gst graph files  
 - ./model: pretrained fine-tune BERT models and wikipedia2vec model
 - ./temcompactgst: completed GST subgraphs with temporal facts and ranked temporal facts files
 - ./TimeQuestions: benchmark including train, dev and test 
 - ./wikidata\_property\_dictionary.json: Wikidata properties with type, label and alias


Code
------
 
The code structure is as follows:
    
- NERD for question entities 
    - `get_seed_entity_elq.py` to get seed entity from ELQ and entity linking results.
    - To run the program, please find the usage of [ELQ](https://github.com/facebookresearch/BLINK/tree/master/elq), and download the pretrained models, indices, and entity embeddings for running ELQ.
    - `get_seed_entity_tagme.py` to get seed entity from TagMe and entity linking results.
    
- Apply the fine-tuned BERT model as classifer
    - `relevant_fact_selection_model.py` to score facts and sort them in descending order of a question relevance likelihood.
    - `engine.py` to fine tune BERT model.
    
- Compute compact subgraph
	- `get_compact_subgraph.py` to compute GST and complete it.
	- `get_GST.py` is a collection of functions for GST algorithm.
	
- Augment subgraphs with temporal facts
    - `temporal_fact_selection_model.py` to score temporal facts and sort them in descending order of a question relevance likelihood.
	     
- Predict answers with R-GCN
	- `get_relational_graph.py` to create relational graphs.
	- `get_dictionary.py` to generate dictionaries including words, entities, relations, categories, signals, temporal facts, etc. 
	- `get_pretrained_embedding.py` to generate pretrained embeddings for words, entities, relations, temporal facts, etc.
    - `train_eva_rgcn.py` to train R-GCN model and evaluate the model on test dataset.
    - `model.py` is a R-GCN model class in answer prediction.
    - `time_encoder.py` is a positional time encoding function.
	- `util.py` is a collection of frequent functions in answer prediction with R-GCN.
	- `data_loader.py` is a data loader function in answer prediction.
	- `script_listscore.py` is a collection of evaluation functions. 

- Other programs
    - `globals.py` to generate global configuration variables.

Graph construction and answer prediction
------
To reproduce the result, (1) download data and pre-trained model, and save them in the root folder of the cloned github repo, and (2) make sure the path variables in the config.yml file under your own settings.

Then run the following commands:

Step 1: NERD

    python get_seed_entity_elq.py
    python get_seed_entity_tagme.py

Step 2: Score and rank question-relevance facts

    python relevant_fact_selection_model.py -d train 
    python relevant_fact_selection_model.py -d dev
    python relevant_fact_selection_model.py -d test

Step 3: Compute compact subgraph

    python get_compact_subgraph.py -d train
    python get_compact_subgraph.py -d dev
    python get_compact_subgraph.py -d test

Step 4: Score and rank question-relevance temporal facts

    python temporal_fact_selection_model.py -d train
    python temporal_fact_selection_model.py -d dev
    python temporal_fact_selection_model.py -d test

Step 5: Train answer prediction model and evaluate on test

    python get_relational_graph.py -d train
    python get_relational_graph.py -d dev
    python get_relational_graph.py -d test
    python get_dictionary.py
    python get_pretrained_embedding.py
    python train_eva_rgcn.py -p exaqt

### Contributors
If you use this code, please cite:
Zhen Jia, Soumajit Pramanik, Rishiraj Saha Roy, Gerhard Weikum. (2021). Complex Temporal Question Answering on Knowledge Graphs. CIKM.
```

```