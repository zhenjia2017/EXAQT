EXAQT + TimeQuestions
============

## Update

${\color{red}Note}$: We provided a cleaner and complete implementation [here](https://github.com/zhenjia2017/EXAQTV2). Hope it helps you to reproduce the result.

Description
------
This repository contains the code and data for our CIKM'21 full paper. In this paper, we present EXAQT, the first end-to-end system for answering complex temporal questions that have multiple entities and predicates, and associated temporal conditions. EXAQT answers natural language questions over KGs in two stages. The first step computes question-relevant compact subgraphs within the KG, and judiciously enhances them with pertinent temporal facts, using Group Steiner Trees and fine-tuned BERT models. The second step constructs relational graph convolutional networks (R-GCNs) from the first step’s output, and enhances the R-GCNs with time-aware
entity embeddings and attention over temporal relations. 

<center><img src="kg.png"  alt="kg" width=80%  /></center>

*Wikidata excerpt showing the relevant KG zone
for the question "where did obama’s children study when he
became president?" with answer Sidwell Friends School.*

For more details see our paper: [Complex Temporal Question Answering on Knowledge Graphs](https://arxiv.org/abs/2109.08935) and visit our project website: https://exaqt.mpi-inf.mpg.de.

If you use this code, please cite:
```bibtex
@article{jia2021complex,
  title={Complex Temporal Question Answering on Knowledge Graphs},
  author={Jia, Zhen and Pramanik, Soumajit and Roy, Rishiraj Saha and Weikum, Gerhard},
  journal={arXiv preprint arXiv:2109.08935},
  year={2021}
}
```

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

TimeQuestions
------
The benchmark can be downloaded from [here](https://qa.mpi-inf.mpg.de/exaqt/TimeQuestions.zip). TimeQuestions contains 16,181 questions. The content of each question includes:
        
 * "Id": question id
 * "Question": question text in lowercase
 * "Temporal signal": OVERLAP, AFTER, BEFORE, START, FINISH, ORDINAL, No signal
 * "Temporal question type": temporal categories including Explicit, Implicit, Ordinal, Temp.Ans
 * "Answer": ground truth answer including answer type, Wikidata Qid,  Wikidata label, and Wikipedia URL
 * "Data source": original dataset
 * "Question creation date": original dataset publication date

Data
------
The preprocessed Wikidata facts for each question, pretrained models, all required intermediate data and our main results can be downloaded from [here](https://qa.mpi-inf.mpg.de/exaqt/exaqt-supp-data.zip) (unzip and put it in the root folder of the cloned github repo; total data size around 40 GB).

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
    ├── phase1_model.bin
    ├── phase2_model.bin
    └── wikipedia2vec_trained
├── result
├── temcompactgst
    ├── train_25_25_temp.json
    ├── train_25_25_temp_rank.pkl
    ├── dev_25_25_temp.json
    ├── dev_25_25_temp_rank.pkl
    ├── test_25_25_temp.json
    └── test_25_25_temp_rank.pkl
├── TimeQuestions
├── phase1_relevant_fact_selection_trainset.csv
├── phase2_temporal_fact_selection_trainset.csv
└── wikidata_property_dictionary.json
```

 - ./compactgst: completed GST subgraph for each question
 - ./connectivity: preprocessed connectivity data including shortest connect path of seed pairs for each question
 - ./dictionaries: relational graphs, dictionaries, and pretrained embedding files used in answer prediction
 - ./files: preprocessed Wikidata facts and intermediate data for each question including seed entity, scored facts, quasi question graph, cornerstones and gst graph files  
 - ./model: pretrained fine-tune BERT models and wikipedia2vec model
 - ./result: answer prediction evaluation results on test data set
 - ./temcompactgst: completed GST subgraphs with temporal facts and ranked temporal facts files 
 - ./phase1\_relevant\_fact\_selection\_trainset.csv: training data of fine-tuning BERT model for finding question-relevant KG facts 
 - ./phase2\_temporal\_fact\_selection\_trainset.csv: training data of fine-tuning BERT model for finding question-relevance of temporal facts
 - ./wikidata\_property\_dictionary.json: Wikidata properties with type, label and alias


Code
------
 
The code structure is as follows:
    
- NERD 
    - `get_seed_entity_elq.py` to get seed entity from ELQ and entity linking results.
    - To run the program, please find the usage of [ELQ](https://github.com/facebookresearch/BLINK/tree/master/elq), and download the pretrained models, indices, and entity embeddings for running ELQ.
    - `get_seed_entity_tagme.py` to get seed entity from TagMe and entity linking results.
    
- Answer Graph
	- Apply the fine-tuned BERT model as classifer
    	- `relevant_fact_selection_model.py` to score facts and sort them in descending order of a question relevance likelihood.
    	- `engine.py` to fine tune BERT model.
    
	- Compute compact subgraph
		- `get_compact_subgraph.py` to compute GST and complete it.
		- `get_GST.py` is a collection of functions for GST algorithm.
	
	- Augment subgraphs with temporal facts
    	- `temporal_fact_selection_model.py` to score temporal facts and sort them in descending order of a question relevance likelihood.
	     
- Answer Predict
	- `get_relational_graph.py` to create relational graphs.
	- `get_dictionary.py` to generate dictionaries including words, entities, relations, categories, signals, temporal facts, etc. 
	- `get_pretrained_embedding.py` to generate pretrained embeddings for words, entities, relations, temporal facts, etc.
    - `train_eva_rgcn.py` to train R-GCN model and evaluate the model.
    - `model.py` is a R-GCN model class in answer prediction.
    - `time_encoder.py` is a positional time encoding function.
	- `util.py` is a collection of frequent functions in answer prediction with R-GCN.
	- `data_loader.py` is a data loader function in answer prediction.
	- `script_listscore.py` is a collection of evaluation functions. 
	- `evaluate.py` to evaluate the model on test dataset.
- Other programs
    - `globals.py` to generate global configuration variables.

Graph construction and answer prediction
------
To reproduce the result, (1) download data and pre-trained model, and save them in the root folder of the cloned github repo, and (2) make sure the path variables in the config.yml file under your own settings.

Then run the following commands:

Step 1: NERD

    python get_seed_entity_elq.py (Note that the program should run under directory of BLINK-master after building ELQ environment)
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

<!-- ### Contributors
If you use this code, please cite:
Zhen Jia, Soumajit Pramanik, Rishiraj Saha Roy, Gerhard Weikum. (2021). Complex Temporal Question Answering on Knowledge Graphs. CIKM.
@article{jia2021complex,
  title={Complex Temporal Question Answering on Knowledge Graphs},
  author={Jia, Zhen and Pramanik, Soumajit and Roy, Rishiraj Saha and Weikum, Gerhard},
  journal={arXiv preprint arXiv:2109.08935},
  year={2021}
} -->
