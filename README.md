#EXAQT
This is the implementation of EXAQT described in CIKM 2021 paper Complex Temporal Question Answering on Knowledge Graphs.

- TimeQuestions: 
    - 16,181 questions, each containing:
        - "Id": question id
        - "Question": question text in lowercase
        - "Temporal signal": temporal signals including OVERLAP, AFTER, BEFORE, START, FINISH, ORDINAL, No signal
        - "Type": temporal categories including Explicit, Implicit, Ordinal, Temp.Ans
        - "Answer": ground truth answer including answer type, Wikidata Qid,  Wikidata label, Wikipedia URL
        - "Data source": original dataset
        - "Question creation date": original dataset publication date 
    - Train, dev, and test datasets are in the 60:20:20 ratio.
        - train.json
        - dev.json
        - test.json
        
- NERD 
    - `get_seed_entity_elq.py` to get seed entity from ELQ and entity linking results.
    - To run the program, please find the usage of ELQ from the following [link](https://github.com/facebookresearch/BLINK/tree/master/elq), and download the pretrained models, indices, and entity embeddings for running ELQ.
    - `get_seed_entity_tagme.py` to get seed entity from TagMe and entity linking results.
    
- Constructing answer graph
  - Finding connectivity between seed entities in questions.
    - We provide the shortest paths between seed entities.
  - Finding the KG facts for seed entities.
    - We provide the KG facts for each question in the benchmark.
  - Computing compact subgraph
    - `relevant_fact_selection_model.py` to score facts using pre-trained fine-tune BERT model.
	- `get_compact_subgraph.py` to compute GST and complete it.
	- `get_GST.py` is a collection of functions for GST algorithm.
  - Augmenting subgraphs with temporal facts
    - We provide the temporal facts for each compact subgraph (top-f=25, top-g=25).
    - `temporal_fact_selection_model.py` to score temporal facts using pre-trained fine-tune BERT model.
  - Please download the fine-tuning BERT models in the two phases in the [link](https://www.dropbox.com/home/exaqt/data).
	     
- Predicting answers with R-GCN
	- `step1_get_relational_graph.py` to generate relational graph.
	- `step2_get_dictionary.py` to create dictionaries including words, entities, relations, categories, signals, temporal facts, etc. 
	- `step3_get_pretrained_embedding.py` to generate pretrained embeddings for words, entities, relations, temporal facts, etc.
    - `step4_train_rgcn.py` to train R-GCN model and evaluate the model on test dataset.
    - `time_encoder.py` is a positional time encoding function.
	- `util.py` is a collection of frequent functions in answer prediction with R-GCN.
	- `data_loader.py` is a data loader function in answer prediction.
	- `model.py` is a R-GCN model class in answer prediction.
	- `script_listscore.py` is a collection of evaluation functions. 

- Other programs
    - globals.py to generate global configuration variables.

First download the data from the following [link](https://www.dropbox.com/home/exaqt/data).

Make sure the path variables in the .yml file under your own settings.
