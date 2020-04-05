# Lexical Symmetry
Classifying lexical symmetry in verbs using machine learning techniques.
This script uses a Logistic Regression classifier with a customized dataset (from English Web Corpus 2015) and pre-trained word embeddings word2vec (Google News Corpus) and GloVe (Wikipedia 2014). 

Scripts:
- analyses (.ipynb): main script that contains the classifiers and their subsequent evaluation 

Note: the word2vec embeddings are too large to upload here. Please clone GoogleNews-vectors-negative300.bin.gz from https://github.com/mmihaltz/word2vec-GoogleNews-vectors
- analyses_helper_functions (.py): contains helper functions to run analyses
  
Data:
- verbcorp (.csv): contains sentences and annotated feature sets used in this study
- LF_errors (.csv): LF model's error sentences
- w2v_errors (.csv): word2vec model's error sentences
- glove_errors (.csv): GloVe model's error sentences

