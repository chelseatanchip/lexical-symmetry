#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Chelsea Tanchip
# Helper functions for Lexical Symmetry Project

import numpy as np 
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.model_selection import LeaveOneOut
from scipy import spatial

import gensim
import nltk
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer 

def cv(est,data,labels):

    """
    Compute 10-fold cross-validation metrics.

    Parameters
    ----------
    est : classifier
        The classifier.

     data : array of arrays
        Feature vector space.

    labels : float array
        The labeled categories.

    Returns
    -------
    results : dictionary
        Accuracy, precision, recall, and f1 score are returned as a dictionary.


    """ 

    scoring = {'accuracy' : make_scorer(accuracy_score), 'precision' : make_scorer(precision_score), 'recall' : make_scorer(recall_score), 'f1_score' : make_scorer(f1_score)}

    kfold = KFold(n_splits=10, random_state=42)

    results = cross_validate(estimator=est, X=data, y=labels, cv=kfold, scoring=scoring)

    return results



def prototype_model(F):

    """
    Returns the prototype feature vector for category.

    Parameters
    ----------
    
    F : float array
        Feature vector

    Returns
    -------
    
    results : float array
        Prototype feature vector computed from the mean of features.


    """
    ft_vector = np.zeros((len(F[0]),1))

    s = 0
    for i in range(len(F[0])):
        ft_vector[s] = np.mean(F[:,i])
        s=s+1
    return ft_vector


def loo_features(sentences, verbs, features, labels):
    """
    Returns the prototype feature vector for category.

    Parameters
    ---------- 
    
    sentences : string array
        List of sentences
        
    verbs : string array
        List of verbs
        
    features : float array
        Feature vector

    labels : float array
        Category vector

    Returns
    ------- 
    
    sent_errors : str array
        List of miscategorized sentences.

    pred_errors : int array
        Indices of sentences that were misclassified.


    """

    loo = LeaveOneOut()
    loo.get_n_splits(features)

    # Store predictive accuracy and errors
    pro_predict = np.zeros(len(features)) 
    pro_err = [] 
    sent_errors = []
    pred_errors = []

    # Count predictive accuracy and errors
    o = 0 
    count_p = 0 

    for train_index, test_index in loo.split(features):

    #     1) Get training and testing data points
        X_train, X_test = features[train_index], features[test_index]

        asym = np.where(labels==0)
        sym = np.where(labels==1)

        asym=asym[0]
        sym=sym[0]

        # Avoid out-of-bound errors
        if max(asym) >= len(X_train):
            asym_train = X_train[asym[0:len(asym)-1]]
        else:
            asym_train = X_train[asym]
        if max(sym) >= len(X_train):
            sym_train = X_train[sym[0:len(sym)-1]]
        else:
            sym_train = X_train[sym]


    #     2.1) Implement and predict with prototype model

        # Feature vectors for symmetry categories
        asym_prototype = prototype_model(asym_train)
        sym_prototype = prototype_model(sym_train)

        # Predicting label for test category
        a = spatial.distance.euclidean(X_test,asym_prototype)
        b = spatial.distance.euclidean(X_test,sym_prototype)

        if a < b:
            pro_predict[o] = 0
            if pro_predict[o]==labels[o]:
                # count number of correctly predicted instances
                count_p=count_p+1
            else:
                # else record the incorrect prediction
                pro_err.append(o)

        else:
            pro_predict[o] = 1
            if pro_predict[o]==labels[o]:
                count_p=count_p+1
            else:
                pro_err.append(o)

        o=o+1

    # 3) Report predictive accuracies (%)
    pro_acc = '{0:.3g}'.format((count_p/len(features))*100)

    print("\nAccuracy of prototype model:", pro_acc, "%")

    # 4) Error analysis
    print("\nNumber of errors:",len(pro_err))
    #print("\nPrototype model errors occurred with: \n")

    for x in pro_err:
        sent_errors.append(sentences[x])
        pred_errors.append(int(pro_predict[x]))
        #print(sentences[x], ". Category:", int(pro_predict[x]))


    print("\nMost frequent misclassifications:\n")
    print(get_error_frequency(verbs, pro_err))

    return sent_errors, pred_errors


def get_error_frequency(verbs,pro_err):
    """
    Returns the most frequently misclassified verbs.

    Parameters
    ----------
    
    verbs : string array
        List of verbs
        
    pro_err : float array
        Feature vector

    Returns
    -------

    sorted_d : list of tuples
        Most frequently misclassified verbs sorted in descending order.


    """

    # Dictionary of misclassified verbs and their frequency
    freq = {}
    for i in pro_err:
        v = verbs[i]
        if v not in freq:
            freq[v] = 1
        else:
            freq[v] += 1

    F = freq.items()
    sorted_d = [] 
  

    # Create a list of tuples sorted in descending order
    F_list = sorted(F, reverse=True, key=lambda x: x[1])

    # Iterate over the sorted sequence
    for i in F_list:
        sorted_d.append(i)

    return sorted_d


### Code for word embedding models adapted from RaRe Technologies
# https://github.com/RaRe-Technologies/movie-plots-by-genre
    
def tokenize_text(sentences):
    """

    Returns tokenized text for word embedding models.

    Parameters
    ----------
    
    sentences : string array
        List of sentences to be tokenized

    Returns
    -------
    
     sents : string array
         List of tokenized sentences
 
    """

    sents=[]

    for sent in sentences:
        sent=nltk.sent_tokenize(sent)
        tokens = []
        for subsent in sent:
            for word in nltk.word_tokenize(subsent):
                tokens.append(word.lower())

        sents.append(" ".join(tokens))

    return sents


def apply_weight(sents):  
      
    """
    Returns TF-IDF weighted words.

    Parameters
    ----------
    
    sents : string array
        List of sentences
 

    Returns
    -------

    word_idf_weight : float array
         TF-IDF weighting of words
    """
    
    word_idf_weight = []  
    tfidf = TfidfVectorizer()
    tfidf.fit(sents)  
    max_idf = max(tfidf.idf_)
    word_idf_weight = defaultdict(lambda: max_idf, [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])
    return word_idf_weight

def word_avg_w2v(wv, words, word_idf_weight):
    """
    Returns sentence average with TD-IDF weighting (word2vec).

    Parameters
    ----------
    
    wv : model
        word2vec

    words: string
        Sentence input

    Returns
    -------

     mean : list of tuples
         Weighted average vector
    """

    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0[wv.vocab[word].index] * word_idf_weight[word])
            all_words.add(wv.vocab[word].index)

    if not mean:
        return np.zeros(wv.layer1_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_avg_glove(gv, words, word_idf_weight):
    """
    Returns sentence average with TD-IDF weighting (GloVe).

    Parameters
    ----------
    
    gv : model
        GloVe

    words: string
        Sentence input

    Returns
    -------

     mean : list of tuples
         Weighted average vector


    """

    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in gv.vocab:
            mean.append(gv.syn0[gv.vocab[word].index] * word_idf_weight[word])
            all_words.add(gv.vocab[word].index)

    if not mean:
        return np.zeros(gv.layer1_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_avg_list(wv, sentences, avg_method):
    """
    Returns word averages for each sentence in the corpus.

    Parameters
    ----------
    
    wv : model
    Either word2vec or GloVe model

    sentences: string array
        Tokenized sentences

    Returns
    -------

     mean : float array
         Array of averages


    """
    
    tf = apply_weight(sentences)
    
    if avg_method==1:
        return np.vstack([word_avg_w2v(wv, s, tf) for s in sentences]) #use word2vec averaging
    else:
        return np.vstack([word_avg_glove(wv, s, tf) for s in sentences]) #use glove averaging