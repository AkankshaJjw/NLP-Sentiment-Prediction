#!/usr/bin/env python
# coding: utf-8

# In[14]:


import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('data1.csv', encoding="ansi")
df = df[pd.notnull(df['tag'])]
print(df.head(10))
print(df['utterance'].apply(lambda x: len(x.split(' '))).sum())


# In[15]:


file = "C:\\Users\\akanksha\\Downloads\\glovefile\\glovefile.txt"
import numpy as np
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    
     
    with open(gloveFile, encoding="utf8" ) as f:
       content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
     
     
model= loadGloveModel(file)   
 
print (model['hello'])
#print(model['also i was the point'])


# In[16]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
#STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    #text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    #text = model[text]
    return text
    
df['utterance'] = df['utterance'].apply(clean_text)
df['utterance'].apply(lambda x: len(x.split(' '))).sum()


# In[17]:


X = df.utterance
print(X[0])
print(model['also'])
#print(model[X[0]])
y = df.tag


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


# In[19]:


my_tags = ['o','s','ag','dag','c','g','ap','b','q','ans','a','oth']
plt.figure(figsize=(10,4))
df.tag.value_counts().plot(kind='bar');


# In[20]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

#%%time
from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))


# In[21]:


from gensim.models import Word2Vec
wv = gensim.models.KeyedVectors.load_word2vec_format("C:\\Users\\akanksha\\Downloads\\GoogleNews-vectors-negative300.bin.gz", binary=True)
wv.init_sims(replace=True)
from itertools import islice
list(islice(wv.vocab, 13030, 13050))


# In[22]:


def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list ])


# In[23]:


def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens
    
train, test = train_test_split(df, test_size=0.3, random_state = 42)

test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['utterance']), axis=1).values
train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['utterance']), axis=1).values

X_train_word_average = word_averaging_list(wv,train_tokenized)
X_test_word_average = word_averaging_list(wv,test_tokenized)


# In[24]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(X_train_word_average, train['tag'])
y_pred = logreg.predict(X_test_word_average)
print('accuracy %s' % accuracy_score(y_pred, test.tag))
print(classification_report(test.tag, y_pred,target_names=my_tags))


# In[13]:


from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train_word_average, y_train)

#%%time

y_pred = sgd.predict(X_test_word_average)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))


# In[ ]:




