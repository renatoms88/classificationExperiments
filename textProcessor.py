import os #para listar arquivos e diretorios
import re #regular expression
import numpy as np
import sklearn as skl

import scipy
from scipy.io import savemat
from scipy import sparse, io
from scipy.sparse import csc_matrix

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

from bs4 import BeautifulSoup

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer #english
from nltk.stem import RSLPStemmer #portuguese
from nltk.stem import snowball #other languages

import unicodedata #remover acentos


#funcao para tratar o texto
def trataTexto(text, stopWords = False, stemming = False, regexType = 0, corpusLanguage = 'english'):
    
    if regexType==0:
        regex = re.compile('\W') #Matches any single letter, digit or underscore
    elif regexType==1: 
        regex = re.compile('[^A-Za-z]') #only letters, remove numbers and others special character
        
    text = re.sub(regex, " ", text) #
    
    #text = remove_accents(text) #remove os acentos
    text = text.lower() # Convert to lower case

    #se stopWords==1, remove as stopWords
    if stopWords==True: 
        words = text.split() # Split into words
        # Remove stop words from "words"
        words = [w for w in words if not w in nltk.corpus.stopwords.words(corpusLanguage)] 
        text = " ".join( words )

    #se stemming==1, use stemming
    if stemming==True:
        words = text.split() # Split into words

        if corpusLanguage == 'english':
            stemmer_method = PorterStemmer()  
        elif corpusLanguage == 'portuguese':
            stemmer_method = RSLPStemmer()    
        elif corpusLanguage == 'dutch':
            stemmer_method = snowball.DutchStemmer()
        elif corpusLanguage == 'french':
             stemmer_method = snowball.FrenchStemmer()
        elif corpusLanguage == 'german':
            stemmer_method = snowball.GermanStemmer()
        elif corpusLanguage == 'italian':
            stemmer_method = snowball.ItalianStemmer()
        elif corpusLanguage == 'spanish':
            stemmer_method = snowball.SpanishStemmer()
        elif corpusLanguage == 'spanish-latam': #spanish latin american
            stemmer_method = snowball.SpanishStemmer()
                
        words = [ stemmer_method.stem(w) for w in words ]
        text = " ".join( words )
    
    return text

#function to remove accents, umlauts, etc  
def remove_accents(text):  
	nfkd_form = unicodedata.normalize('NFKD', text)
	return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

#============================================
#Função para converter tf para tf-idf
#============================================
class tf2tfidf():
    """
    Faz a conversão para tf_idf. 
    Quando for usado na fase de teste, deve ser passado a frequência de documentos 
    da base de treinamento que contém cada token e a quantidade de documentos de treinamento. 
    
    Uma das diferença dessa função para a função sklearn.feature_extraction.text.TfidfVectorizer 
    é que ela usa log na base 10 em vez do logaritmo natural. Além disso, o Tf é normalizado como 
    np.log10( 1+tf.data ), enquanto no scikit é normalizado como 1 + np.log( tf.data ). Ainda,
    o IDF é calculado como np.log10( (nDocs+1)/(df+1) ), enquanto no scikit é 
    np.log(nDocs / df) + 1.0
    
    O calculo é feito usando a equação mostrada no artigo "MDLText: An efficient and lightweight text classifier" 
    """    

    def __init__(self, normalize_tf=False, normalize_tfidf=True):
        self.df = None
        self.nDocs = None
        self.normalize_tf = False
        self.normalize_tfidf = True
    
    def fit_transform(self,tf):
        """
        Fit to data, then transform it to a tf-idf matrix
        """
        
        #se não é esparsa, converte em esparsa
        if not scipy.sparse.issparse(tf):
            tf = csc_matrix(tf)
            
        if self.df is None:    
            self.df = (tf != 0).sum(axis=0) #document frequency -- number of documents where term i occurs
        else:
            self.df += (tf != 0).sum(axis=0) #document frequency -- number of documents where term i occurs
            
        if self.nDocs is None:   
            self.nDocs = tf.shape[0] 
        else:
            self.nDocs += tf.shape[0]

        tf_idf = self.transform(tf)
        return tf_idf
    
    def transform(self,tf):
        """
        Transform a TF matrix to a tf-idf matrix
        """
         #se não é esparsa, converte em esparsa
        if not scipy.sparse.issparse(tf):
            tf = csc_matrix(tf)
            
        if self.normalize_tf == True:
            tf.data = np.log10( 1.0+tf.data )
        
        idf = np.log10( (self.nDocs+1.0)/(self.df+1.0) ) #-- we add 1 to avoid 0 division
        idf = csc_matrix(idf);
        
        #tf_idf = csc_matrix( (tf.shape) )
        
        tf_idf = tf.multiply(idf)
            
        #tf_idf = np.nan_to_num(tf_idf) #Replace nan with zero and inf with finite numbers
        #tf_idf2 = csc_matrix(tf_idf)  
            
        if self.normalize_tfidf==True:
            tf_idf = skl.preprocessing.normalize(tf_idf, norm='l2')
            
        return tf_idf    