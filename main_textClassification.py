import numpy as np
import pandas as pd
import sklearn as skl
import re #regular expression

from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import naive_bayes
from sklearn import model_selection
from sklearn import metrics

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer #english
from nltk.stem import RSLPStemmer #portuguese
from nltk.stem import snowball #other languages

import textProcessor
import myFunctions 

from sklearn.model_selection import StratifiedKFold

def import_dataset(pathDataset):
    
    datasetFile = open(pathDataset,'r') #abre arquivo para leitura
    
    dataset = [] #lista vazia que ira guardar as mensagens
    target = []
    for line in datasetFile:
        splitLine = line.split(',')
        
        classe = splitLine[0] 
        texto = ' '.join(splitLine[1::])
        
        target.append(classe)
        dataset.append(texto)
        
        #print('%s - %s' %(classe,texto))
    
    target = np.asarray(target) #convert a lista para uma array do numpy
    dataset = np.asarray(dataset) #convert a lista para uma array do numpy
        
    return dataset, target
   
def return_classifier(method):

    if method == 'M.NB': #multinomial naive Bayes
        classifier = skl.naive_bayes.MultinomialNB()   
        
    elif method == 'SVM':
        classifier_svm = skl.svm.SVC()
        param_grid = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4], 'kernel': ['rbf']},
        ]
        
        classifier = skl.model_selection.GridSearchCV(classifier_svm, cv=5, param_grid=param_grid, scoring = 'f1_macro')       
                
    elif method == 'DT': #Decision Trees 
        classifier = skl.tree.DecisionTreeClassifier(random_state = 5)
        
    elif method == 'LR': #Logistic Regression
        classifier = skl.linear_model.LogisticRegression() 

    elif method == 'KNN': #K nearest Neighbors 
        classifier_knn  = skl.neighbors.KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(5,31,5)} 
        
        classifier = skl.model_selection.GridSearchCV(classifier_knn, cv=5, param_grid=param_grid, scoring = 'f1_macro')

    elif method == 'RF': #Random Forest 
        classifier_rf = skl.ensemble.RandomForestClassifier(random_state = 5) 
        param_grid = {'n_estimators': np.arange(10,101,10)} 
        
        classifier = skl.model_selection.GridSearchCV(classifier_rf, cv=5, param_grid=param_grid, scoring = 'f1_macro')
        
    elif method == 'bagging':  
        classifier_bagging  = skl.ensemble.BaggingClassifier()
        param_grid = {'n_estimators': np.arange(10,101,10)} 
        
        classifier = skl.model_selection.GridSearchCV(classifier_bagging, cv=5, param_grid=param_grid, scoring = 'f1_macro')

    elif method == 'adaboost':  
        classifier_adaboost  = skl.ensemble.AdaBoostClassifier()
        param_grid = {'n_estimators': np.arange(10,101,10)} 
        
        classifier = skl.model_selection.GridSearchCV(classifier_adaboost, cv=5, param_grid=param_grid, scoring = 'f1_macro')

    return classifier

def perform_experiment(dataset, target, methodName, nomeDataset, pathResults, stopWords, stemming, termWeighting):
    
    classesDataset = list(set(target)) #possiveis classes da base de dados
    
    #realiza as etapas de pre-processamento na base de dados
    for i in range( len(dataset) ):
        dataset[i] = textProcessor.trataTexto(dataset[i], stopWords = stopWords, stemming = stemming, corpusLanguage = 'english')
    
    #divide a base de dados usando validacao cruzada k-folds
    cv = skl.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state = 0)
    
    resultados=[]
    i=0
    for train_index, test_index in cv.split(dataset, target):
        print('\n\t==============================================')
        print('\tK-folds: %d' %(i+1))
        print('\t==============================================')
        
        dataset_train, dataset_test = dataset[train_index], dataset[test_index]
        y_train, y_test = target[train_index], target[test_index]
        
        #Convert a collection of text documents to a matrix of token counts
        vectorizer = skl.feature_extraction.text.CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, lowercase = True, binary=False, dtype=np.int32 )
        x_train = vectorizer.fit_transform(dataset_train) 
        x_test = vectorizer.transform(dataset_test) #converte os dados de teste, usando o modelo gerado pelos dados de treinamento 
                
        if termWeighting == 'TFIDF_sklearn':    
            tfidf_model = skl.feature_extraction.text.TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
            x_train = tfidf_model.fit_transform(x_train)
            x_test = tfidf_model.transform(x_test)
        elif termWeighting == 'TFIDF':  #TFIDF calculado de forma diferente do scikit learn, conforme feito no artigo "MDLText: An efficient and lightweight text classifier" 
            tfidf_model = textProcessor.tf2tfidf(normalize_tf=True, normalize_tfidf=True)
            x_train = tfidf_model.fit_transform(x_train)
            x_test = tfidf_model.transform(x_test)
        elif termWeighting == 'binary':
            x_train[x_train!=0]=1 #convert os dados para representação binária
            x_test[x_test!=0]=1 #convert os dados para representação binária
        
        classifier = return_classifier(methodName)
        classifier.fit(x_train, y_train) #treina o classificador com os dados de treinameto
        y_pred = classifier.predict(x_test) #classifica os dados de teste
        
        # Compute confusion matrix
        cm = skl.metrics.confusion_matrix(y_test, y_pred, classesDataset)
        auxResults = myFunctions.inf_teste(cm, classesDataset, printResults=True)
        resultados.append( auxResults ) 
               
        i+=1
        
    auxMethod = methodName+'_'+termWeighting
    myFunctions.imprimiResultados(resultados,classesDataset,pathResults,auxMethod,nomeDataset)
    
def main():
    '''
    Parâmetros:

	pathResults: é o endereço da base de dados. A base de dados precisa estar no formato CSV.
                 A primeira linha dessa base deve conter o nome dos atributos. A última coluna
                 deve ter o nome "class" e conter as classes do problema.
                 
	termWeighting:
		TF: term frequency 
		binary: os pesos dos termos são 0 se o termo aparece no texto ou 1 caso não apareça
		TFIDF_sklearn: TFIDF calculado por meio da função do scikit learn
		TFIDF: TFIDF calculado por meio da função apresentada no artigo "MDLText: An efficient and lightweight text classifier"

	stopWords: 
		True: remove as stopwords dos textos
		False: não remove as stopwords dos textos

	stemming: 
		True: aplica stemming nos textos
		False: não aplica stemming nos textos
    '''

    pathDataset = 'datasets/SMSSpamCollection.txt'
    nomeDataset = 'SMS' #vai ser usado quando for imprimir os resultados da classificação
    
    pathResults = 'resultados/results.csv'
    
    termWeighting = 'TFIDF_sklearn' 
    stopWords = True;
    stemming = True;
    
    dataset, target = import_dataset(pathDataset)
    
    metodos = ['M.NB'] #['M.NB','SVM','DT','LR','KNN','RF','bagging','adaboost']
    
    for methodName in metodos:
        print('%s' %(methodName))
        perform_experiment(dataset, target, methodName, nomeDataset, pathResults, stopWords, stemming, termWeighting)
    
if __name__ == "__main__":
    
    main() #executa a função principal    
    
