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


def main():
    """
    Função principal
    """ 
    
    # 'pathDataset' é o endereço da base de dados. Cada linha da base de dados, deve ter o formato <classe, mensagem>.
    pathDataset = 'datasets/SMSSpamCollection.txt'
    
    # dê um nome qualquer para a base de dados para identificar o experimento no arquivo de resultados que será gerado pelo algoritmo
    nomeDataset = 'SMS' 
    
    # indique o endereço do arquivo onde você deseja que os resultados da classificação sejam guardados.
    # Se o arquivo indicado não existir, ele será criado. Caso já exista, os resultados serão acrescentados ao fim do arquivo.
    pathResults = 'resultados/results.csv'
    
    # termWeighting: usado para indicar qual esquema de pesos você quer usar para os termos
    #     Possíveis valores: 'TF', 'binary', 'TFIDF_sklearn', 'TFIDF'
    #          'TF': term frequency 
    #          'binary': os pesos dos termos são 0 se o termo aparece no texto ou 1 caso não apareça
    #          'TFIDF_sklearn': TFIDF calculado por meio da função do scikit learn
    #          'TFIDF': TFIDF calculado por meio da função apresentada no artigo "MDLText: An efficient and lightweight text classifier"
    termWeighting = 'TFIDF_sklearn' 
    
    # stopWords: 
    #    Possíveis valores: True, False
    #	       True: remove as stopwords dos textos
    #	       False: não remove as stopwords dos textos
    stopWords = True;

    # stemming: 
    #    Possíveis valores: True, False
    #	       True: aplica stemming nos textos
    #	       False: não aplica stemming nos textos   
    stemming = True;
    
    # função usada para importar a base de dados. 
    # Essa função retorna as seguintes variáveis:
    #      dataset: um array de 1 coluna, onde cada linha corresponde a uma mensagem 
    #      target: um vetor com as classes de cada mensagem contida no array "dataset"
    dataset, target = import_dataset(pathDataset)
    
    # crie uma lista com os métodos que você deseja executar:
    #      'M.NB': Multinomial naive Bayes
    #      'SVM': Support vector machines
    #      'DT': Decision trees
    #      'LR': Logistic regression
    #      'KNN': k-nearest neighbors
    #      'RF': Random forest
    #      'bagging': Bagging
    #      'adaboost': AdaBoost
    #
    # Você pode adicionar outros métodos na função "return_classifier()" 
    metodos = ['M.NB'] #['M.NB','SVM','DT','LR','KNN','RF','bagging','adaboost'] 
    
    # Para cada método da lista de métodos, executa um experimento com os parâmetros informados
    for methodName in metodos:
        # imprimi o nome do método que será executado nessa iteração
        print('\n\n\n########################################')
        print('%s' %(methodName)) 
        print('########################################\n')
        
        # executa um experimento com o método da iteração atual
        perform_experiment(dataset, target, methodName, nomeDataset, pathResults, stopWords, stemming, termWeighting)


def import_dataset(pathDataset):
    """
    Função usada para importar a base de dados textual
    """
     
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
    """
    Função usada para selecionar um método de classificação para ser usado no experimento
    """   

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
    """
    Função usada para executar os experimentos
    """
    
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
        
if __name__ == "__main__":
    
    main() #executa a função principal    
    
