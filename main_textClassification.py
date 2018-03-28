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
    metodos = ['M.NB','SVM','DT','LR','KNN','RF','bagging','adaboost'] 
    
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
    
    Parameters:
    -----------
    pathDataset: string
        É o endereço da base de dados. Cada linha da base de dados, deve ter o formato <classe, mensagem>.
        
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
    
    Parameters:
    -----------
    method: string
        Um nome usado para identificar o método. Caso deseje, acrescente outros métodos dentro da função. 

        'M.NB': Multinomial naive Bayes
        'SVM': Support vector machines
        'DT': Decision trees
        'LR': Logistic regression
        'KNN': k-nearest neighbors
        'RF': Random forest
        'bagging': Bagging
        'adaboost': AdaBoost        
    """    

    if method == 'M.NB': #multinomial naive Bayes
        # inicia o classificador
        classifier = skl.naive_bayes.MultinomialNB()   
        
    elif method == 'SVM':
        # inicia o classificador SVM. Os parâmetros 'C' e 'kernel' não foram indicados na função SVC(), pois serão selecionados
        # usando uma busca em grade.
        classifier_svm = skl.svm.SVC()
        
        # alguns métodos (e.g. SVM, KNN e Random Forest) são sensíveis a variação de parâmetros. 
        # Por isso, é recomendado selecionar os parâmetros usando uma busca em grade.
        param_grid = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4], 'kernel': ['rbf']},
        ]
        
        # inicia o classificador SVM, implementado usando uma busca em grade
        classifier = skl.model_selection.GridSearchCV(classifier_svm, cv=5, param_grid=param_grid, scoring = 'f1_macro')       
                
    elif method == 'DT': #Decision Trees 
        # inicia o classificador
        classifier = skl.tree.DecisionTreeClassifier(random_state = 5)
        
    elif method == 'LR': #Logistic Regression
        # inicia o classificador
        classifier = skl.linear_model.LogisticRegression() 

    elif method == 'KNN': #K nearest Neighbors 
        # inicia o classificador. O parâmetro 'n_neighbors' não foi indicado, pois será selecionado usando busca em grade.
        classifier_knn  = skl.neighbors.KNeighborsClassifier()
        
        # alguns métodos (e.g. SVM, KNN e Random Forest) são sensíveis a variação de parâmetros. 
        # Por isso, é recomendado selecionar os parâmetros usando uma busca em grade.
        param_grid = {'n_neighbors': np.arange(5,31,5)} 
        
        # inicia o classificador KNN, implementado usando uma busca em grade
        classifier = skl.model_selection.GridSearchCV(classifier_knn, cv=5, param_grid=param_grid, scoring = 'f1_macro')

    elif method == 'RF': #Random Forest 
        # inicia o classificador. O parâmetro 'n_estimators' não foi indicado, pois será selecionado usando busca em grade.
        classifier_rf = skl.ensemble.RandomForestClassifier(random_state = 5) 
        
        # alguns métodos (e.g. SVM, KNN e Random Forest) são sensíveis a variação de parâmetros. 
        # Por isso, é recomendado selecionar os parâmetros usando uma busca em grade.
        param_grid = {'n_estimators': np.arange(10,101,10)} 
        
        # inicia o classificador RF, implementado usando uma busca em grade
        classifier = skl.model_selection.GridSearchCV(classifier_rf, cv=5, param_grid=param_grid, scoring = 'f1_macro')
        
    elif method == 'bagging':  
        # inicia o classificador. O parâmetro 'n_estimators' não foi indicado, pois será selecionado usando busca em grade.
        classifier_bagging  = skl.ensemble.BaggingClassifier()
        
        # alguns métodos (e.g. SVM, KNN e Random Forest) são sensíveis a variação de parâmetros. 
        # Por isso, é recomendado selecionar os parâmetros usando uma busca em grade.
        param_grid = {'n_estimators': np.arange(10,101,10)} 
        
        # inicia o classificador Bagging, implementado usando uma busca em grade
        classifier = skl.model_selection.GridSearchCV(classifier_bagging, cv=5, param_grid=param_grid, scoring = 'f1_macro')

    elif method == 'adaboost':  
        # inicia o classificador. O parâmetro 'n_estimators' não foi indicado, pois será selecionado usando busca em grade.
        classifier_adaboost  = skl.ensemble.AdaBoostClassifier()
        
        # alguns métodos (e.g. SVM, KNN e Random Forest) são sensíveis a variação de parâmetros. 
        # Por isso, é recomendado selecionar os parâmetros usando uma busca em grade.
        param_grid = {'n_estimators': np.arange(10,101,10)} 

        # inicia o classificador AdaBoost, implementado usando uma busca em grade        
        classifier = skl.model_selection.GridSearchCV(classifier_adaboost, cv=5, param_grid=param_grid, scoring = 'f1_macro')

    return classifier

def perform_experiment(dataset, target, methodName, nomeDataset, pathResults, stopWords, stemming, termWeighting):
    """
    Função usada para executar os experimentos
    """
    
    classesDataset = list(set(target)) #possiveis classes da base de dados
    
    for i in range( len(dataset) ):
        # realiza as etapas de pre-processamento no texto, tais como stemming e remoção de stopWords
        dataset[i] = textProcessor.trataTexto(dataset[i], stopWords = stopWords, stemming = stemming, corpusLanguage = 'english')
    
    # divide a base de dados usando validacao cruzada k-folds
    cv = skl.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state = 0)
    
    resultados=[]  # cria uma lista vazia para guardar os resultados obtidos em cada fold
    i=0
    for train_index, test_index in cv.split(dataset, target):
        print('\n\t==============================================')
        print('\tK-folds: %d' %(i+1))
        print('\t==============================================')
        
        dataset_train, dataset_test = dataset[train_index], dataset[test_index]
        y_train, y_test = target[train_index], target[test_index]
        
        # inicia o modelo usado para gerar a representação TF (term frequency)
        vectorizer = skl.feature_extraction.text.CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, lowercase = True, binary=False, dtype=np.int32 )
        
        # treina o modelo TF com os dados de treinamento e converte os dados de treinamento para uma array que contém a frequência dos termos em cada documento (TF - term frequency)
        x_train = vectorizer.fit_transform(dataset_train) 
        
        # converte os dados de teste para uma array que contém a frequência dos termos em cada documento (TF - term frequency)
        x_test = vectorizer.transform(dataset_test) #converte os dados de teste, usando o modelo gerado pelos dados de treinamento 
        
        # converte a representação TF para binária ou TF-IDF  
        #
        # nesse primeiro if, ele converte TF para TFIDF usando a função do scikit-learn
        if termWeighting == 'TFIDF_sklearn':    
            tfidf_model = skl.feature_extraction.text.TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
            x_train = tfidf_model.fit_transform(x_train)
            x_test = tfidf_model.transform(x_test)
            
        # converte TF para TFIDF usando uma equação um pouco diferente do scikit learn. Veja o artigo "MDLText: An efficient and lightweight text classifier" 
        elif termWeighting == 'TFIDF':  
            tfidf_model = myFunctions.tf2tfidf(normalize_tf=True, normalize_tfidf=True)
            x_train = tfidf_model.fit_transform(x_train)
            x_test = tfidf_model.transform(x_test)
            
        # converte TF para binário
        elif termWeighting == 'binary':
            x_train[x_train!=0]=1 #convert os dados para representação binária
            x_test[x_test!=0]=1 #convert os dados para representação binária
        
        # chama a função para retornar um classificador baseado no nome fornecido como parâmetro
        classifier = return_classifier(methodName)
        
        # treina o classificador com os dados de treinameto
        classifier.fit(x_train, y_train) 
        
        #classifica os dados de teste
        y_pred = classifier.predict(x_test) 
        
        # Compute confusion matrix
        cm = skl.metrics.confusion_matrix(y_test, y_pred, classesDataset)
        
        # chama a função 'inf_teste' para calcular e retornar o desempenho da classificação. 
        # Essa função calcula a acurácia, F-medida, Precisão e várias outras medidas.
        auxResults = myFunctions.inf_teste(cm, classesDataset, printResults=True)
        
        # adiciona os resultados do fold atual na lista de resultados
        resultados.append( auxResults ) 
               
        i+=1
    
    # une o nome do método ao nome do esquema de pesos usados para facilitar a identificação do experimento no arquivo de resultados    
    auxMethod = methodName+'_'+termWeighting
    
    # a função 'imprimiResultados' salva os resultados da classificação em formato CSV.
    # Se o arquivo indicado pela variável 'pathResults' não existir, ele será criado. Caso já exista, os resultados serão acrescentados ao fim do arquivo.
    myFunctions.imprimiResultados(resultados,classesDataset,pathResults,auxMethod,nomeDataset)
        
if __name__ == "__main__":
    
    main() #executa a função principal    
    
