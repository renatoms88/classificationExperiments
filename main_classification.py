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

    if method == 'G.NB': #gaussian naive Bayes
        classifier = skl.naive_bayes.GaussianNB()  
        
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

def perform_experiment(dataset, target, methodName, nomeDataset, pathResults):
    
    classesDataset = list(set(target)) #possiveis classes da base de dados
    
    #divide a base de dados usando validacao cruzada k-folds
    cv = skl.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state = 0)
    
    resultados=[]
    i=0
    for train_index, test_index in cv.split(dataset, target):
        print('\n\t==============================================')
        print('\tK-folds: %d' %(i+1))
        print('\t==============================================')
        
        x_train, x_test = dataset[train_index], dataset[test_index]
        y_train, y_test = target[train_index], target[test_index]
        
          
        #aplica normalização nos dados
        scaler = skl.preprocessing.StandardScaler().fit(x_train)
        x_train= scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        
        classifier = return_classifier(methodName)
        classifier.fit(x_train, y_train) #treina o classificador com os dados de treinameto
        y_pred = classifier.predict(x_test) #classifica os dados de teste
        
        # Compute confusion matrix
        cm = skl.metrics.confusion_matrix(y_test, y_pred, classesDataset)
        auxResults = myFunctions.inf_teste(cm, classesDataset, printResults=True)
        resultados.append( auxResults ) 
               
        i+=1
        
    auxMethod = methodName
    myFunctions.imprimiResultados(resultados,classesDataset,pathResults,auxMethod,nomeDataset)

def main(): 
    '''
    Parâmetros:

	pathResults: é o endereço da base de dados. A base de dados precisa estar no formato CSV.
                 A primeira linha dessa base deve conter o nome dos atributos. A última coluna
                 deve ter o nome "class" e conter as classes do problema.
    '''  
          
    pathDataset = 'datasets/iris.csv'
    nomeDataset = 'iris' #vai ser usado quando for imprimir os resultados da classificação
    
    pathResults = 'resultados/results.csv'
       
    df_dataset = pd.read_csv( pathDataset, sep=',') 
    
    dataset = df_dataset.drop(['class'], axis=1).values #remove a coluna 'Class' e pega os valores das colunas restantes no formato de um array numpy
    target = df_dataset['class'].values #pega os valores da coluna Class e converte para o formato array numpy
    
    metodos = ['G.NB','SVM','DT','LR','KNN','RF','bagging','adaboost'] 
    
    for methodName in metodos:
        print('\n\n\n########################################')
        print('%s' %(methodName))
        print('########################################\n')
        perform_experiment(dataset, target, methodName, nomeDataset, pathResults)
        
if __name__ == "__main__":
    
    main() #executa a função principal
