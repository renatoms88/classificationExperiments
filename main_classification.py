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

def main():
    """
    Função principal
    """  
    
    # pathDataset: é o endereço da base de dados. A base de dados precisa estar no formato CSV.
    # A primeira linha dessa base deve conter o nome dos atributos. A última coluna deve conter
    # o nome "class" e conter as classes do problema.
    pathDataset = 'datasets/iris.csv'
    
    # dê um nome qualquer para a base de dados para identificar o experimento no arquivo de resultados que será gerado pelo algoritmo    
    nomeDataset = 'iris'
    
    # indique o endereço do arquivo onde você deseja que os resultados da classificação sejam guardados.
    # Se o arquivo indicado não existir, ele será criado. Caso já exista, os resultados serão acrescentados ao fim do arquivo.
    pathResults = 'resultados/results.csv'

    # performGrid: 
    #    Possíveis valores: True, False
    #	       True: usa grid search (busca em grade) nos métodos (e.g. SVM, KNN e Random Forest) que são sensíveis a variação de parâmetros.
    #	       False: não usa grid search em nenhum método 
    performGrid = True
       
    # importa o arquivo e guarda em um data frame do Pandas
    df_dataset = pd.read_csv( pathDataset, sep=',') 
    
    # remove a coluna 'class' do data frame e pega os valores das colunas restantes no formato de um array numpy
    dataset = df_dataset.drop(['class'], axis=1).values 
    
    # pega os valores da coluna 'class' e converte para o formato array numpy
    target = df_dataset['class'].values 
    
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
    metodos = ['G.NB','SVM','DT','LR','KNN','RF','bagging','adaboost'] 

    # Para cada método da lista de métodos, executa um experimento com os parâmetros informados    
    for methodName in metodos:
        # imprimi o nome do método que será executado nessa iteração
        print('\n\n\n########################################')
        print('%s' %(methodName)) 
        print('########################################\n')
              
        # executa um experimento com o método da iteração atual
        perform_experiment(dataset, target, methodName, nomeDataset, pathResults, performGrid)
        
   
def return_classifier(method, performGrid):
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
        
    performGrid: boolean
    	       True: usa grid search (busca em grade) nos métodos (e.g. SVM, KNN e Random Forest) que são sensíveis a variação de parâmetros.
    	       False: não usa grid search em nenhum método  
    """  
    
    if method == 'M.NB': #multinomial naive Bayes
        # inicia o classificador
        classifier = skl.naive_bayes.MultinomialNB()   

    if method == 'G.NB': #gaussian naive Bayes
        # inicia o classificador
        classifier = skl.naive_bayes.GaussianNB()  
        
    elif method == 'SVM':
        # inicia o classificador com os parâmetros default do Scikit
        classifier = skl.svm.SVC()
        
        # alguns métodos (e.g. SVM, KNN e Random Forest) são sensíveis a variação de parâmetros. 
        # Por isso, é recomendado selecionar os parâmetros usando uma busca em grade.
        if performGrid:
            
            param_grid = [
              {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4], 'kernel': ['rbf']},
            ]
            
            # inicia o classificador SVM, implementado usando uma busca em grade
            classifier = skl.model_selection.GridSearchCV(classifier, cv=5, param_grid=param_grid, scoring = 'f1_macro')       
                
    elif method == 'DT': #Decision Trees 
        # inicia o classificador
        classifier = skl.tree.DecisionTreeClassifier(random_state = 5)
        
    elif method == 'LR': #Logistic Regression
        # inicia o classificador
        classifier = skl.linear_model.LogisticRegression() 

    elif method == 'KNN': #K nearest Neighbors 
        # inicia o classificador com os parâmetros default do Scikit
        classifier  = skl.neighbors.KNeighborsClassifier()
        
        # alguns métodos (e.g. SVM, KNN e Random Forest) são sensíveis a variação de parâmetros. 
        # Por isso, é recomendado selecionar os parâmetros usando uma busca em grade.
        if performGrid:
            param_grid = {'n_neighbors': np.arange(5,31,5)} 
            
            # inicia o classificador KNN, implementado usando uma busca em grade
            classifier = skl.model_selection.GridSearchCV(classifier, cv=5, param_grid=param_grid, scoring = 'f1_macro')

    elif method == 'RF': #Random Forest 
        # inicia o classificador com os parâmetros default do Scikit
        classifier = skl.ensemble.RandomForestClassifier(random_state = 5) 
        
        # alguns métodos (e.g. SVM, KNN e Random Forest) são sensíveis a variação de parâmetros. 
        # Por isso, é recomendado selecionar os parâmetros usando uma busca em grade.
        if performGrid:
            param_grid = {'n_estimators': np.arange(10,101,10)} 
            
            # inicia o classificador RF, implementado usando uma busca em grade
            classifier = skl.model_selection.GridSearchCV(classifier, cv=5, param_grid=param_grid, scoring = 'f1_macro')
        
    elif method == 'bagging':  
        # inicia o classificador com os parâmetros default do Scikit
        classifier  = skl.ensemble.BaggingClassifier()
        
        # alguns métodos (e.g. SVM, KNN e Random Forest) são sensíveis a variação de parâmetros. 
        # Por isso, é recomendado selecionar os parâmetros usando uma busca em grade.
        if performGrid:
            param_grid = {'n_estimators': np.arange(10,101,10)} 
            
            # inicia o classificador Bagging, implementado usando uma busca em grade
            classifier = skl.model_selection.GridSearchCV(classifier, cv=5, param_grid=param_grid, scoring = 'f1_macro')

    elif method == 'adaboost':  
        # inicia o classificador com os parâmetros default do Scikit
        classifier  = skl.ensemble.AdaBoostClassifier()
        
        # alguns métodos (e.g. SVM, KNN e Random Forest) são sensíveis a variação de parâmetros. 
        # Por isso, é recomendado selecionar os parâmetros usando uma busca em grade.
        if performGrid:
            param_grid = {'n_estimators': np.arange(10,101,10)} 
    
            # inicia o classificador AdaBoost, implementado usando uma busca em grade        
            classifier = skl.model_selection.GridSearchCV(classifier, cv=5, param_grid=param_grid, scoring = 'f1_macro')

    return classifier

def perform_experiment(dataset, target, methodName, nomeDataset, pathResults, performGrid):
    """
    Função usada para executar os experimentos
    """
    
    classesDataset = list(set(target)) # possiveis classes da base de dados
    
    # divide a base de dados usando validacao cruzada k-folds
    cv = skl.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state = 0)
    
    resultados=[] # cria uma lista vazia para guardar os resultados obtidos em cada fold
    i=0 
    for train_index, test_index in cv.split(dataset, target):
        print('\n\t==============================================')
        print('\tK-folds: %d' %(i+1))
        print('\t==============================================')
        
        x_train, x_test = dataset[train_index], dataset[test_index]
        y_train, y_test = target[train_index], target[test_index]
        
          
        # Standardize features by removing the mean and scaling to unit variance
        scaler = skl.preprocessing.StandardScaler().fit(x_train)
        x_train= scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        
        # chama a função para retornar um classificador baseado no nome fornecido como parâmetro
        classifier = return_classifier(methodName, performGrid)
        
        # treina o classificador com os dados de treinameto
        classifier.fit(x_train, y_train) 
        
        # classifica os dados de teste
        y_pred = classifier.predict(x_test) 
        
        # Compute confusion matrix
        cm = skl.metrics.confusion_matrix(y_test, y_pred, classesDataset)
        
        # chama a função 'inf_teste' para calcular e retornar o desempenho da classificação. 
        # Essa função calcula a acurácia, F-medida, Precisão e várias outras medidas.
        auxResults = myFunctions.inf_teste(cm, classesDataset, printResults=True)
        
        # adiciona os resultados do fold atual na lista de resultados
        resultados.append( auxResults ) 
               
        i+=1
        
    # a função 'imprimiResultados' salva os resultados da classificação em formato CSV.
    # Se o arquivo indicado pela variável 'pathResults' não existir, ele será criado. Caso já exista, os resultados serão acrescentados ao fim do arquivo.
    myFunctions.imprimiResultados(resultados,classesDataset,pathResults,methodName,nomeDataset)
        
if __name__ == "__main__":
    
    main() #executa a função principal
