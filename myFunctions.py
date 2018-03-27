# -*- coding: utf-8 -*- 

import numpy as np
import sklearn as skl
import scipy
from scipy.sparse import csc_matrix

from sklearn import preprocessing

import pandas as pd
import re #regular expression
import os
import sys 

#==================================================
#Função para calcular as métricas de classificação
#==================================================
def inf_teste(matriz_confusao, classes, printResults=True):
  #print(matriz_confusao)
  
  n_teste = sum(sum(matriz_confusao))
  
  nClasses = len( matriz_confusao ) #numero de classes
  vp=np.zeros( (1,nClasses) )
  vn=np.zeros( (1,nClasses) )
  fp=np.zeros( (1,nClasses) )
  fn=np.zeros( (1,nClasses) )

  #Laço para encontrar vp, vn, fp e fn de todas as classes
  for i in range(0,nClasses):
    vp[0,i] = matriz_confusao[i,i];
    fn[0,i] = sum(matriz_confusao[i,:])-matriz_confusao[i,i];
    fp[0,i] = sum(matriz_confusao[:,i])-matriz_confusao[i,i];
    vn[0,i] = n_teste - vp[0,i] - fp[0,i] - fn[0,i];

  sensitividade = vp/(vp+fn) #recall
  sensitividade = np.nan_to_num(sensitividade) #Replace nan with zero and inf with finite numbers

  fpr = fp/(fp+vn) #false positive rate 
  fpr = np.nan_to_num(fpr) #Replace nan with zero and inf with finite numbers
  
  especificidade = vn/(fp+vn) #especificidade
  #acuracia = (vp+vn)/(vp+vn+fp+fn)
  acuracia = np.zeros( (1,nClasses ) ) #inicializa a variavel acuracia
  acuracia[0:nClasses]=np.sum(vp)/(vp[0,0]+vn[0,0]+fp[0,0]+fn[0,0])#quantidade de acertos dividido pelo numero de testes
  
  precisao = vp/(vp+fp); #precision
  precisao = np.nan_to_num(precisao) #Replace nan with zero and inf with finite numbers
  
  f_medida = (2*precisao*sensitividade)/(precisao+sensitividade)
  f_medida = np.nan_to_num(f_medida) #Replace nan with zero and inf with finite numbers
  
  mcc = ( (vp*vn)-(fp*fn) ) / np.sqrt( (vp+fp)*(vp+fn)*(vn+fp)*(vn+fn) )

  #microAverage average
  sensitividade_microAverage = sum(vp[0,:])/sum(vp[0,:]+fn[0,:])#sensitividade ou recall
  precisao_microAverage = sum(vp[0,:])/sum(vp[0,:]+fp[0,:])
  f_medida_microAverage = (2*precisao_microAverage*sensitividade_microAverage)/(precisao_microAverage+sensitividade_microAverage)

  #macro average
  auxSensitividade_macroAverage = vp[0,:]/(vp[0,:]+fn[0,:])
  auxSensitividade_macroAverage = np.nan_to_num(auxSensitividade_macroAverage) #Replace nan with zero and inf with finite numbers
  sensitividade_macroAverage = sum( auxSensitividade_macroAverage )/nClasses#sensitividade ou recall

  auxPrecisao_macroAverage = vp[0,:]/(vp[0,:]+fp[0,:])
  auxPrecisao_macroAverage = np.nan_to_num(auxPrecisao_macroAverage) #Replace nan with zero and inf with finite numbers
  precisao_macroAverage = sum( auxPrecisao_macroAverage )/nClasses

  f_medida_macroAverage = (2*precisao_macroAverage*sensitividade_macroAverage)/(precisao_macroAverage+sensitividade_macroAverage);

  #coeficiente Kappa
  sumLinhas = np.zeros( (1,nClasses) )
  sumColunas = np.zeros( (1,nClasses) )
  for i in range(0,nClasses):
    sumLinhas[0,i] = sum(matriz_confusao[i,:]);
    sumColunas[0,i] = sum(matriz_confusao[:,i]);

  rand = sum( (sumLinhas[0,:]/n_teste)*(sumColunas[0,:]/n_teste) )
  kappa_coefficient = (acuracia[0][0] - rand)/(1-rand);
  
  if printResults == True:
      print('\n\tFPR        Recall     Espec.   Precisao   F-medida   Classe')
      for i in range(0,nClasses):
        print('\t%1.3f      %1.3f      %1.3f    %1.3f      %1.3f      %s' % (fpr[0,i], sensitividade[0,i], especificidade[0,i], precisao[0,i], f_medida[0,i],classes[i] ) )
    
      print('\t------------------------------------------------------------------------------------');
      #imprimi as médias
      print('\t%1.3f      %1.3f      %1.3f    %1.3f      %1.3f      Media' % (np.mean(fpr), np.mean(sensitividade), np.mean(especificidade), np.mean(precisao), np.mean(f_medida) ) )
      print('\t.....      %1.3f      .....    %1.3f      %1.3f      Macro-Average' % (sensitividade_macroAverage, precisao_macroAverage, f_medida_macroAverage) )
      print('\t.....      %1.3f      .....    %1.3f      %1.3f      Micro-Average\n' % (sensitividade_microAverage, precisao_microAverage, f_medida_microAverage) )
    
      print('\tacuracia: %1.3f' %acuracia[0,0])
      print('\tkappa_coefficient:  %1.3f' %kappa_coefficient)
      if nClasses==2:
          print('\tMCC:  %1.3f' %mcc[0,0])

  resultados = {'fpr': fpr, 'sensitividade': sensitividade, 'especificidade': especificidade, 'acuracia': acuracia, 'precisao':precisao, 'f_medida':f_medida, 'mcc':mcc}
  resultados.update({'sensitividade_macroAverage':sensitividade_macroAverage, 'precisao_macroAverage':precisao_macroAverage, 'f_medida_macroAverage':f_medida_macroAverage})
  resultados.update({'sensitividade_microAverage':sensitividade_microAverage, 'precisao_microAverage':precisao_microAverage, 'f_medida_microAverage':f_medida_microAverage})
  resultados.update({'kappa_coefficient': kappa_coefficient})
  resultados.update({'confusionMatrix': matriz_confusao})

  return resultados #return like a dictionary


#============================================
#Função para imprimir as médias dos folds
#============================================
def imprimiResultados(resultados,classes,end_resultados,metodo,nomeDataset,print_class=None):
    nfolds = len(resultados)

    if not os.path.isfile(end_resultados):#se o arquivo não existe, adiciona os labels das colunas do .csv
        fileWrite  = open(end_resultados,"a") #abre arquivo em modo de ediçãos
        fileWrite.write('base_dados,metodo,acuraciaMean,fprMean,sensitividadeMean,especificidadeMean,precisaoMean,F-medidaMean,mcc,sensitividadeMacro,precisaoMacro,F-medidaMacro,sensitividadeMicro,precisaoMicro,F-medidaMicro,kappa,roc_auc,tempo,confusionMatrix')    
        fileWrite.close();
        
    fileWrite  = open(end_resultados,"a") #abre arquivo em modo de ediçãos
    
    for i in range(0,len(resultados)):
        if print_class is None: #imprimi a média
            fileWrite.write('\n%-30s,%-20s,%1.3f,%1.3f,' %( nomeDataset, metodo, np.mean(resultados[i]['acuracia']), np.mean(resultados[i]['fpr']) ))
            fileWrite.write('%1.3f,%1.3f,%1.3f,' %( np.mean(resultados[i]['sensitividade']), np.mean(resultados[i]['especificidade']), np.mean(resultados[i]['precisao']) )) 
            fileWrite.write('%1.3f,%1.3f,' %( np.mean(resultados[i]['f_medida']), np.mean(resultados[i]['mcc']) ))
            
        else: 
            idClass = classes.index( print_class )
            fileWrite.write('\n%-30s,%-20s,%1.3f,%1.3f,' %( nomeDataset, metodo, resultados[i]['acuracia'][0,idClass], resultados[i]['fpr'][0,idClass] ))
            fileWrite.write('%1.3f,%1.3f,%1.3f,' %( resultados[i]['sensitividade'][0,idClass], resultados[i]['especificidade'][0,idClass], resultados[i]['precisao'][0,idClass] )) 
            fileWrite.write('%1.3f,%1.3f,' %( resultados[i]['f_medida'][0,idClass], resultados[i]['mcc'][0,idClass] ))

            
        fileWrite.write('%1.3f,%1.3f,%1.3f,' %( resultados[i]['sensitividade_macroAverage'],  resultados[i]['precisao_macroAverage'], resultados[i]['f_medida_macroAverage'] ))
        fileWrite.write('%1.3f,%1.3f,%1.3f,' %( resultados[i]['sensitividade_microAverage'],  resultados[i]['precisao_microAverage'], resultados[i]['f_medida_microAverage'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['kappa_coefficient'] ))
        if 'roc_auc' in resultados[i]:
            roc_auc = resultados[i]['roc_auc']
        else:
            roc_auc = 0.0
        fileWrite.write('%1.3f,' %( roc_auc ))
        
        if 'timeFolds' in resultados[i]:
            timeFolds = resultados[i]['timeFolds']
        else:
            timeFolds = 0.0
        fileWrite.write('%1.3f,' %( timeFolds ))	
            
        xmlLabels = labels_to_xml(classes);
        xmlConfusionMatrix = matrix_to_xml( resultados[i]['confusionMatrix'] );
        fileWrite.write('%s%s' %( xmlLabels,xmlConfusionMatrix ))	
        
    fileWrite.close();

def labels_to_xml(labels):
     xmlString = ''
     if type(labels) is list:
         for i in range( len(labels) ):
            xmlString = xmlString+r'<l>'+str(labels[i])+r'</l>'
     else:
         for i in range( len(labels) ):
            xmlString = xmlString+r'<l>'+labels[i][0]+r'</l>'

                                               
     xmlString = r'<labels>'+xmlString+r'</labels>'
    
     return xmlString
    
def matrix_to_xml(data):
    
    xmlString = '';
    for i in range( data.shape[0] ):
        xmlString = xmlString+r'<row id="'+str(i+1)+r'">';
        for j in range( data.shape[1] ):
            if j==0:
                xmlString = xmlString+str( data[i,j] )
            else:
                xmlString = xmlString+' '+str( data[i,j] )
        xmlString = xmlString+r'</row>'
    xmlString = r'<matrix>'+xmlString+r'</matrix>';
                
    return xmlString


