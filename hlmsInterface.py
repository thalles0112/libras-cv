import numpy as np
import pickle
from hlmsGetter import get_training_data
import os


def getandsavedata(DIR, labels, size):
    '''
    Parametros:
    - DIR:    str | Pasta principal com subpastas com videos, 
    - labels: list | Nome das subpastas,
    - size:  tuple | Dimensao destino do arquivo de video.

    Este modulo faz ligacao com o hlmsGetter. Sua vantagem e que
    ele guarda os dados treinados num arquivo .pickle para ser usado
    depois sem ter que ler os videos novamente.
    '''
    arr = get_training_data(DIR, labels, size)
    with open('array.pickle', 'wb') as arrfile:
        pickle.dump(arr, arrfile)
        arrfile.close()    
    


def readdata():
    '''
    Esta funcao le o arquivo .pickle gerado pelo metodo getandsavedata
    e retorna os dados para serem treinados e seus rotulos nas variaveis
    array e label respectivamente.
    '''

    with open('array.pickle', 'rb') as arrfile:
        array, labels = pickle.load(arrfile)
        array = np.array(array, 'float32')
        #labels = np.array(labels)
        arrfile.close()
        return array, labels


