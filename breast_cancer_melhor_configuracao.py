"""
Parâmetros mudados:
    3 camadas ocultas
    época de 150
    32 neuronios por camada oculta
    dropout de 0.1 - 10%
    ativação relu
    iniciação dos pesos random_uniform
    otimizador adam padrão
    batch_size=10
    
Media= 91,4%
"""
import pandas as pd
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def criarRede():
    classificador = Sequential()

    # 1° camada oculta a rede neural
    classificador.add(Dense(units = 32, activation = 'relu',
                            kernel_initializer= 'random_uniform', input_dim=30))
    classificador.add(Dropout(0.1))
    
    # 2° camada oculta da rede neural
    classificador.add(Dense(units = 32, activation = 'relu',
                            kernel_initializer= 'random_uniform'))
    classificador.add(Dropout(0.1))
    
    # 3° camada oculta da rede neural
    classificador.add(Dense(units = 32, activation = 'relu',
                            kernel_initializer= 'random_uniform'))
    classificador.add(Dropout(0.1))
    
    # Camada de saida
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    
    classificador.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=['binary_accuracy'])
    return classificador


classificador = KerasClassifier(build_fn = criarRede, epochs = 150, batch_size=10)


resultados = cross_val_score(estimator=classificador, X = previsores, y=classe, cv = 10, scoring='accuracy')

# media dos resultados
media = resultados.mean()
# desvio padrão dos resultados
desvioPadrao = resultados.std()
    

