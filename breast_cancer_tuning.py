"""
Esse arquivo tem como objetivo testar diversos parametros para saber a 
melhor escolha na rede neural
"""
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
# Fará uma pesquisa em grade para saber os melhores parâmetros
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(optmizer, loss, kernel_initializer, activation, neurons, dropout):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation,
                            kernel_initializer=kernel_initializer , input_dim=30))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units = neurons, activation = activation,
                            kernel_initializer=kernel_initializer ))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units = neurons, activation = activation,
                            kernel_initializer=kernel_initializer ))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    classificador.compile(optimizer=optmizer, loss=loss,
                          metrics=['binary_accuracy'])
    return classificador


classificador = KerasClassifier(build_fn=criarRede)
# lista de parâmetros que queremos testar na nossa rede neural
parametros={'batch_size':[10],
            'epochs':[150],
            'optmizer':['adam'],
            'loss':['binary_crossentropy'],
            'kernel_initializer':['random_uniform'],
            'activation':['relu'],
            'neurons':[8,32],
            'dropout':[0.1,0.3,0.2]}
# Faz as combinações e os testes de todos com todos
grid_search = GridSearchCV(estimator=classificador,
                           param_grid=parametros,
                           scoring='accuracy',
                           cv=5)
# Coloca as bases para teste
grid_search = grid_search.fit(previsores,classe)
# separa o melhor resultado de accuracy
melhores_parametros = grid_search.best_params_
# mostra o valor do melhor resultado
melhor_precisao = grid_search.best_score_ 