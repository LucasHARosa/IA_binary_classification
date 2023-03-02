"""
Esse arquivo tem como objetivo usar todos os registros para realizar o
treinamento
"""
import pandas as pd
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

import keras
# Modelo que a rede neural irá funcionar sequencial
from keras.models import Sequential
# Tipo de camadas que será gerada. Denso significa que cada neuronio se conectará a todos os neurinios da camada seguinte
from keras.layers import Dense, Dropout
# Para validação cruzada
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def criarRede():
    # Será nossa rede neural
    classificador = Sequential()

    # 1° camada oculta a rede neural
    """
    Parametros Dense:
        units: quantidade de neuronios
            units = (entradas + saidas)/2
        activation: Função de ativação
            relu: max(0,x) 
        kernel_initializer: Inicialização dos pesos
            random_uniform: Inicializador que gera tensores com distribuição uniforme
        input_dim: Número de entradas, somente na primeira camada oculta 
        use_bias: true ou false, cada neurônio terá um bias
    """
    classificador.add(Dense(units = 32, activation = 'relu',
                            kernel_initializer= 'random_uniform', input_dim=30))
    
    # Dropout é uma maneira de não causar um overfitting, zeras algumas entradas
    """
    Parametros Dropout:
        rate: porcentagem que será ignorada
    """
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
    """
    Parametros Dense:
        units: quantidade de neuronios
            units = saida
        activation: Função de ativação
            sigmoid: 1 / (1 + exp(-x)) retorna valor entre 0 e 1
    """
    classificador.add(Dense(units = 1, activation = 'sigmoid'))

    # configurando um otimizador
    #otimizador = keras.optimizers.Adam(lr=0.001,decay=0.0001,clipvalue=0.5)

    # Compilador de rede neural
    """
    Parametros compile:
        optimizer: padrão de atualização
            adam: É um método de descida de gradiente estocástico baseado na estimativa adaptativa de momentos de primeira e segunda ordem
        loss: função de perda ou erro
            binary_crossentropy: entropia cruzada para aplicações de classificação binária (0 ou 1)
        metric: Metrica de avaliação
            binary_accuracy
    """
    classificador.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=['binary_accuracy'])
    return classificador

# Classificador treinamento da rede
"""
Parametros:
    build_fn: função que está a rede neural compilada
    batch_size: número de registros que será calculado o erro
    epochs: Quantas épocas terão o aprendizado
"""
classificador = KerasClassifier(build_fn = criarRede, epochs = 150, batch_size=10)

# Para fazer os testes várias vezes
"""
Parametros:
    estimator: classe da rede neural
    X: Atributos previsores
    y: Atributos de resultado
    cv: Número de vezes que será dividida a base de dados
    scoring: Como os resultados serão retornados
"""
resultados = cross_val_score(estimator=classificador, X = previsores, y=classe, cv = 10, scoring='accuracy')

# media dos resultados
media = resultados.mean()
# desvio padrão dos resultados
desvioPadrao = resultados.std()
    

