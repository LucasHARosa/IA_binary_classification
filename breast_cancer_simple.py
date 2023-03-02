# Para abrir a base de dados
import pandas as pd
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

# Dividindo a base de dados para treinamento e teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe, test_size=0.25)

# rede neural
import keras
# Modelo que a rede neural irá funcionar sequencial
from keras.models import Sequential
# Tipo de camadas que será gerada. Denso significa que cada neuronio se conectará a todos os neurinios da camada seguinte
from keras.layers import Dense

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
classificador.add(Dense(units = 16, activation = 'relu',
                        kernel_initializer= 'random_uniform', input_dim=30))
# 2° camada oculta a rede neural
classificador.add(Dense(units = 16, activation = 'relu',
                        kernel_initializer= 'random_uniform'))

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
otimizador = keras.optimizers.Adam(lr=0.001,decay=0.0001,clipvalue=0.5)

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
classificador.compile(optimizer=otimizador, loss='binary_crossentropy',
                      metrics=['binary_accuracy'])
""",
classificador.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['binary_accuracy'])
"""

# treinamento da rede
"""
Parametros fit:
    batch_size: número de registros que será calculado o erro
    epochs: Quantas épocas terão o aprendizado
"""
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, epochs = 100)

# Mostrar os pesos das ligações dos neuronios de entrada até a primeira camada oculta
pesos0 = classificador.layers[0].get_weights()
# Mostrar os pesos das ligações dos neuronios da primeira camada até a segunda camada oculta
pesos1 = classificador.layers[1].get_weights()
# Mostrar os pesos das ligações dos neuronios da segunda camada até a saida
pesos2 = classificador.layers[2].get_weights()

# Para testar a rede neural usamos o predict na base de teste
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes>0.5)

# Comparar resultados com o previsto
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)