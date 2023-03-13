![GitHub repo size](https://img.shields.io/github/repo-size/LucasHARosa/IA_binary_classification)
![GitHub top language](https://img.shields.io/github/languages/top/LucasHARosa/IA_binary_classification)
![GitHub last commit](https://img.shields.io/github/last-commit/LucasHARosa/IA_binary_classification)

# Redes Neurais artificiais: Classificação binária
## Identificação de tumores de câncer


### Introdução

Redes neurais artificiais são modelos computacionais inspirados no funcionamento do cérebro humano. Elas são compostas por unidades interconectadas, conhecidas como neurônios artificiais, que trabalham juntas para realizar uma tarefa específica, como reconhecimento de padrões, classificação de dados ou previsão de valores.

As redes neurais artificiais são capazes de aprender a partir de exemplos, ajustando os pesos das conexões entre os neurônios para melhorar a precisão da saída. Esse processo de ajuste é chamado de treinamento da rede neural e pode ser supervisionado ou não supervisionado.

No treinamento supervisionado, a rede neural recebe exemplos de entrada, juntamente com as saídas corretas correspondentes, e ajusta seus pesos para minimizar o erro entre as saídas reais e as saídas desejadas. Já no treinamento não supervisionado, a rede neural é exposta apenas aos exemplos de entrada e deve descobrir padrões e estruturas por conta própria.

Uma das principais vantagens das redes neurais artificiais é sua capacidade de lidar com dados complexos e não lineares. Elas são capazes de lidar com informações de múltiplas fontes, como texto, imagem e som, e podem ser usadas em uma variedade de aplicações, como reconhecimento de fala, detecção de fraudes, diagnóstico médico e previsão de tendências.

No entanto, as redes neurais artificiais também têm algumas limitações. Elas podem ser computacionalmente intensivas e exigir grandes quantidades de dados para um treinamento adequado. Além disso, as redes neurais podem ser difíceis de interpretar, tornando difícil entender como e por que elas tomam determinadas decisões.

Apesar dessas limitações, as redes neurais artificiais continuam a ser uma área de pesquisa ativa e uma ferramenta poderosa para resolver problemas complexos em uma variedade de áreas. Com a crescente disponibilidade de dados e avanços na computação, as redes neurais artificiais estão se tornando cada vez mais importantes em nossa sociedade tecnologicamente avançada.

### Problema

Esse projeto tem como objetivo realizar uma classificação binária de um diagnóstico médico, se o tumor é cancerígeno ou não. Temos como base de treinamento um dataframe com 30 aspectos de tumores e seus respectivos resultados. Com isso somos capazes de determinar a partir de uma entrada se o tumor é benigno ou não.

### Arquivos

Treinamento da rede neural com divisão da base entre treinamento e testes

    breast_cancer_simple.py

Treinamento da rede neural cruzada, onde todos os dados são treinados e testados

    breast_cancer_crusade.py

Trienamento para encontrar os melhores parâmetros

    breast_cancer_tuning.py

Treinamento cruzado com os melhores parâmetros encontrados

    breast_cancer_melhor_configuracao.py

Salvando a rede neural com os pesos e bias ajustados

    breast_cancer_salvar.py

Carregando o treinamento da base

    breast_cancer_carregar.py

Verificação da base para um registro externo a base

    breast_cancer_um_registro.py
