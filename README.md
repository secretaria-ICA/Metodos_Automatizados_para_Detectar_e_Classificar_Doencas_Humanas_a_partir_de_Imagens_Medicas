# Métodos automatizados para detectar e classificar doenças humanas a partir de imagens médicas

Projeto de conclusão do curso [BI-MASTER](https://ica.puc-rio.ai/bi-master/) e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

#### Aluna: [Renata Regina da Fonseca Santos](https://github.com/rrfsantos)

#### Orientadora: Professora Evelyn Conceição

* [Link para o código]

### Resumo

A tomografia de coerência óptica da retina (OCT) é uma técnica de imagem usada para capturar seções transversais de alta resolução das retinas de pacientes vivos. Aproximadamente 30 milhões de varreduras de OCT são realizadas a cada ano, e a análise e interpretação dessas imagens levam um tempo significativo (Swanson e Fujimoto, 2017). 
O objetivo do trabalho é a propor um modelo de inteligência artificial pré-treinado como alternativa para predição de diagnóstico utilizando essas de imagens.

### 1. Introdução

Este trabalho baseou-se na API Keras para a construção da rede neural e no módulo scikit-learn para validação do treinamento do modelo.

#### Itens do trabalho:

* Análise exploratória dos dados
* Tratamento dos dados para uso na rede neural
* Avaliação da melhor configuração da rede neural


#### Descrição dos dados

O dataset é composto por imagens de Tomografia de Coerência Óptica da Retina (OCT), técnica de imagem usada para capturar seções transversais de alta resolução das retinas. É organizado em 3 diretórios (train, test, val). Cada um desses diretórios contém subdiretórios para cada categoria de imagem (NORMAL, CNV, DME,DRUSEN). São 84,495 imagens (JPEG) e 4 categorias:

* CNV (choroidal neovascularization) - Processo patológico que consiste da formação de novos vasos sanguíneos na COROIDE.
* DME (diabetic macular edema) - Ao longo do tempo, níveis glicêmicos altos podem levar a complicações vasculares em vários tecidos e órgãos, como no coração, sistema nervoso, rins, membros inferiores (pernas e pés) e inclusive nos olhos.
* DRUSEN - pequenos depósitos amarelos ou brancos na retina do olho ou na cabeça do nervo óptico. A presença de drusas é um dos sinais precoces mais comuns de degeneração macular relacionada à idade.
* NORMAL

> Labeled Optical Coherence Tomography (OCT) Images for Classification - Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, v2

### 2. Modelagem

Foram realizadas dezenas de simulações do modelo, utilizando as técnicas e configurações abaixo:

* Separação da base em Treino, Validação e Teste: Foi utilizado 25% da base para validação e 10% para teste.

* Data Augmentation - Aumento da quantidade de imagens, adicionando cópias ligeiramente modificadas de imagens já existentes e redimensionamento para o padrão de entrada da rede neural.

* Transfer Learning - RNN Xception pré-treinada com a base de dados "imageNet". Foram feitos testes utilizando as RNN VGG16 e EfficientNet B0 a B7, com resultados inferiores.

* Quantidade de Neurônios das Camadas Densas (Dense) - Foi utilizada somente uma camada densa com quatro neurônios (número de classes) para a classificação das imagens e função de ativação softmax.

* Otimizador - O melhor resultado foi obtido com o otimizador SGD, utilizando os parâmetros: Learning Rate = 0,045 / Decay = Learning Rate/n° de épocas = 0,1/30 / Momentum = 0,9.

* Indicador de Perda - Foi utilizada a categorical_crossentropy como uma função de perda para classificação multiclasse.

* Épocas de Treinamento - Foram realizadas simulações com 10, 20 e 30 épocas. O ajuste desse parâmetro evidenciou que são necessárias 30 épocas para obter os melhores resultados.

* Callbacks - Evitam o sobretreino da rede (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).

* Stratified K-Fold cross validation - As partições são feitas preservando a porcentagem de amostras para cada classe (estratificada). Foram utilizadas 3 partições para validação do modelo final.

* Balanceamento da base (removido) - Inicialmente, foi feito o balanceamento da base para treinamento, atribuindo pesos a cada classe para evitar qualquer viés por meio de dados não balanceados. Durante os testes de validação do modelo, a utilização dessa técnica mostrou-se ineficaz, pois diminuiu a performance na predição utilizando os dados de teste.

### 3. Resultados

O modelo apresentou a mesma performance com a base de teste, com melhores resultados na classificação de imagens das classes NORMAL e CNV e pior resultado para a classe DRUSEN. com os dados de teste.

![Training and Validation Accuracy](/Projeto-Redes-Neurais-OCT-Images/Training and validation accuracy.PNG/Training and validation accuracy.PNG)

[Training and Validation Loss](Training and validation loss.PNG)
