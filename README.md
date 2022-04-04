# Métodos_Automatizados_para_Detectar_e_Classificar_Doenças_Humanas_a_partir_de_Imagens_Médicas

Projeto de conclusão do curso [BI-MASTER](https://ica.puc-rio.ai/bi-master/) e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

#### Aluna: [Renata Regina da Fonseca Santos](https://github.com/rrfsantos)

#### Orientadora: [Professora Evelyn Conceição](https://github.com/evysb)

#### Links para o código
1. [Notebook para o download da base, divisão das imagens em 5 folds (treino e validação) e teste. Salva os folds em diretórios no Google Drive.](https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/OCT2017_pre_processamento_split_StratifiedKFold.ipynb)

2. Notebooks para treinamento e avaliação da Rede Neural para cada fold: [Fold 1](https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/OCT2017_Xception_classifier_kfold1.ipynb), [Fold 2](https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/OCT2017_Xception_classifier_kfold2.ipynb), [Fold 3](https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/OCT2017_Xception_classifier_kfold3.ipynb), [Fold 4](https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/OCT2017_Xception_classifier_kfold4.ipynb), [Fold 5](https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/OCT2017_Xception_classifier_kfold5.ipynb)

3. [Notebook com métricas para avaliação do Modelo.](https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/OCT2017_Xception_metricas.ipynb)

## Resumo

A tomografia de coerência óptica da retina (OCT) é uma técnica de imagem usada para capturar seções transversais de alta resolução das retinas de pacientes vivos. Aproximadamente 30 milhões de varreduras de OCT são realizadas a cada ano, e a análise e interpretação dessas imagens levam um tempo significativo (Swanson e Fujimoto, 2017). 
O objetivo do trabalho é a propor um modelo de inteligência artificial pré-treinado como alternativa para predição de diagnóstico utilizando essas de imagens.

## 1. Introdução

Este trabalho baseou-se na API Keras para a construção da rede neural e no módulo scikit-learn para validação do treinamento do modelo.

### Itens do trabalho:

* Análise exploratória dos dados

* Tratamento dos dados para uso na rede neural

* Avaliação da melhor configuração da rede neural

* Avaliação do modelo utilizando validação cruzada


### Descrição dos dados

O dataset é composto por imagens de Tomografia de Coerência Óptica da Retina (OCT), técnica de imagem usada para capturar seções transversais de alta resolução das retinas. É organizado em 3 diretórios (train, test, val). Cada um desses diretórios contém subdiretórios para cada categoria de imagem (NORMAL, CNV, DME,DRUSEN). São 84,495 imagens (JPEG) e 4 categorias:

* CNV (choroidal neovascularization) - Processo patológico que consiste da formação de novos vasos sanguíneos na COROIDE.

* DME (diabetic macular edema) - Ao longo do tempo, níveis glicêmicos altos podem levar a complicações vasculares em vários tecidos e órgãos, como no coração, sistema nervoso, rins, membros inferiores (pernas e pés) e inclusive nos olhos.

* DRUSEN - pequenos depósitos amarelos ou brancos na retina do olho ou na cabeça do nervo óptico. A presença de drusas é um dos sinais precoces mais comuns de degeneração macular relacionada à idade.

* NORMAL

> Labeled Optical Coherence Tomography (OCT) Images for Classification - Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, v2

## 2. Modelagem

Neste estudo, a classificação de OCT foi realizada com modelos de aprendizado profundo e algumas abordagens foram testadas até que se chegasse ao modelo de melhor performance:

2.1. Transfer Learning utilizando CNN VGG16 pré-treinada com a base de dados "imageNet": apresentou resultado, aproximadamente, 10% inferior ao do modelo final.

2.2. Transfer Learning utilizando CNN Xception pré-treinada com os pesos do dataset ImageNet: apresentou o melhor resultado.

2.3. Transfer Learning utilizando CNN EfficientNet B0 a B7 pré-treinadas com a base de dados "imageNet": apresentaram resultados muito inferiores ao do modelo final.

2.4. CNN VGG16 pré-treinada para extração de características das imagens:

   * CNN VGG16 pré-treinada para extração de vetor de características de cada imagem.
   
   * Redução de dimensionalidade do vetor de características da imagem, utilizando Principal Component Analysis (PCA).
   
   * Utilização do vetor para entrada em modelos de machine learning sklearn RandomForestClassifier, DecisionTreeClassifier, KNeighborsClassifier e LogisticRegression.<br>   
Essa abordagem mostrou-se muito ineficaz, pois como as imagens são muito semelhantes, todos os modelos apresentaram acurácia de 100%, porém, a matriz de confusão feita         utilizando os dados de teste apresentou um baixíssimo número de acertos para as classes.
   
## 3. Parâmetros utilizados no modelo final:

* Separação da base em Treino, Validação e Teste: Foi utilizado 25% da base para validação e 10% para teste.

* Data Augmentation - Aumento da quantidade de imagens, adicionando cópias ligeiramente modificadas de imagens já existentes e redimensionamento para o padrão de entrada da rede neural.

* Transfer Learning - CNN Xception pré-treinada com a base de dados "imageNet".

* Quantidade de Neurônios das Camadas Densas (Dense) - Foi utilizada somente uma camada densa com quatro neurônios (número de classes) para a classificação das imagens e função de ativação softmax.

* Otimizador - O melhor resultado foi obtido com o otimizador SGD, utilizando os parâmetros: Learning Rate = 0,045 / Decay = Learning Rate/n° de épocas = 0,1/30 / Momentum = 0,9.

* Indicador de Perda - Categorical crossentropy ou softmax loss, a rede neural foi treinada para emitir a probabilidade de a imagem pertencer a cada uma das quatros classes. Foram feitos testes utilizando sparce categorical crossentropy com resultados inferiores.

* Callbacks - Evitam o sobretreino da rede (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).

* Stratified K-Fold cross validation - As partições são feitas preservando a porcentagem de amostras para cada classe (estratificada). Foram utilizados 5 folds para validação do modelo final.

* Balanceamento da base - Não aumenta as amostras, mas atribui pesos a cada classe para evitar qualquer viés por meio de dados não balanceados (class_weight). O balanceamento da base melhorou a performance do modelo na inferência da classe DRUSEN, que possue o menor número de imagens.

## 4. Resultados

### Métricas de Treinamento (fold 5)

Representação gráfica da precisão do treinamento versus perda, para melhor compreensão do treinamento do modelo.
<table class="center">
   <tr>
      <td valign="top"><img src="https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/images/Training_and_validation_accuracy.JPG"/></td>
      <td valign="top"><img src="https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/images/Training_and_validation_loss.JPG"/></td>
    </tr>
</table>

### Avaliação do modelo utilizando os dados de teste (fold 5)

<table class="center">
   <tr>
      <td valign="top"><img src="https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/images/Matriz%20de%20confusao.JPG"/></td>
      <td valign="top"><img src="https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/images/Matriz%20de%20confusao%20normalizada.JPG"/></td>
    </tr>
</table>

### Relatório de métricas de performance do classificador utilizando os dados de teste (fold 5)

<p align="center">
  <img src="https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/images/classification_report.JPG">
</p>

### Avaliação do modelo utilizando Validação Cruzada (Stratified K-Fold)

Accuracy: De todas as imagens, quantas foram classificadas corretamente?
<p align="center">
  <img src="https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/images/accuracy.JPG">
</p>

Precision: Quantas imagens foram rotuladas para uma classe, são realmente dessa classe? O classificador está apresentando baixo desempenho ao inferir a classe DRUSEN para essa métrica.
<p align="center">
  <img src="https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/images/precision.JPG">
</p>

Recall: De todas as imagens de cada classe, quantas foram rotuladas corretamente? O classificador está apresentando um melhor desempenho ao inferir a classe DRUSEN para essa métrica. Para nosso conjunto de dados, podemos considerar que alcançar um melhor recall é mais importante do que obter uma alta precisão.
<p align="center">
  <img src="https://github.com/rrfsantos/Projeto-Redes-Neurais-OCT-Images/blob/main/images/recall.JPG">
</p>

## 5. Conclusão

Neste estudo, a classificação de OCT foi realizada com modelos de aprendizado profundo e algumas abordagens foram testadas até que se chegasse ao modelo de melhor performance. Na primeira etapa, os dados foram padronizados e, em seguida, usados como entrada para a CNN Xception pré-treinada com os pesos do dataset ImageNet. 

Para validação, foi utilizada a técnica de validação cruzada estratificada com 5 folds e a divisão aleatória do dataset em subsets de treino, validação e teste. As duas abordagens apresentaram resultados similares.

O desempenho do modelo foi medido utilizando as métricas: acurácia, precisão e recall. Sendo recall a mais importante para o nosso conjunto de dados, pois devemos considerar o diagnóstico errado prejudicial, principalmente a classificação de uma imagem com uma das três anomalias como uma imagem normal. Apresentou bons resultados para a classificação de imagens das classes NORMAL e CNV e o pior resultado para as imagens da classe DRUSEN. 

---

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*
