# Métodos automatizados para detectar e classificar doenças humanas a partir de imagens médicas

Projeto de conclusão do curso [BI-MASTER](https://ica.puc-rio.ai/bi-master/) e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

**Aluna**: [Renata Regina da Fonseca Santos](https://github.com/rrfsantos)

**Orientadora**: Professora Evelyn Conceição

* [Link para o código]

## Objetivo

Utilização de modelos de inteligência artificial pré-treinados como alternativa para predição de diagnóstico em exames de imagem. 

### Itens do trabalho:

* Análise exploratória dos dados
* Tratamento dos dados para uso na rede neural
* Avaliação da melhor configuração da rede neural


## Descrição dados
O dataset é composto por imagens de Tomografia de Coerência Óptica da Retina (OCT), técnica de imagem usada para capturar seções transversais de alta resolução das retinas. É organizado em 3 diretórios (train, test, val). Cada um desses diretórios contém subdiretórios para cada categoria de imagem (NORMAL, CNV, DME,DRUSEN). São 84,495 imagens (JPEG) e 4 categorias:

* CNV (choroidal neovascularization) - Processo patológico que consiste da formação de novos vasos sanguíneos na COROIDE.
* DME (diabetic macular edema) - Ao longo do tempo, níveis glicêmicos altos podem levar a complicações vasculares em vários tecidos e órgãos, como no coração, sistema nervoso, rins, membros inferiores (pernas e pés) e inclusive nos olhos.
* DRUSEN - pequenos depósitos amarelos ou brancos na retina do olho ou na cabeça do nervo óptico. A presença de drusas é um dos sinais precoces mais comuns de degeneração macular relacionada à idade.
* NORMAL

> Labeled Optical Coherence Tomography (OCT) Images for Classification - Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, v2

## Solução do Problema

Este trabalho baseou-se na API Keras para a construção da rede neural e inferência dos diagnósticos.

### Técnicas utilizadas 

* Data Augmentation para aumento da quantidade de dados, adicionando cópias ligeiramente modificadas de dados já existentes ou dados sintéticos recém-criados a partir de dados existentes
* Rede neural Xception pré-treinada com a base de dados "imageNet"
* Callbacks para evitar o sobretreino da rede (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
