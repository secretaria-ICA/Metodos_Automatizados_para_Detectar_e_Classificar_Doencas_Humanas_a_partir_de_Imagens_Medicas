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

Foram utilizados as técnicas abaixo para a modelagem da problema:

* Data Augmentation - Aumento da quantidade de dados, adicionando cópias ligeiramente modificadas de dados já existentes ou dados sintéticos recém-criados a partir de dados existentes.
* Transfer Learning - RNN Xception pré-treinada com a base de dados "imageNet". Foram feitos testes utilizando as RNN VGG16 e EfficientNet B0 a B7, com resultados inferiores.
* Callbacks - Evitam o sobretreino da rede (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
* Stratified K-Fold cross validation - As partições são feitas preservando a porcentagem de amostras para cada classe (estratificada). Utilizada para validação do modelo final.
* Balanceamento da base (removido) - Inicialmente, foi feito o balanceamento da base para treinamento, atribuindo pesos a cada classe para evitar qualquer viés por meio de dados não balanceados. Durante os testes de validação do modelo, a utilização dessa técnica mostrou-se ineficaz, pois diminuiu a performance na predição utilizando os dados de teste.

Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 299, 299, 3) 0                                            
__________________________________________________________________________________________________
block1_conv1 (Conv2D)           (None, 149, 149, 32) 864         input_1[0][0]                    
__________________________________________________________________________________________________
block1_conv1_bn (BatchNormaliza (None, 149, 149, 32) 128         block1_conv1[0][0]               
__________________________________________________________________________________________________
block1_conv1_act (Activation)   (None, 149, 149, 32) 0           block1_conv1_bn[0][0]            
__________________________________________________________________________________________________
block1_conv2 (Conv2D)           (None, 147, 147, 64) 18432       block1_conv1_act[0][0]           
__________________________________________________________________________________________________
block1_conv2_bn (BatchNormaliza (None, 147, 147, 64) 256         block1_conv2[0][0]               
__________________________________________________________________________________________________
block1_conv2_act (Activation)   (None, 147, 147, 64) 0           block1_conv2_bn[0][0]            
__________________________________________________________________________________________________
block2_sepconv1 (SeparableConv2 (None, 147, 147, 128 8768        block1_conv2_act[0][0]           
__________________________________________________________________________________________________
block2_sepconv1_bn (BatchNormal (None, 147, 147, 128 512         block2_sepconv1[0][0]            
__________________________________________________________________________________________________
block2_sepconv2_act (Activation (None, 147, 147, 128 0           block2_sepconv1_bn[0][0]         
__________________________________________________________________________________________________
block2_sepconv2 (SeparableConv2 (None, 147, 147, 128 17536       block2_sepconv2_act[0][0]        
__________________________________________________________________________________________________
block2_sepconv2_bn (BatchNormal (None, 147, 147, 128 512         block2_sepconv2[0][0]            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 74, 74, 128)  8192        block1_conv2_act[0][0]           
__________________________________________________________________________________________________
block2_pool (MaxPooling2D)      (None, 74, 74, 128)  0           block2_sepconv2_bn[0][0]         
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 74, 74, 128)  512         conv2d[0][0]                     
__________________________________________________________________________________________________
add (Add)                       (None, 74, 74, 128)  0           block2_pool[0][0]                
                                                                 batch_normalization[0][0]        
__________________________________________________________________________________________________
block3_sepconv1_act (Activation (None, 74, 74, 128)  0           add[0][0]                        
__________________________________________________________________________________________________
block3_sepconv1 (SeparableConv2 (None, 74, 74, 256)  33920       block3_sepconv1_act[0][0]        
__________________________________________________________________________________________________
block3_sepconv1_bn (BatchNormal (None, 74, 74, 256)  1024        block3_sepconv1[0][0]            
__________________________________________________________________________________________________
block3_sepconv2_act (Activation (None, 74, 74, 256)  0           block3_sepconv1_bn[0][0]         
__________________________________________________________________________________________________
block3_sepconv2 (SeparableConv2 (None, 74, 74, 256)  67840       block3_sepconv2_act[0][0]        
__________________________________________________________________________________________________
block3_sepconv2_bn (BatchNormal (None, 74, 74, 256)  1024        block3_sepconv2[0][0]            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 37, 37, 256)  32768       add[0][0]                        
__________________________________________________________________________________________________
block3_pool (MaxPooling2D)      (None, 37, 37, 256)  0           block3_sepconv2_bn[0][0]         
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 37, 37, 256)  1024        conv2d_1[0][0]                   
__________________________________________________________________________________________________
add_1 (Add)                     (None, 37, 37, 256)  0           block3_pool[0][0]                
                                                                 batch_normalization_1[0][0]      
__________________________________________________________________________________________________
block4_sepconv1_act (Activation (None, 37, 37, 256)  0           add_1[0][0]                      
__________________________________________________________________________________________________
block4_sepconv1 (SeparableConv2 (None, 37, 37, 728)  188672      block4_sepconv1_act[0][0]        
__________________________________________________________________________________________________
block4_sepconv1_bn (BatchNormal (None, 37, 37, 728)  2912        block4_sepconv1[0][0]            
__________________________________________________________________________________________________
block4_sepconv2_act (Activation (None, 37, 37, 728)  0           block4_sepconv1_bn[0][0]         
__________________________________________________________________________________________________
block4_sepconv2 (SeparableConv2 (None, 37, 37, 728)  536536      block4_sepconv2_act[0][0]        
__________________________________________________________________________________________________
block4_sepconv2_bn (BatchNormal (None, 37, 37, 728)  2912        block4_sepconv2[0][0]            
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 19, 19, 728)  186368      add_1[0][0]                      
__________________________________________________________________________________________________
block4_pool (MaxPooling2D)      (None, 19, 19, 728)  0           block4_sepconv2_bn[0][0]         
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 19, 19, 728)  2912        conv2d_2[0][0]                   
__________________________________________________________________________________________________
add_2 (Add)                     (None, 19, 19, 728)  0           block4_pool[0][0]                
                                                                 batch_normalization_2[0][0]      
__________________________________________________________________________________________________
block5_sepconv1_act (Activation (None, 19, 19, 728)  0           add_2[0][0]                      
__________________________________________________________________________________________________
block5_sepconv1 (SeparableConv2 (None, 19, 19, 728)  536536      block5_sepconv1_act[0][0]        
__________________________________________________________________________________________________
block5_sepconv1_bn (BatchNormal (None, 19, 19, 728)  2912        block5_sepconv1[0][0]            
__________________________________________________________________________________________________
block5_sepconv2_act (Activation (None, 19, 19, 728)  0           block5_sepconv1_bn[0][0]         
__________________________________________________________________________________________________
block5_sepconv2 (SeparableConv2 (None, 19, 19, 728)  536536      block5_sepconv2_act[0][0]        
__________________________________________________________________________________________________
block5_sepconv2_bn (BatchNormal (None, 19, 19, 728)  2912        block5_sepconv2[0][0]            
__________________________________________________________________________________________________
block5_sepconv3_act (Activation (None, 19, 19, 728)  0           block5_sepconv2_bn[0][0]         
__________________________________________________________________________________________________
block5_sepconv3 (SeparableConv2 (None, 19, 19, 728)  536536      block5_sepconv3_act[0][0]        
__________________________________________________________________________________________________
block5_sepconv3_bn (BatchNormal (None, 19, 19, 728)  2912        block5_sepconv3[0][0]            
__________________________________________________________________________________________________
add_3 (Add)                     (None, 19, 19, 728)  0           block5_sepconv3_bn[0][0]         
                                                                 add_2[0][0]                      
__________________________________________________________________________________________________
block6_sepconv1_act (Activation (None, 19, 19, 728)  0           add_3[0][0]                      
__________________________________________________________________________________________________
block6_sepconv1 (SeparableConv2 (None, 19, 19, 728)  536536      block6_sepconv1_act[0][0]        
__________________________________________________________________________________________________
block6_sepconv1_bn (BatchNormal (None, 19, 19, 728)  2912        block6_sepconv1[0][0]            
__________________________________________________________________________________________________
block6_sepconv2_act (Activation (None, 19, 19, 728)  0           block6_sepconv1_bn[0][0]         
__________________________________________________________________________________________________
block6_sepconv2 (SeparableConv2 (None, 19, 19, 728)  536536      block6_sepconv2_act[0][0]        
__________________________________________________________________________________________________
block6_sepconv2_bn (BatchNormal (None, 19, 19, 728)  2912        block6_sepconv2[0][0]            
__________________________________________________________________________________________________
block6_sepconv3_act (Activation (None, 19, 19, 728)  0           block6_sepconv2_bn[0][0]         
__________________________________________________________________________________________________
block6_sepconv3 (SeparableConv2 (None, 19, 19, 728)  536536      block6_sepconv3_act[0][0]        
__________________________________________________________________________________________________
block6_sepconv3_bn (BatchNormal (None, 19, 19, 728)  2912        block6_sepconv3[0][0]            
__________________________________________________________________________________________________
add_4 (Add)                     (None, 19, 19, 728)  0           block6_sepconv3_bn[0][0]         
                                                                 add_3[0][0]                      
__________________________________________________________________________________________________
block7_sepconv1_act (Activation (None, 19, 19, 728)  0           add_4[0][0]                      
__________________________________________________________________________________________________
block7_sepconv1 (SeparableConv2 (None, 19, 19, 728)  536536      block7_sepconv1_act[0][0]        
__________________________________________________________________________________________________
block7_sepconv1_bn (BatchNormal (None, 19, 19, 728)  2912        block7_sepconv1[0][0]            
__________________________________________________________________________________________________
block7_sepconv2_act (Activation (None, 19, 19, 728)  0           block7_sepconv1_bn[0][0]         
__________________________________________________________________________________________________
block7_sepconv2 (SeparableConv2 (None, 19, 19, 728)  536536      block7_sepconv2_act[0][0]        
__________________________________________________________________________________________________
block7_sepconv2_bn (BatchNormal (None, 19, 19, 728)  2912        block7_sepconv2[0][0]            
__________________________________________________________________________________________________
block7_sepconv3_act (Activation (None, 19, 19, 728)  0           block7_sepconv2_bn[0][0]         
__________________________________________________________________________________________________
block7_sepconv3 (SeparableConv2 (None, 19, 19, 728)  536536      block7_sepconv3_act[0][0]        
__________________________________________________________________________________________________
block7_sepconv3_bn (BatchNormal (None, 19, 19, 728)  2912        block7_sepconv3[0][0]            
__________________________________________________________________________________________________
add_5 (Add)                     (None, 19, 19, 728)  0           block7_sepconv3_bn[0][0]         
                                                                 add_4[0][0]                      
__________________________________________________________________________________________________
block8_sepconv1_act (Activation (None, 19, 19, 728)  0           add_5[0][0]                      
__________________________________________________________________________________________________
block8_sepconv1 (SeparableConv2 (None, 19, 19, 728)  536536      block8_sepconv1_act[0][0]        
__________________________________________________________________________________________________
block8_sepconv1_bn (BatchNormal (None, 19, 19, 728)  2912        block8_sepconv1[0][0]            
__________________________________________________________________________________________________
block8_sepconv2_act (Activation (None, 19, 19, 728)  0           block8_sepconv1_bn[0][0]         
__________________________________________________________________________________________________
block8_sepconv2 (SeparableConv2 (None, 19, 19, 728)  536536      block8_sepconv2_act[0][0]        
__________________________________________________________________________________________________
block8_sepconv2_bn (BatchNormal (None, 19, 19, 728)  2912        block8_sepconv2[0][0]            
__________________________________________________________________________________________________
block8_sepconv3_act (Activation (None, 19, 19, 728)  0           block8_sepconv2_bn[0][0]         
__________________________________________________________________________________________________
block8_sepconv3 (SeparableConv2 (None, 19, 19, 728)  536536      block8_sepconv3_act[0][0]        
__________________________________________________________________________________________________
block8_sepconv3_bn (BatchNormal (None, 19, 19, 728)  2912        block8_sepconv3[0][0]            
__________________________________________________________________________________________________
add_6 (Add)                     (None, 19, 19, 728)  0           block8_sepconv3_bn[0][0]         
                                                                 add_5[0][0]                      
__________________________________________________________________________________________________
block9_sepconv1_act (Activation (None, 19, 19, 728)  0           add_6[0][0]                      
__________________________________________________________________________________________________
block9_sepconv1 (SeparableConv2 (None, 19, 19, 728)  536536      block9_sepconv1_act[0][0]        
__________________________________________________________________________________________________
block9_sepconv1_bn (BatchNormal (None, 19, 19, 728)  2912        block9_sepconv1[0][0]            
__________________________________________________________________________________________________
block9_sepconv2_act (Activation (None, 19, 19, 728)  0           block9_sepconv1_bn[0][0]         
__________________________________________________________________________________________________
block9_sepconv2 (SeparableConv2 (None, 19, 19, 728)  536536      block9_sepconv2_act[0][0]        
__________________________________________________________________________________________________
block9_sepconv2_bn (BatchNormal (None, 19, 19, 728)  2912        block9_sepconv2[0][0]            
__________________________________________________________________________________________________
block9_sepconv3_act (Activation (None, 19, 19, 728)  0           block9_sepconv2_bn[0][0]         
__________________________________________________________________________________________________
block9_sepconv3 (SeparableConv2 (None, 19, 19, 728)  536536      block9_sepconv3_act[0][0]        
__________________________________________________________________________________________________
block9_sepconv3_bn (BatchNormal (None, 19, 19, 728)  2912        block9_sepconv3[0][0]            
__________________________________________________________________________________________________
add_7 (Add)                     (None, 19, 19, 728)  0           block9_sepconv3_bn[0][0]         
                                                                 add_6[0][0]                      
__________________________________________________________________________________________________
block10_sepconv1_act (Activatio (None, 19, 19, 728)  0           add_7[0][0]                      
__________________________________________________________________________________________________
block10_sepconv1 (SeparableConv (None, 19, 19, 728)  536536      block10_sepconv1_act[0][0]       
__________________________________________________________________________________________________
block10_sepconv1_bn (BatchNorma (None, 19, 19, 728)  2912        block10_sepconv1[0][0]           
__________________________________________________________________________________________________
block10_sepconv2_act (Activatio (None, 19, 19, 728)  0           block10_sepconv1_bn[0][0]        
__________________________________________________________________________________________________
block10_sepconv2 (SeparableConv (None, 19, 19, 728)  536536      block10_sepconv2_act[0][0]       
__________________________________________________________________________________________________
block10_sepconv2_bn (BatchNorma (None, 19, 19, 728)  2912        block10_sepconv2[0][0]           
__________________________________________________________________________________________________
block10_sepconv3_act (Activatio (None, 19, 19, 728)  0           block10_sepconv2_bn[0][0]        
__________________________________________________________________________________________________
block10_sepconv3 (SeparableConv (None, 19, 19, 728)  536536      block10_sepconv3_act[0][0]       
__________________________________________________________________________________________________
block10_sepconv3_bn (BatchNorma (None, 19, 19, 728)  2912        block10_sepconv3[0][0]           
__________________________________________________________________________________________________
add_8 (Add)                     (None, 19, 19, 728)  0           block10_sepconv3_bn[0][0]        
                                                                 add_7[0][0]                      
__________________________________________________________________________________________________
block11_sepconv1_act (Activatio (None, 19, 19, 728)  0           add_8[0][0]                      
__________________________________________________________________________________________________
block11_sepconv1 (SeparableConv (None, 19, 19, 728)  536536      block11_sepconv1_act[0][0]       
__________________________________________________________________________________________________
block11_sepconv1_bn (BatchNorma (None, 19, 19, 728)  2912        block11_sepconv1[0][0]           
__________________________________________________________________________________________________
block11_sepconv2_act (Activatio (None, 19, 19, 728)  0           block11_sepconv1_bn[0][0]        
__________________________________________________________________________________________________
block11_sepconv2 (SeparableConv (None, 19, 19, 728)  536536      block11_sepconv2_act[0][0]       
__________________________________________________________________________________________________
block11_sepconv2_bn (BatchNorma (None, 19, 19, 728)  2912        block11_sepconv2[0][0]           
__________________________________________________________________________________________________
block11_sepconv3_act (Activatio (None, 19, 19, 728)  0           block11_sepconv2_bn[0][0]        
__________________________________________________________________________________________________
block11_sepconv3 (SeparableConv (None, 19, 19, 728)  536536      block11_sepconv3_act[0][0]       
__________________________________________________________________________________________________
block11_sepconv3_bn (BatchNorma (None, 19, 19, 728)  2912        block11_sepconv3[0][0]           
__________________________________________________________________________________________________
add_9 (Add)                     (None, 19, 19, 728)  0           block11_sepconv3_bn[0][0]        
                                                                 add_8[0][0]                      
__________________________________________________________________________________________________
block12_sepconv1_act (Activatio (None, 19, 19, 728)  0           add_9[0][0]                      
__________________________________________________________________________________________________
block12_sepconv1 (SeparableConv (None, 19, 19, 728)  536536      block12_sepconv1_act[0][0]       
__________________________________________________________________________________________________
block12_sepconv1_bn (BatchNorma (None, 19, 19, 728)  2912        block12_sepconv1[0][0]           
__________________________________________________________________________________________________
block12_sepconv2_act (Activatio (None, 19, 19, 728)  0           block12_sepconv1_bn[0][0]        
__________________________________________________________________________________________________
block12_sepconv2 (SeparableConv (None, 19, 19, 728)  536536      block12_sepconv2_act[0][0]       
__________________________________________________________________________________________________
block12_sepconv2_bn (BatchNorma (None, 19, 19, 728)  2912        block12_sepconv2[0][0]           
__________________________________________________________________________________________________
block12_sepconv3_act (Activatio (None, 19, 19, 728)  0           block12_sepconv2_bn[0][0]        
__________________________________________________________________________________________________
block12_sepconv3 (SeparableConv (None, 19, 19, 728)  536536      block12_sepconv3_act[0][0]       
__________________________________________________________________________________________________
block12_sepconv3_bn (BatchNorma (None, 19, 19, 728)  2912        block12_sepconv3[0][0]           
__________________________________________________________________________________________________
add_10 (Add)                    (None, 19, 19, 728)  0           block12_sepconv3_bn[0][0]        
                                                                 add_9[0][0]                      
__________________________________________________________________________________________________
block13_sepconv1_act (Activatio (None, 19, 19, 728)  0           add_10[0][0]                     
__________________________________________________________________________________________________
block13_sepconv1 (SeparableConv (None, 19, 19, 728)  536536      block13_sepconv1_act[0][0]       
__________________________________________________________________________________________________
block13_sepconv1_bn (BatchNorma (None, 19, 19, 728)  2912        block13_sepconv1[0][0]           
__________________________________________________________________________________________________
block13_sepconv2_act (Activatio (None, 19, 19, 728)  0           block13_sepconv1_bn[0][0]        
__________________________________________________________________________________________________
block13_sepconv2 (SeparableConv (None, 19, 19, 1024) 752024      block13_sepconv2_act[0][0]       
__________________________________________________________________________________________________
block13_sepconv2_bn (BatchNorma (None, 19, 19, 1024) 4096        block13_sepconv2[0][0]           
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 10, 10, 1024) 745472      add_10[0][0]                     
__________________________________________________________________________________________________
block13_pool (MaxPooling2D)     (None, 10, 10, 1024) 0           block13_sepconv2_bn[0][0]        
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 10, 10, 1024) 4096        conv2d_3[0][0]                   
__________________________________________________________________________________________________
add_11 (Add)                    (None, 10, 10, 1024) 0           block13_pool[0][0]               
                                                                 batch_normalization_3[0][0]      
__________________________________________________________________________________________________
block14_sepconv1 (SeparableConv (None, 10, 10, 1536) 1582080     add_11[0][0]                     
__________________________________________________________________________________________________
block14_sepconv1_bn (BatchNorma (None, 10, 10, 1536) 6144        block14_sepconv1[0][0]           
__________________________________________________________________________________________________
block14_sepconv1_act (Activatio (None, 10, 10, 1536) 0           block14_sepconv1_bn[0][0]        
__________________________________________________________________________________________________
block14_sepconv2 (SeparableConv (None, 10, 10, 2048) 3159552     block14_sepconv1_act[0][0]       
__________________________________________________________________________________________________
block14_sepconv2_bn (BatchNorma (None, 10, 10, 2048) 8192        block14_sepconv2[0][0]           
__________________________________________________________________________________________________
block14_sepconv2_act (Activatio (None, 10, 10, 2048) 0           block14_sepconv2_bn[0][0]        
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 2, 2, 2048)   0           block14_sepconv2_act[0][0]       
__________________________________________________________________________________________________
flatten (Flatten)               (None, 8192)         0           average_pooling2d_1[0][0]        
__________________________________________________________________________________________________
dense (Dense)                   (None, 4)            32772       flatten[0][0]                    
==================================================================================================
Total params: 20,894,252
Trainable params: 32,772
Non-trainable params: 20,861,480
__________________________________________________________________________________________________

