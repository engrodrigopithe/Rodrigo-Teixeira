# Classificação de Doenças em Folhas de Feijão-Caupi (Cercospora e CABMV)

Este repositório contém o script utilizado no projeto de pesquisa de mestrado para classificação de imagens de folhas de **feijão-caupi** afetadas por **Cercospora** e **CABMV**, empregando técnicas de **Deep Learning**. O objetivo é comparar diferentes arquiteturas de redes neurais convolucionais e analisar seu desempenho em um cenário real de diagnóstico automatizado.

---

## 📌 Objetivo do Projeto
Desenvolver, treinar e avaliar modelos de **Visão Computacional** capazes de identificar automaticamente sintomas de **Cercospora** e **CABMV** em imagens de folhas de feijão-caupi, apoiando pesquisas na área de agricultura digital e fitopatologia.

---

## 🧠 Modelos Utilizados
O script implementa e compara três arquiteturas:

### **1. LeNet-5**  
- Arquitetura clássica de CNN.
- Treinada totalmente do zero.

### **2. ResNet50**  
- Pré-treinada no ImageNet.
- Fine-Tuning das últimas 30 camadas.
- Ideal para extração profunda de características.

### **3. EfficientNetB0**  
- Pré-treinada no ImageNet.
- Fine-Tuning parcial das últimas camadas.
- Modelo eficiente e leve com ótimo desempenho.

---

## 📁 Estrutura do Script
O código é composto pelos seguintes módulos e funções:

### **🔹 Carregamento e preparação dos dados** (`prepare_data`)
- Carrega imagens de duas pastas (cercospora / saudáveis).
- Aplica preprocessamento adequado para cada modelo.
- Divide automaticamente em treino (70%), validação (10%) e teste (20%).
- Gera datasets otimizados com `tf.data.Dataset`.

### **🔹 Construção dos modelos**
- `create_lenet()`
- `create_resnet()`
- `create_efficientnet()`

Cada função prepara a arquitetura e retorna o modelo compilado.

### **🔹 Treinamento e métricas** (`train_and_measure`)
- Realiza o treinamento.
- Mede tempo de treinamento.
- Mede tempo médio de inferência por imagem.
- Calcula métricas:
  - Acurácia
  - Precisão
  - Recall
  - F1-Score
  - Kappa

### **🔹 Geração de gráficos** (`plot_training_history`)
- Gráficos automáticos de perda e acurácia (treino/validação).
- Exportados como PNG.

---

## 🧪 Processo Experimental
O script treina cada modelo com as seguintes quantidades de épocas:

- **5 épocas**
- **25 épocas**
- **50 épocas**

Para cada valor, são gerados resultados completos e tabelas comparativas.

---

## 📊 Métricas Geradas
O experimento registra:
- **Loss de validação**
- **Acurácia**
- **Precisão (weighted)**
- **Recall (weighted)**
- **F1-Score (weighted)**
- **Coeficiente Kappa**
- **Tempo de treinamento (s)**
- **Tempo de inferência por imagem (ms)**

---

## 📂 Estrutura Esperada do Dataset
```
dataset_caupi/
│
├── cercospora/
│     ├── img1.jpg
│     ├── img2.jpg
│     └── ...
│
└── saudaveis/
      ├── img1.jpg
      ├── img2.jpg
      └── ...
```

---

## ▶️ Como Executar
### **Pré-requisitos**
- Python 3.9+
- TensorFlow 2.x
- NumPy, Pandas, Scikit-Learn, Matplotlib

Instale com:
```
pip install tensorflow numpy pandas scikit-learn matplotlib
```

### **Executar o script**
```
python nome_do_script.py
```

---

## 📘 Saídas do Código
O script gera automaticamente:
- Tabelas comparativas de métricas.
- Gráficos de perda e acurácia.
- Logs detalhados no terminal.

---

## 🎓 Sobre o Projeto
Este código foi desenvolvido como parte do **projeto de mestrado**, cujo objetivo é investigar modelos de deep learning aplicados à detecção de doenças em plantas, com foco no feijão-caupi. A pesquisa busca contribuir para soluções de agricultura de precisão, auxiliando produtores, pesquisadores e sistemas de monitoramento.

---

## 📄 Licença
Este projeto pode ser utilizado para fins acadêmicos e de pesquisa. Para usos comerciais, consulte o autor.

---

## ✉️ Contato
Para dúvidas ou sugestões:
- **Autor:** Rodrigo Teixeira Pereira
- **Instituição:** IFPI / PPGEE

---

Se desejar incluir imagens de resultados, gráficos, benchmarks ou complementar o README com referências bibliográficas, posso adicionar também.
