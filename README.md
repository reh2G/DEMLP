# DEMLP

Pipeline de classificação de imagens médicas para **detecção de cálculos renais** usando uma CNN treinada (Conv4) com validação cruzada e análise de interpretabilidade via Grad-CAM.

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Dataset](#2-dataset)
3. [Arquitetura do Modelo](#3-arquitetura-do-modelo)
4. [Pipeline de Execução](#4-pipeline-de-execução)
5. [Estratégias de Dados](#5-estratégias-de-dados)
6. [Validação Cruzada](#6-validação-cruzada)
7. [Saídas do Experimento](#7-saídas-do-experimento)
8. [Estrutura do Projeto](#8-estrutura-do-projeto)
9. [Dependências](#9-dependências)
10. [Ambiente e Execução](#10-ambiente-e-execução)
11. [Parâmetros Configuráveis](#11-parâmetros-configuráveis)

---

## 1. Visão Geral

Este projeto implementa um pipeline completo de aprendizado profundo para classificação binária de imagens médicas, distinguindo rins **saudáveis** (*healthy*) de rins com **cálculos renais** (*stone*). O pipeline avalia três diferentes estratégias de pré-processamento de dados — padrão, com data augmentation e com subamostragem — usando validação cruzada estratificada por grupos para garantir que imagens similares (de um mesmo paciente/exame) nunca vazem entre treino e teste.

**Problema:** Classificação binária de imagens de ultrassom renal  
**Classes:** `healthy` (0) · `stone` (1)  
**Abordagem:** CNN customizada (Conv4) + 5-Fold Cross-Validation + Grad-CAM

---

## 2. Dataset

O dataset é composto por imagens de ultrassom renal organizadas em subdiretórios por classe, dentro do diretório `data/combined/`:

```
data/
└── combined/
    ├── healthy/   # imagens de rins saudáveis
    └── stone/     # imagens de rins com cálculos
```

### Tamanho original

| Classe     | Quantidade |
|------------|------------|
| `healthy`  | 2.079 imagens |
| `stone`    | 1.304 imagens |
| **Total**  | **3.383 imagens** |

### Pré-processamento de leitura

- Leitura em **escala de cinza** (`cv2.IMREAD_GRAYSCALE`)
- **Redimensionamento** para `224 × 224` pixels
- **Normalização** para `[0, 1]` (divisão por 255)
- **Adição de dimensão de canal** para compatibilidade com a CNN: `(224, 224, 1)`
- **Codificação one-hot** dos rótulos para `categorical_crossentropy`

### Agrupamento por Similaridade

Para evitar **data leakage** entre folds de imagens do mesmo paciente/exame, as imagens são agrupadas usando o índice de similaridade estrutural (**SSIM**). Imagens consecutivas com `SSIM ≥ 0.75` são atribuídas ao mesmo grupo. A divisão entre treino, validação e teste é feita **no nível de grupos**, e não de imagens individuais.

| Parâmetro | Valor |
|-----------|-------|
| Threshold de similaridade (SSIM) | `0.75` |
| Comparação | Imagens consecutivas (ordenação numérica) |

---

## 3. Arquitetura do Modelo

O modelo utilizado é uma **CNN customizada denominada Conv4**, com 4 blocos convolucionais seguidos de camadas densas com regularização.

### Topologia

| Camada | Tipo | Filtros/Unidades | Kernel/Pool | Ativação | Extras |
|--------|------|-----------------|-------------|----------|--------|
| Input | — | — | `224×224×1` | — | — |
| Conv2D | Convolucional | 16 | `3×3` | ReLU | — |
| MaxPooling2D | Pooling | — | `2×2` | — | — |
| Conv2D | Convolucional | 32 | `3×3` | ReLU | — |
| MaxPooling2D | Pooling | — | `2×2` | — | — |
| Conv2D | Convolucional | 64 | `3×3` | ReLU | — |
| MaxPooling2D | Pooling | — | `2×2` | — | — |
| Conv2D (`last_conv`) | Convolucional | 64 | `3×3` | ReLU | Anchor para Grad-CAM |
| MaxPooling2D | Pooling | — | `2×2` | — | — |
| Flatten | — | — | — | — | — |
| Dense | Totalmente conectada | 150 | — | ReLU | L2 (0.01) |
| Dropout | Regularização | — | — | — | taxa: 0.25 |
| Dense | Totalmente conectada | 100 | — | ReLU | L2 (0.01) |
| Dropout | Regularização | — | — | — | taxa: 0.25 |
| Dense (Output) | Classificação | 2 | — | Softmax | — |

### Configurações de Treinamento

| Parâmetro | Valor |
|-----------|-------|
| Otimizador | Adam (padrão) |
| Função de perda | `categorical_crossentropy` |
| Métrica de monitoramento | `accuracy` |
| EarlyStopping (monitor) | `val_loss` |
| EarlyStopping (patience) | 20 épocas |
| EarlyStopping (restore best weights) | `True` |
| Épocas máximas | 200 |
| Batch size | 32 |

---

## 4. Pipeline de Execução

O experimento segue o fluxo abaixo:

```
1. Ler o dataset
      └─ Leitura das imagens por classe (grayscale, 224×224)

2. Agrupar por similaridade (SSIM ≥ 0.75)
      └─ Imagens similares → mesmo grupo → nunca separadas entre treino e teste

3. Separar Holdout de Teste (20% dos grupos)
      └─ Feito UMA vez, antes de qualquer estratégia
      └─ Usado como teste final de todas as estratégias

4. Para cada estratégia de dados [default | augmentation | small-data]:
   │
   ├─ 5. Validação Cruzada (KFold, 5 folds, nível de grupos)
   │       └─ 80% dos grupos de treino/val → divididos em 5 folds
   │
   └─ Para cada fold:
         ├─ 6. Aplicar estratégia de dados no treino
         ├─ 7. Preparar os dados (normalizar, one-hot, canal)
         ├─ 8. Treinar o modelo (Conv4 + EarlyStopping)
         ├─ 9. Avaliar métricas (acc, prec, recall, F1, matriz de confusão)
         ├─ 10. Salvar modelo .keras
         └─ 11. Gerar Grad-CAMs por fold

5. Ao final de cada estratégia: gerar visualizações globais
```

> [!IMPORTANT]
> O **holdout de teste (20%)** é separado **antes** da validação cruzada e é **o mesmo para todas as estratégias**, garantindo comparação justa. As estratégias de dados (augmentation, undersampling) são aplicadas **apenas nos dados de treino de cada fold**, nunca na validação ou no teste.

---

## 5. Estratégias de Dados

São executadas 3 estratégias sequencialmente para comparação:

### 5.1 Default (Padrão)

Sem balanceamento. O modelo é treinado com a distribuição original do dataset, que apresenta desbalanceamento de classes.

| Classe     | Quantidade (treino approx.) |
|------------|----------------------------|
| `healthy`  | ~1.663 imagens |
| `stone`    | ~1.043 imagens |

### 5.2 Data Augmentation

A classe minoritária (`stone`) é aumentada artificialmente com transformações para igualar a quantidade da classe majoritária (`healthy`). A augmentação é aplicada **por fold**, apenas sobre os dados de treino.

**Operações aplicadas:**

| Operação | Parâmetro |
|----------|-----------|
| Rotação | até `±10°` |
| Deslocamento horizontal | até `2%` da largura |
| Deslocamento vertical | até `2%` da altura |
| Ruído aditivo uniforme | amplitude `±2%` (i.e., `±0.02 × 255`) |
| Zoom | até `2%` |
| Espelhamento horizontal | Ativado |
| Modo de preenchimento | `nearest` |

| Classe     | Quantidade após augmentation |
|------------|------------------------------|
| `healthy`  | ~1.663 imagens (sem alteração) |
| `stone`    | ~1.663 imagens (balanceado) |

> [!NOTE]
> Amostras augmentadas do primeiro fold são salvas em `output/test_N/<estratégia>/augmented_samples/` para inspeção visual.

### 5.3 Small-data (Subamostragem)

A classe majoritária (`healthy`) é reduzida aleatoriamente para igualar a quantidade da classe minoritária (`stone`). Nenhuma imagem nova é gerada.

| Classe     | Quantidade após subamostragem |
|------------|-------------------------------|
| `healthy`  | ~1.043 imagens (reduzido) |
| `stone`    | ~1.043 imagens (sem alteração) |

> [!NOTE]
> As imagens excluídas no primeiro fold são registradas em `output/test_N/<estratégia>/excluded_images.txt`.

---

## 6. Validação Cruzada

- **Tipo:** K-Fold (`K = 5`), embaralhado
- **Nível de divisão:** Grupos (não imagens individuais), prevenindo data leakage
- **Semente aleatória:** `random_state=53` (reprodutibilidade)
- **Estratificação:** Por grupos de imagens similares

### Divisão dos dados

```
Total de imagens
├── 80% → Treino/Validação (grupos)
│    └── 5-Fold CV (por grupo)
│         ├── Fold 1: 80% treino | 20% val
│         ├── Fold 2: 80% treino | 20% val
│         ├── ...
│         └── Fold 5: 80% treino | 20% val
└── 20% → Teste (holdout, fixo para todas as estratégias)
```

---

## 7. Saídas do Experimento

Cada execução gera um diretório auto-incrementado `output/test_N/`, contendo:

```
output/
└── test_N/
    ├── config.txt                  # Configurações do experimento
    ├── detected_groups.txt         # Grupos de imagens similares detectados
    │
    ├── default/
    │   ├── fold_1/
    │   │   ├── model.keras                    # Melhor modelo (val_loss)
    │   │   ├── metrics.txt                    # Métricas do fold
    │   │   ├── groups.txt                     # Grupos usados (train/val/test)
    │   │   ├── gradcam_healthy_1.jpg          # Grad-CAM: healthy (aleatório)
    │   │   ├── gradcam_healthy_2.jpg
    │   │   ├── gradcam_healthy_3.jpg
    │   │   ├── gradcam_stone_1.jpg            # Grad-CAM: stone (aleatório)
    │   │   ├── gradcam_stone_2.jpg
    │   │   └── gradcam_stone_3.jpg
    │   ├── fold_2/ ... fold_5/
    │   ├── learning_curves.png               # Curvas de aprendizado (média ± desvio)
    │   ├── confusion_matrix.png              # Matriz de confusão cumulativa
    │   ├── classification_report.png         # Gráfico de barras agrupado
    │   ├── roc_curve.png                     # Curva ROC com IC
    │   ├── pr_curve.png                      # Curva Precisão-Recall
    │   ├── metrics_boxplot.png               # Boxplot das métricas por fold
    │   ├── worst_errors_fp.png               # Piores Falsos Positivos + Grad-CAM
    │   └── worst_errors_fn.png               # Piores Falsos Negativos + Grad-CAM
    │
    ├── augmentation/
    │   ├── augmented_samples/                # Exemplos de imagens augmentadas
    │   └── [mesma estrutura do default]
    │
    └── small-data/
        ├── excluded_images.txt               # Imagens excluídas pelo undersampling
        └── [mesma estrutura do default]
```

### Saídas por Fold

| Arquivo | Descrição |
|---------|-----------|
| `model.keras` | Melhor modelo salvo (monitorado por `val_loss`) |
| `metrics.txt` | Acurácia, Precisão, Recall, F1-Score e Matriz de Confusão |
| `groups.txt` | IDs dos grupos usados em Treino, Validação e Teste |
| `gradcam_healthy_[1-3].jpg` | Grad-CAM de 3 imagens *healthy* aleatórias da validação |
| `gradcam_stone_[1-3].jpg` | Grad-CAM de 3 imagens *stone* aleatórias da validação |

### Saídas por Estratégia (ao final dos 5 folds)

| Arquivo | Descrição |
|---------|-----------|
| `learning_curves.png` | Curvas de loss e acurácia (média ± desvio padrão dos folds) |
| `confusion_matrix.png` | Matriz de confusão cumulativa (soma de todos os folds) |
| `classification_report.png` | Gráfico de barras agrupado: Precision, Recall, F1 por classe e médias |
| `roc_curve.png` | Curva ROC com curva média, AUC ± desvio padrão e intervalo de confiança |
| `pr_curve.png` | Curva Precisão-Recall com AUC e baseline (no-skill) |
| `metrics_boxplot.png` | Boxplot com swarmplot das métricas por fold (acc, prec, recall, F1) |
| `worst_errors_fp.png` | Top-3 Falsos Positivos com maior confiança + Grad-CAM |
| `worst_errors_fn.png` | Top-3 Falsos Negativos com menor confiança + Grad-CAM |

### Grad-CAM

O Grad-CAM (*Gradient-weighted Class Activation Mapping*) é gerado a partir da **última camada convolucional** (`last_conv`, Conv2D com 64 filtros). O mapa de calor é sobreposto à imagem original usando o colormap **viridis**, com intensidade controlada por `alpha = 0.4`. Cada imagem exibe:
- Predição do modelo e confiança (%)
- Diagnóstico real (verde = correto, vermelho = incorreto)
- Barra de cores de intensidade

---

## 8. Estrutura do Projeto

```
DEMLP/
├── main.py                        # Ponto de entrada do pipeline
├── pyproject.toml                 # Dependências e metadados do projeto
├── setup.sh                       # Configuração do ambiente GPU (Linux/NVIDIA)
│
├── src/
│   ├── dataset/
│   │   ├── read_dataset.py        # Leitura e carregamento das imagens
│   │   ├── groups_dataset.py      # Agrupamento por similaridade (SSIM)
│   │   ├── strategies_dataset.py  # Data augmentation e undersampling
│   │   ├── prepare_dataset.py     # Normalização, canal, one-hot encoding
│   │   └── utils_dataset.py       # Utilitários (ordenação, diretório de saída)
│   │
│   ├── model/
│   │   ├── define_model.py        # Arquitetura Conv4
│   │   ├── run_model.py           # Orquestração do experimento (folds + estratégias)
│   │   └── utils_model.py         # Salvamento de configs, métricas, Grad-CAM e amostras
│   │
│   └── visual/
│       ├── grad_cam.py            # Cálculo do heatmap Grad-CAM (GradientTape)
│       ├── plot_metrics.py        # Curvas de aprendizado, ROC, PR, boxplot, relatório
│       └── plot_errors.py         # Análise dos piores erros com Grad-CAM
│
├── data/
│   └── combined/
│       ├── healthy/               # Imagens de rins saudáveis
│       └── stone/                 # Imagens de rins com cálculos
│
└── output/
    └── test_N/                    # Gerado automaticamente por execução
```

---

## 9. Dependências

| Biblioteca | Versão mínima | Uso |
|------------|--------------|-----|
| `tensorflow[and-cuda]` | 2.21.0 | CNN, treinamento, GradientTape (Linux + GPU) |
| `tensorflow` | 2.21.0 | CNN, treinamento, GradientTape (outros SOs) |
| `numpy` | 2.4.4 | Operações matriciais |
| `opencv-python` | 4.13.0.92 | Leitura e redimensionamento de imagens |
| `scikit-image` | 0.26.0 | Cálculo de SSIM para agrupamento |
| `scikit-learn` | 1.8.0 | KFold, métricas, ROC/PR curves |
| `matplotlib` | 3.10.9 | Geração de todos os gráficos |
| `seaborn` | 0.13.2 | Heatmap (confusion matrix), boxplot, barplot |

**Python:** `>= 3.13`  
**Gerenciador de pacotes:** [`uv`](https://github.com/astral-sh/uv)

---

## 10. Ambiente e Execução

Este projeto utiliza o `uv` para facilitar o gerenciamento de dependências e a execução do código.

Caso você ainda não tenha o `uv` instalado, consulte a documentação oficial para instalação:
https://github.com/astral-sh/uv

### 1. Instalar as dependências

```bash
uv sync
```

### 2. Configurar o ambiente (Caso esteja usando Linux com GPU NVIDIA)

Necessário para uso de GPU durante o treinamento do modelo.

Este projeto utiliza TensorFlow com suporte a GPU via CUDA. As bibliotecas CUDA são instaladas como pacotes Python dentro do ambiente virtual, mas o sistema operacional precisa ser informado de onde encontrá-las.

O script abaixo configura automaticamente esse ambiente para a sua máquina:

```bash
./setup.sh
```

Caso ocorra um erro de permissão ao executar o script, rode:

```bash
chmod +x setup.sh
```

e tente novamente.

> [!IMPORTANT]
> Após rodar o `setup.sh`, abra um novo terminal no VS Code para que as configurações entrem em vigor.

### 3. Executar o projeto

```bash
uv run main.py
```

> [!NOTE]
> * O `setup.sh` precisa ser executado apenas **uma vez por máquina**, ou novamente caso o ambiente virtual seja recriado.
> * O arquivo `.vscode/settings.json` é gerado localmente pelo `setup.sh` e não é versionado no repositório. Cada colaborador terá sua própria configuração com os caminhos corretos para sua máquina.
> * O projeto foi desenvolvido e testado em **Linux com GPU NVIDIA**. Em outros sistemas operacionais, a configuração da GPU pode variar.

---

## 11. Parâmetros Configuráveis

Os parâmetros do experimento são definidos no início de `main.py`:

| Parâmetro | Variável | Valor padrão | Descrição |
|-----------|----------|-------------|-----------|
| Threshold SSIM | `SIMILARITY` | `0.75` | Limiar para considerar duas imagens do mesmo grupo |
| Número de folds | `KFOLD` | `5` | Quantidade de folds na validação cruzada |
| Épocas máximas | `EPOCHS` | `200` | Máximo de épocas por fold |
| Batch size | `BATCH_SIZE` | `32` | Tamanho do batch durante o treinamento |
| Paciência EarlyStopping | `EARLY_STOPPING_PATIENCE` | `20` | Épocas sem melhora no `val_loss` antes de parar |
| Nome do dataset | `name` | `"combined"` | Subdiretório dentro de `data/` |
| Modo debug | `DEBUG` | `False` | Exibe detalhes dos grupos detectados |
