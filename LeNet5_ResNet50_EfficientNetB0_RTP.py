# -------------------------------------------------------
# Bibliotecas necessárias para processamento de imagens,
# construção dos modelos e métricas de avaliação
# -------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_effnet
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# -------------------------------------------------------
# Configurações gerais do experimento
# -------------------------------------------------------
img_size = (224, 224)       # Tamanho das imagens a serem redimensionadas
batch_size = 32             # Tamanho dos lotes de treinamento
num_classes = 2             # Número de classes do problema (doente/saudável)

# -------------------------------------------------------
# Função para carregar e preparar os dados com o preprocessamento adequado
# -------------------------------------------------------
def prepare_data(folder_1, folder_2, img_size=(224, 224), batch_size=32, num_classes=2, model_type="lenet"):
    """
    Carrega imagens de duas pastas (duas classes), aplica preprocessamento
    específico para cada arquitetura (LeNet, ResNet ou EfficientNet), divide os
    dados em treino/validação/teste e retorna tf.data.Datasets.
    """
    
    # Seleciona o pré-processamento dependendo do modelo
    if model_type == "resnet":
        preprocess = preprocess_resnet
    elif model_type == "efficientnet":
        preprocess = preprocess_effnet
    else:
        preprocess = lambda x: x / 255.0  # Normalização padrão para LeNet

    def load_images(folder, label):
        """Carrega imagens de uma pasta e aplica o pré-processamento."""
        images, labels = [], []
        for file in tf.io.gfile.listdir(folder):
            filepath = f"{folder}/{file}"
            if filepath.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = tf.keras.utils.load_img(filepath, target_size=img_size)
                img_array = tf.keras.utils.img_to_array(img)
                img_array = preprocess(img_array)
                images.append(img_array)
                labels.append(label)
        return images, labels

    # Carrega imagens das duas classes
    images_1, labels_1 = load_images(folder_1, label=0)
    images_2, labels_2 = load_images(folder_2, label=1)

    # Combina os dados em um único conjunto
    X = np.array(images_1 + images_2)
    y = np.array(labels_1 + labels_2)
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    # Divide o dataset: 70% treino, 10% validação, 20% teste
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)

    # Converte para tf.data.Dataset para treinamento eficiente
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

# -------------------------------------------------------
# Função para plotar o histórico de treinamento (perda e acurácia)
# -------------------------------------------------------
def plot_training_history(history, model_name, epochs):
    """Gera gráficos de perda e acurácia para treino e validação."""
    plt.figure(figsize=(12, 6))

    # Curva de perda
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Perda ({model_name})')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()

    # Curva de acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title(f'Acurácia ({model_name})')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{epochs}_{model_name}.png')

# -------------------------------------------------------
# Função para treinar o modelo e medir métricas e tempo
# -------------------------------------------------------
def train_and_measure(model, train_data, test_data, val_data, epochs, model_name="Modelo"):
    """
    Treina o modelo, calcula métricas de desempenho, tempo de treinamento
    e tempo médio de inferência por imagem.
    """

    print('Model:', model_name)

    # Compilação do modelo
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=CategoricalCrossentropy(), metrics=["accuracy"])

    # Medição do tempo de treinamento
    start_time = time.time()
    history = model.fit(train_data, validation_data=val_data, epochs=epochs, verbose=1)
    training_time = time.time() - start_time

    # Medição do tempo de inferência
    start_time = time.time()
    y_pred, y_true = [], []
    for images, labels in test_data:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    inference_time = (time.time() - start_time) / len(y_pred)

    # Gera gráficos do treinamento
    plot_training_history(history, model_name, epochs)

    # Cálculo das métricas finais
    metrics = {
        "Loss": history.history["val_loss"][-1],
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1-Score": f1_score(y_true, y_pred, average="weighted"),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "Training Time (s)": training_time,
        "Inference Time (ms)": inference_time * 1000
    }

    return metrics

# -------------------------------------------------------
# Definição dos modelos utilizados
# -------------------------------------------------------

def create_lenet():
    """Implementação da arquitetura clássica LeNet-5."""
    return Sequential([
        Conv2D(32, (5, 5), activation="relu", input_shape=img_size + (3,)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (5, 5), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(120, activation="relu"),
        Dense(84, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

def create_resnet():
    """Modelo baseado na ResNet50 pré-treinada no ImageNet."""
    base = ResNet50(include_top=False, weights="imagenet", input_shape=img_size + (3,))

    # Estratégia de Fine-Tuning: congela todas menos as últimas 30 camadas
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    return Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

def create_efficientnet():
    """Modelo baseado no EfficientNetB0 com Fine-Tuning parcial."""
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=img_size + (3,))

    # Congela parte das camadas para acelerar e estabilizar o treinamento
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    return Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

# -------------------------------------------------------
# Código principal (Main)
# -------------------------------------------------------

# Diretórios das classes (doentes e saudáveis)
folder_doentes = "dataset_caupi/cercospora"
folder_saudaveis = "dataset_caupi/saudaveis"

# Lista de épocas a serem testadas para comparar desempenho
epochs_list = [5, 25, 50]

# Prepara datasets para cada modelo
train_lenet, val_lenet, test_lenet = prepare_data(folder_doentes, folder_saudaveis, img_size, batch_size, model_type="lenet")
train_resnet, val_resnet, test_resnet = prepare_data(folder_doentes, folder_saudaveis, img_size, batch_size, model_type="resnet")
train_eff, val_eff, test_eff = prepare_data(folder_doentes, folder_saudaveis, img_size, batch_size, model_type="efficientnet")

# Loop para avaliar os modelos com diferentes números de épocas
for ep in epochs_list:
    models = {
        "LeNet": create_lenet(),
        "ResNet50": create_resnet(),
        "EfficientNetB0": create_efficientnet()
    }

    datasets = {
        "LeNet": (train_lenet, val_lenet, test_lenet),
        "ResNet50": (train_resnet, val_resnet, test_resnet),
        "EfficientNetB0": (train_eff, val_eff, test_eff)
    }

    results = {}
    for name, model in models.items():
        tr, val, ts = datasets[name]
        results[name] = train_and_measure(model, tr, ts, val, ep, name)

    print(10 * '------')
    print('Epochs:', ep)
    results_df = pd.DataFrame.from_dict(results, orient="index")
    print(results_df)
