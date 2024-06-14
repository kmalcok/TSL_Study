import gc
import glob
import os
from random import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image

# GPU kullanımı için ayar
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Bellek temizleme
def clear_memory():
    gc.collect()
    tf.keras.backend.clear_session()

# Sabitler
IMG_SIZE_ALEXNET = 200
IMG_SIZE_MOBILENET = 224
IMG_SIZE_VGG16 = 224
BATCH_SIZE = 64
EPOCHS = 50
VALIDATION_IMAGE_IDS = [2, 6, 12, 16, 22, 26, 32, 36, 42, 46, 52, 56, 62, 66, 72, 76, 82, 86, 92, 96]
TEST_IMAGE_IDS = [1, 5, 11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61, 65, 71, 75, 81, 85, 91, 95]
DATASET_PATH = "C:/TSL Finger Spelling/Images"

# Etiketleme Fonksiyonu
def label_img(name):
    labels = [0 for _ in range(26)]
    char_code = ord(name[0])
    if 65 <= char_code <= 90:  # A-Z harfleri
        labels[char_code - 65] = 1
        return np.array(labels)
    return None

# Veri Yükleme Fonksiyonu
def load_data(DIR, data_type='train', validation_ids=None, test_ids=None):
    if validation_ids is None:
        validation_ids = []
    if test_ids is None:
        test_ids = []

    exclude = []
    include = []
    if data_type == 'train':
        exclude.extend(validation_ids)
        exclude.extend(test_ids)
    elif data_type == 'validation':
        include.extend(validation_ids)
    elif data_type == 'test':
        include.extend(test_ids)

    data = []
    for img_path in glob.glob(os.path.join(DIR, '*.png')):
        img_id = int(img_path[img_path.find("(") + 1: img_path.find(")")])
        if exclude and img_id in exclude:
            continue
        elif include and img_id not in include:
            continue

        label = label_img(os.path.basename(img_path))
        if label is None:
            continue

        img = Image.open(img_path).convert('L')
        img = np.stack((img,)*3, axis=-1)  # Gri ölçekli görüntüyü 3 kanallı hale getir
        data.append([np.array(img), label])

    shuffle(data)
    return data

# JSON dosyasından model yükleme ve H5 dosyasından ağırlık yükleme
def load_model_from_json_and_weights(json_file_path, weights_file_path):
    with open(json_file_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_file_path)
    return model

# H5 dosyasından model yükleme (hem yapı hem ağırlık)
def load_model_structure_and_weights(structure_file_path, weights_file_path):
    model = tf.keras.models.load_model(structure_file_path)
    model.load_weights(weights_file_path)
    return model

# Modelleri Yükle
alexnet_model = load_model_from_json_and_weights('AlexNet_model.json', 'AlexNet_Weights.weights.h5')
vgg16_model = load_model_structure_and_weights('VGG16_model.h5', 'VGG16_Weights.weights.h5')
mobilenet_model = load_model_structure_and_weights('MobileNet_model.h5', 'MobileNet_Weights.weights.h5')

# Resim yeniden boyutlandırma
def preprocess_image(image, size):
    return np.array(Image.fromarray(image).resize((size, size), Image.Resampling.LANCZOS))

# Veri yükleme ve ön işleme
train_data = load_data(DATASET_PATH, 'train', validation_ids=VALIDATION_IMAGE_IDS, test_ids=TEST_IMAGE_IDS)
validation_data = load_data(DATASET_PATH, 'validation', validation_ids=VALIDATION_IMAGE_IDS, test_ids=TEST_IMAGE_IDS)
test_data = load_data(DATASET_PATH, 'test', validation_ids=VALIDATION_IMAGE_IDS, test_ids=TEST_IMAGE_IDS)

train_images = [i[0] for i in train_data]
validation_images = [i[0] for i in validation_data]
test_images = [i[0] for i in test_data]

train_labels = [np.argmax(i[1]) for i in train_data]
validation_labels = [np.argmax(i[1]) for i in validation_data]
test_labels = [np.argmax(i[1]) for i in test_data]

# AlexNet için veri hazırlama
alexnet_train_images = np.array([preprocess_image(img, IMG_SIZE_ALEXNET) for img in train_images]).reshape(-1, IMG_SIZE_ALEXNET, IMG_SIZE_ALEXNET, 3) / 255.0
alexnet_validation_images = np.array([preprocess_image(img, IMG_SIZE_ALEXNET) for img in validation_images]).reshape(-1, IMG_SIZE_ALEXNET, IMG_SIZE_ALEXNET, 3) / 255.0
alexnet_test_images = np.array([preprocess_image(img, IMG_SIZE_ALEXNET) for img in test_images]).reshape(-1, IMG_SIZE_ALEXNET, IMG_SIZE_ALEXNET, 3) / 255.0

alexnet_train_preds = alexnet_model.predict(alexnet_train_images)
alexnet_validation_preds = alexnet_model.predict(alexnet_validation_images)
alexnet_test_preds = alexnet_model.predict(alexnet_test_images)

# VGG16 için veri hazırlama
vgg16_train_images = np.array([preprocess_image(img, IMG_SIZE_VGG16) for img in train_images]).reshape(-1, IMG_SIZE_VGG16, IMG_SIZE_VGG16, 3) / 255.0
vgg16_validation_images = np.array([preprocess_image(img, IMG_SIZE_VGG16) for img in validation_images]).reshape(-1, IMG_SIZE_VGG16, IMG_SIZE_VGG16, 3) / 255.0
vgg16_test_images = np.array([preprocess_image(img, IMG_SIZE_VGG16) for img in test_images]).reshape(-1, IMG_SIZE_VGG16, IMG_SIZE_VGG16, 3) / 255.0

vgg16_train_preds = vgg16_model.predict(vgg16_train_images)
vgg16_validation_preds = vgg16_model.predict(vgg16_validation_images)
vgg16_test_preds = vgg16_model.predict(vgg16_test_images)

# MobileNet için veri hazırlama
mobilenet_train_images = np.array([preprocess_image(img, IMG_SIZE_MOBILENET) for img in train_images]).reshape(-1, IMG_SIZE_MOBILENET, IMG_SIZE_MOBILENET, 3) / 255.0
mobilenet_validation_images = np.array([preprocess_image(img, IMG_SIZE_MOBILENET) for img in validation_images]).reshape(-1, IMG_SIZE_MOBILENET, IMG_SIZE_MOBILENET, 3) / 255.0
mobilenet_test_images = np.array([preprocess_image(img, IMG_SIZE_MOBILENET) for img in test_images]).reshape(-1, IMG_SIZE_MOBILENET, IMG_SIZE_MOBILENET, 3) / 255.0

mobilenet_train_preds = mobilenet_model.predict(mobilenet_train_images)
mobilenet_validation_preds = mobilenet_model.predict(mobilenet_validation_images)
mobilenet_test_preds = mobilenet_model.predict(mobilenet_test_images)

# Tahminleri birleştirme (stacking)
stacked_train_preds = np.concatenate([alexnet_train_preds, vgg16_train_preds, mobilenet_train_preds], axis=1)
stacked_validation_preds = np.concatenate([alexnet_validation_preds, vgg16_validation_preds, mobilenet_validation_preds], axis=1)
stacked_test_preds = np.concatenate([alexnet_test_preds, vgg16_test_preds, mobilenet_test_preds], axis=1)

# Meta-model eğitme (Logistic Regression)
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(stacked_train_preds, train_labels)

# Meta-model ile tahmin yapma
meta_validation_preds = meta_model.predict(stacked_validation_preds)
meta_test_preds = meta_model.predict(stacked_test_preds)

# Doğruluk oranını hesaplama
validation_accuracy = accuracy_score(validation_labels, meta_validation_preds)
test_accuracy = accuracy_score(test_labels, meta_test_preds)
print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Harf bazında doğruluk oranını hesaplama
char_lists = {chr(i + 65): {'total': 0, 'predicted': 0} for i in range(26)}
for index, label in enumerate(test_labels):
    predicted = meta_test_preds[index]
    char = chr(label + 65)
    char_lists[char]['total'] += 1
    if predicted == label:
        char_lists[char]['predicted'] += 1

for char in char_lists:
    total_char = char_lists[char]['total']
    predicted_char = char_lists[char]['predicted']
    accuracy_char = 100 * predicted_char / total_char if total_char > 0 else 0
    print(f'{char}: {accuracy_char:.2f}%')
