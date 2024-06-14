import gc
import glob
import os
import random
from random import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image


os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


def clear_memory():
    gc.collect()
    tf.keras.backend.clear_session()

# Sabitler
IMG_SIZE_INCEPTIONV3 = 224
IMG_SIZE_MOBILENET = 224
IMG_SIZE_VGG16 = 224

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
def load_data_random_sample(DIR, sample_size_per_class=100):
    data = []
    class_data = {chr(i + 65): [] for i in range(26)}

    for img_path in glob.glob(os.path.join(DIR, '*.png')):
        label = label_img(os.path.basename(img_path))
        if label is None:
            continue

        img = Image.open(img_path).convert('L')
        img = np.stack((img,)*3, axis=-1)  # Gri ölçekli görüntüyü 3 kanallı hale getir
        class_data[os.path.basename(img_path)[0]].append([np.array(img), label])

    for char in class_data:
        if len(class_data[char]) >= sample_size_per_class:
            data.extend(random.sample(class_data[char], sample_size_per_class))
        else:
            data.extend(class_data[char])

    shuffle(data)
    return data

# H5 dosyasından model yükleme (hem yapı hem ağırlık)
def load_model_structure_and_weights(structure_file_path, weights_file_path):
    model = tf.keras.models.load_model(structure_file_path)
    model.load_weights(weights_file_path)
    return model

# Modelleri Yükle
inceptionv3_model = load_model_structure_and_weights('InceptionV3_Model.h5', 'InceptionV3_Weights.weights.h5')
vgg16_model = load_model_structure_and_weights('VGG16_model.h5', 'VGG16_Weights.weights.h5')
mobilenet_model = load_model_structure_and_weights('MobileNet_model.h5', 'MobileNet_Weights.weights.h5')

# Resim yeniden boyutlandırma
def preprocess_image(image, size, grayscale=False):
    img = Image.fromarray(image).resize((size, size), Image.Resampling.LANCZOS)
    if grayscale:
        img = img.convert('L')
    return np.array(img)

# Model Performansını Yazdırma
def evaluate_model(model, test_images, test_labels, image_size, model_name, grayscale=False):
    processed_images = np.array([preprocess_image(img, image_size, grayscale) for img in test_images])
    if grayscale:
        processed_images = processed_images.reshape(-1, image_size, image_size, 1) / 255.0
    else:
        processed_images = processed_images.reshape(-1, image_size, image_size, 3) / 255.0
    predictions = np.argmax(model.predict(processed_images), axis=1)
    correct_predictions = np.sum(predictions == np.argmax(test_labels, axis=1))
    accuracy = correct_predictions / len(test_labels) * 100

    print(f"{model_name} Model Accuracy: {accuracy:.2f}%")
    return accuracy

# Model Test Etme
def testModel(models):
    test_data = load_data_random_sample(DATASET_PATH, sample_size_per_class=100)
    test_images = [i[0] for i in test_data]
    test_labels = np.array([i[1] for i in test_data])

    inceptionv3_accuracy = evaluate_model(inceptionv3_model, test_images, test_labels, IMG_SIZE_INCEPTIONV3, "InceptionV3")
    vgg16_accuracy = evaluate_model(vgg16_model, test_images, test_labels, IMG_SIZE_VGG16, "VGG16")
    mobilenet_accuracy = evaluate_model(mobilenet_model, test_images, test_labels, IMG_SIZE_MOBILENET, "MobileNet")

    inceptionv3_images = np.array([preprocess_image(img, IMG_SIZE_INCEPTIONV3) for img in test_images]).reshape(-1,
                                                                                                                IMG_SIZE_INCEPTIONV3,
                                                                                                                IMG_SIZE_INCEPTIONV3,
                                                                                                                3) / 255.0
    vgg16_images = np.array([preprocess_image(img, IMG_SIZE_VGG16) for img in test_images]).reshape(-1, IMG_SIZE_VGG16,
                                                                                                    IMG_SIZE_VGG16,
                                                                                                    3) / 255.0
    mobilenet_images = np.array([preprocess_image(img, IMG_SIZE_MOBILENET) for img in test_images]).reshape(-1,
                                                                                                            IMG_SIZE_MOBILENET,
                                                                                                            IMG_SIZE_MOBILENET,
                                                                                                            3) / 255.0

    test_images = [inceptionv3_images, vgg16_images, mobilenet_images]

    totalList = []
    charLists = {chr(i + 65): {'total': 0, 'predicted': 0} for i in range(26)}

    for index in range(len(test_labels)):
        images_for_models = [images[index:index + 1] for images in test_images]
        predictions = [np.argmax(model.predict(images_for_model), axis=1)[0] for model, images_for_model in zip(models, images_for_models)]

        label_index = np.where(test_labels[index] == 1)[0][0]

        # Çoğunluk oylaması
        if predictions.count(label_index) > len(predictions) // 2:
            correct_prediction = True
        else:
            correct_prediction = False

        totalList.append(correct_prediction)

        char = chr(label_index + 65)
        charLists[char]['total'] += 1
        if correct_prediction:
            charLists[char]['predicted'] += 1

    total = len(totalList)
    predicted = totalList.count(True)
    print(f'Ensemble Total: {total}')
    print(f'Ensemble Predicted: {predicted}')
    print(f'Ensemble Accuracy: {100 * predicted / total:.2f}%')

    for char in charLists:
        total_char = charLists[char]['total']
        predicted_char = charLists[char]['predicted']
        accuracy_char = 100 * predicted_char / total_char if total_char > 0 else 0
        print(f'{char}: {accuracy_char:.2f}%')

# Ensemble modelini test et
models = [inceptionv3_model, vgg16_model, mobilenet_model]
testModel(models)
