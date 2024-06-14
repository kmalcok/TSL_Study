import gc
import glob
import os
from random import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
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
        img = np.stack((img,) * 3, axis=-1)  # Gri ölçekli görüntüyü 3 kanallı hale getir
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


# Model Test Etme
def testModel(models):
    test_data = load_data(DATASET_PATH, 'test', test_ids=TEST_IMAGE_IDS)
    test_images = [i[0] for i in test_data]
    test_labels = np.array([i[1] for i in test_data])

    alexnet_images = np.array([preprocess_image(img, IMG_SIZE_ALEXNET) for img in test_images]).reshape(-1,
                                                                                                        IMG_SIZE_ALEXNET,
                                                                                                        IMG_SIZE_ALEXNET,
                                                                                                        1) / 255.0
    vgg16_images = np.array([preprocess_image(img, IMG_SIZE_VGG16) for img in test_images]).reshape(-1, IMG_SIZE_VGG16,
                                                                                                    IMG_SIZE_VGG16,
                                                                                                    3) / 255.0
    mobilenet_images = np.array([preprocess_image(img, IMG_SIZE_MOBILENET) for img in test_images]).reshape(-1,
                                                                                                            IMG_SIZE_MOBILENET,
                                                                                                            IMG_SIZE_MOBILENET,
                                                                                                            3) / 255.0

    test_images = [alexnet_images, vgg16_images, mobilenet_images]

    totalList = []
    charLists = {chr(i + 65): {'total': 0, 'predicted': 0} for i in range(26)}

    for index in range(len(test_labels)):
        images_for_models = [images[index:index + 1] for images in test_images]
        predictions = [model.predict(images_for_model) for model, images_for_model in zip(models, images_for_models)]

        label_index = np.where(test_labels[index] == 1)[0][0]

        # Her modelin tahminlerini kontrol etme
        correct_prediction = any(np.argmax(pred, axis=1)[0] == label_index for pred in predictions)
        totalList.append(correct_prediction)

        char = chr(label_index + 65)
        charLists[char]['total'] += 1
        if correct_prediction:
            charLists[char]['predicted'] += 1

    total = len(totalList)
    predicted = totalList.count(True)
    print(f'Total: {total}')
    print(f'Predicted: {predicted}')
    print(f'Accuracy: {100 * predicted / total:.2f}%')

    for char in charLists:
        total_char = charLists[char]['total']
        predicted_char = charLists[char]['predicted']
        accuracy_char = 100 * predicted_char / total_char if total_char > 0 else 0
        print(f'{char}: {accuracy_char:.2f}%')


# Modellerin başarı oranları
accuracy_mobilenet = 0.9867
accuracy_alexnet = 0.9991
accuracy_vgg16 = 1.0

# Ensemble modelini test et
models = [alexnet_model, vgg16_model, mobilenet_model]
testModel(models)
