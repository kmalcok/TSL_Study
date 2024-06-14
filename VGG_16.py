import os
import glob
import numpy as np
from PIL import Image
from random import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# Mixed Precision Training devre dışı bırakıldı
# tf.keras.mixed_precision.set_global_policy('float32')

# Constants
IMG_SIZE = 224  # VGG16 için uygun resim boyutu
BATCH_SIZE = 50
EPOCHS = 10  # Başlangıçta daha az epoch deneyin
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
def load_data(DIR, data_type='train', validation_ids=None, test_ids=None, img_size=224):
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
        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        img = np.stack((img,)*3, axis=-1)  # Gri ölçekli görüntüyü 3 kanallı hale getir
        data.append([np.array(img), label])

    shuffle(data)
    return data

# Model Eğitme
def trainModel():
    train_data = load_data(DATASET_PATH, 'train')
    validation_data = load_data(DATASET_PATH, 'validation')

    trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    trainLabels = np.array([i[1] for i in train_data])

    validationImages = np.array([i[0] for i in validation_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    validationLabels = np.array([i[1] for i in validation_data])

    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    vgg_base.trainable = False  # Önceden eğitilmiş katmanları dondur

    model = Sequential()
    model.add(vgg_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation='softmax'))

    # Adam optimizer kullanımı
    optimizer = Adam(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(trainImages, trainLabels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validationImages, validationLabels))

    return model

# Model Kaydetme
def save_model(model, model_file_path, weight_file_path):
    model.save(model_file_path)
    model.save_weights(weight_file_path)

# Model Yükleme
def load_model(model_file_path, weight_file_path):
    model = tf.keras.models.load_model(model_file_path)
    model.load_weights(weight_file_path)
    return model

# Model Test Etme
def testModel(model):
    test_data = load_data(DATASET_PATH, 'test')
    testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    testLabels = np.array([i[1] for i in test_data])

    totalList = []
    charLists = {}
    for index, test_image in enumerate(testImages):
        test_image = np.expand_dims(test_image, axis=0)
        prediction, labelIndex = predict(model, test_image)
        predicted = labelIndex == np.where(testLabels[index] == 1)[0][0]
        totalList.append(predicted)

        if prediction not in charLists:
            charLists[prediction] = {"total": 0, "predicted": 0}
        charLists[prediction]['total'] += 1
        if predicted:
            charLists[prediction]['predicted'] += 1

    total = len(totalList)
    predicted = totalList.count(True)
    print(total)
    print(predicted)
    print(100 * predicted / total)

    charLists = dict(sorted(charLists.items()))
    for prediction in charLists:
        success = 100 * charLists[prediction]['predicted'] / charLists[prediction]['total']
        print(prediction + ': ' + str(round(success)))

# Tahmin Fonksiyonu
def predict(model, test_image):
    result = model.predict(test_image, batch_size=10, verbose=0)
    maxPosibility = max(result[0])
    classIds = [i for i, j in enumerate(result[0]) if j == maxPosibility]

    for value in classIds:
        return [chr(value + 65), value]

# Model Eğitme ve Kaydetme
#model = trainModel()
#save_model(model, 'VGG16_model.h5', 'VGG16_Weights.weights.h5')

# Model Yükleme ve Test Etme
model = load_model('VGG16_model.h5', 'VGG16_Weights.weights.h5')
testModel(model)
