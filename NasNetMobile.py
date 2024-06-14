import os
import glob
import numpy as np
from PIL import Image
from random import shuffle
import tensorflow as tf
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc

# GPU kullanımı için ayar
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


# Bellek temizleme
def clear_memory():
    gc.collect()
    tf.keras.backend.clear_session()


# Sabitler
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
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
        img = np.stack((img,) * 3, axis=-1)  # Gri ölçekli görüntüyü 3 kanallı hale getir
        data.append([np.array(img), label])

    shuffle(data)
    return data


# Model Eğitme
def trainModel():
    train_data = load_data(DATASET_PATH, 'train', validation_ids=VALIDATION_IMAGE_IDS, test_ids=TEST_IMAGE_IDS)
    validation_data = load_data(DATASET_PATH, 'validation', validation_ids=VALIDATION_IMAGE_IDS,
                                test_ids=TEST_IMAGE_IDS)

    trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    trainLabels = np.array([i[1] for i in train_data])

    validationImages = np.array([i[0] for i in validation_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    validationLabels = np.array([i[1] for i in validation_data])

    # Veri artırma
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = datagen.flow(trainImages, trainLabels, batch_size=BATCH_SIZE)
    validation_dataset = tf.data.Dataset.from_tensor_slices((validationImages, validationLabels)).batch(
        BATCH_SIZE).prefetch(AUTOTUNE)

    nasnet_base = NASNetMobile(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # NASNetMobile katmanlarını dondur
    for layer in nasnet_base.layers:
        layer.trainable = False

    model = Sequential()
    model.add(nasnet_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(26, activation='softmax'))

    # Daha düşük öğrenme oranı ve callbacks
    optimizer = Adam(learning_rate=0.1)

    # Early Stopping ve ReduceLROnPlateau callback'leri
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, callbacks=[early_stopping, reduce_lr])

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
    test_data = load_data(DATASET_PATH, 'test', test_ids=TEST_IMAGE_IDS, img_size=IMG_SIZE)
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


# Belleği Temizle
clear_memory()
# Modeli Eğit
model = trainModel()
# Modeli Kaydet
save_model(model, 'NASNetMobile_Model.h5', 'NASNetMobile_Weights.weights.h5')
# Modeli Yükle
model = load_model('NASNetMobile_Model.h5', 'NASNetMobile_Weights.weights.h5')
# Modeli Test Et
testModel(model)
