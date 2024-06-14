import os
import glob
import numpy as np
from PIL import Image
from random import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Constants
IMG_SIZE = 200
VALIDATION_IMAGE_IDS = [2, 6, 12, 16, 22, 26, 32, 36, 42, 46, 52, 56, 62, 66, 72, 76, 82, 86, 92, 96]
TEST_IMAGE_IDS = [1, 5, 11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61, 65, 71, 75, 81, 85, 91, 95]
DATASET_PATH = "C:/TSL Finger Spelling/Images"

# Model Kaydetme
def save_model(model, model_file_path, weight_file_path):
    with open(model_file_path, "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(weight_file_path)

# Model Yükleme
def load_model(model_file_path, weight_file_path):
    with open(model_file_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight_file_path)
    return loaded_model

# Etiketleme Fonksiyonu
def label_img(name):
    labels = [0 for _ in range(26)]
    char_code = ord(name[0])
    if 65 <= char_code <= 90:  # A-Z harfleri
        labels[char_code - 65] = 1
        print(f"Image: {name}, Label: {labels}")  # Etiketi yazdır
        return np.array(labels)
    return None

# Veri Yükleme Fonksiyonu
def load_data(DIR, data_type='train', validation_ids=None, test_ids=None, img_size=200):
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
        data.append([np.array(img), label])

    shuffle(data)
    return data

# Model Eğitme
def trainModel():
    train_data = load_data(DATASET_PATH, 'train')
    validation_data = load_data(DATASET_PATH, 'validation')
    # print(train_data)
    """
    import matplotlib.pyplot as plt
    plt.imshow(train_data[43][0], cmap = 'gist_gray')
    """
    trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    trainLabels = np.array([i[1] for i in train_data])

    validationImages = np.array([i[0] for i in validation_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    validationLabels = np.array([i[1] for i in validation_data])

    # Veri artırma
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(trainImages)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

    model.fit(datagen.flow(trainImages, trainLabels, batch_size=50), epochs=50, verbose=1,
              validation_data=(validationImages, validationLabels), callbacks=[early_stopping, reduce_lr])

    return model

# Model Test Etme
def testModel(model):
    test_data = load_data(DATASET_PATH, 'test')
    testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
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

def predict(model, test_image):
    result = model.predict(test_image, batch_size=10, verbose=0)
    maxPosibility = max(result[0])
    classIds = [i for i, j in enumerate(result[0]) if j == maxPosibility]
    """
    print(result)
    print(maxPosibility)
    """
    # print ('prediction result: ')
    for value in classIds:
        return [chr(value + 65), value]
    """
    print ('all possibilities: ')
    for index, value in enumerate(result[0]):
        print (chr(index + 65) + ' -> ' + str(value))
    """

# Modeli Eğit
model = trainModel()
# Modeli Kaydet
save_model(model, 'AlexNet_model.json', 'AlexNet_Weights.weights.h5')
model = load_model('AlexNet_model.json', 'AlexNet_Weights.weights.h5')
# Modeli Test Et
testModel(model)
