import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras import regularizers
from collections import Counter

def load_and_split_data(csv_file, image_size=(64, 64), train_size=6000, val_size=2000):
    df = pd.read_csv(csv_file, encoding='utf-8')
    csv_dir = os.path.dirname(csv_file)
    images = []
    labels = []
    mapping = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'}
    for index, row in df.iterrows():
        image_path = os.path.normpath(os.path.join(csv_dir, row.iloc[0]))
        label_str = str(row.iloc[1]).strip()  # Get label from csv file as str-

        if len(label_str) < 6:
           label_str = '0' * (6 - len(label_str)) + label_str
        label = []
        for char in label_str:
            if char in mapping:
                label.append(mapping[char])
            else:
                continue
        try:
            image = Image.open(image_path).convert('L').resize(image_size)
            image = np.array(image) / 255.0
            images.append(image)
            labels.append(label)
        except Exception as e:
            print(f"Ошибка открытия файла по пути: {image_path}, {e}")

    X = np.array(images)
    y = np.array(labels)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Путь к CSV
csv_path = 'практика 1/Creative/captcha_data.csv'

# Загрузка и разделение данных
X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data(
    csv_path, image_size=(64, 64), train_size=6000, val_size=2000
)

# Кодирование меток
num_classes = 10
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
y_train_encoded = onehot_encoder.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape[0], 6, num_classes)
y_val_encoded = onehot_encoder.transform(y_val.reshape(-1, 1)).reshape(y_val.shape[0], 6, num_classes)
y_test_encoded = onehot_encoder.transform(y_test.reshape(-1, 1)).reshape(y_test.shape[0], 6, num_classes)

label_encoder = LabelEncoder()
label_encoder.fit(np.array(['0','1','2','3','4','5','6','7','8','9']))

integer_encoded_train = label_encoder.transform(y_train.reshape(-1)).reshape(y_train.shape[0], 6)
integer_encoded_val = label_encoder.transform(y_val.reshape(-1)).reshape(y_val.shape[0], 6)
integer_encoded_test = label_encoder.transform(y_test.reshape(-1)).reshape(y_test.shape[0], 6)

# Построение модели
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(num_classes*6, activation='relu'),
    Reshape((6, num_classes)),
    tf.keras.layers.Activation('softmax')
])

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Обучение модели
model.fit(X_train, integer_encoded_train, epochs=200, batch_size=32, validation_data=(X_val, integer_encoded_val), shuffle=True)

# Оценка модели
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=2)
accuracy = np.mean(np.all(integer_encoded_test == y_pred_labels, axis=1))
print(f'Точность на тестовой выборке: {accuracy * 100:.2f}%')