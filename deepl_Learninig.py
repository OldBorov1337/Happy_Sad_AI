import tensorflow as tf
import os
from matplotlib import pyplot as plt
import keras
import numpy as np
from PIL import Image

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'data'
image_exts = [".jpeg", ".jpg", ".bmp", ".png"]

# Проверка файлов на читаемость
verified_files = []
for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    if not os.path.isdir(class_path):
        continue

    for image in os.listdir(class_path):
        image_path = os.path.join(class_path, image)
        try:
            # Открываем изображение для проверки
            with Image.open(image_path) as img:
                img.verify()  # Проверяем, действительно ли это изображение
            verified_files.append(image_path)
        except (IOError, SyntaxError) as e:
            print(f"Skipping unsupported or corrupt file: {image_path}")

# Создаем аугментацию данных
data_augmentation = keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Загрузка данных с использованием validation_split и subset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    batch_size=32,
    label_mode="int"
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=32,
    label_mode="int"
)

# Создаем отдельный тестовый датасет, используя ещё одну часть данных
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=32,
    label_mode="int"
)

# Применение аугментации к тренировочным данным и масштабирование
train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True) / 255.0, y))
val_data = val_data.map(lambda x, y: (x / 255.0, y))
test_data = test_data.map(lambda x, y: (x / 255.0, y))

# Построение модели с Dropout и регуляризацией
model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(256, 256, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(16, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.5),  # Добавили Dropout для регуляризации
    keras.layers.Dense(1, activation="sigmoid")
])

# Уменьшение скорости обучения
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

# Логирование и обучение с ранней остановкой
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

hist = model.fit(train_data, epochs=25, validation_data=val_data,
                 callbacks=[tensorboard_callback, early_stopping])

# Визуализация метрик обучения
fig = plt.figure()
plt.plot(hist.history['loss'], color="teal", label='loss')
plt.plot(hist.history['val_loss'], color="orange", label="val_loss")
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# Оценка модели на тестовом датасете
from keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test_data.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f"Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}")


import cv2

img = cv2.imread("test1.jpg")
# img2 = cv2.imread("test_happy.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
# plt.imshow(img2)
plt.show()

resize = tf.image.resize(img,(256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255,0))
if yhat > 0.5:
    print(f"predicted class is Sad")
else:
    print("predicted class is Happy")



