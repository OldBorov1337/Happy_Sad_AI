import os
import tensorflow as tf

data_dir = 'data'  # Путь к папке с изображениями
supported_exts = [".jpeg", ".jpg", ".bmp", ".png"]
problem_files = []

# Проверка каждого файла в папке
for root, dirs, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(root, file)

        # Проверяем расширение файла
        if not any(file.lower().endswith(ext) for ext in supported_exts):
            print(f"Skipping unsupported file: {file_path}")
            continue

        # Пробуем загрузить изображение с помощью TensorFlow
        try:
            img_raw = tf.io.read_file(file_path)
            img_tensor = tf.io.decode_image(img_raw)  # Декодирование изображения
            img_tensor.shape  # Проверка размера изображения (если декодирование прошло успешно)
        except Exception as e:
            print(f"Problematic file detected: {file_path}, error: {e}")
            problem_files.append(file_path)

# Итоговый список проблемных файлов
if problem_files:
    print("\nDetected files that cause issues with TensorFlow decoding:")
    for file in problem_files:
        print(file)
    print("\nPlease remove or replace these files and try again.")
else:
    print("All files are verified and compatible with TensorFlow.")
