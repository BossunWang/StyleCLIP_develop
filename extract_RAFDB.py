import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil


def main():
    data_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/Image/original"
    emotion_label_file_name = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/EmoLabel/list_patition_label.txt"
    target_train_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/org_train_class"
    target_test_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/org_test_class"

    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_test_dir, exist_ok=True)
    emotion_list = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

    for emotion_name in emotion_list:
        os.makedirs(os.path.join(target_train_dir, emotion_name), exist_ok=True)
        os.makedirs(os.path.join(target_test_dir, emotion_name), exist_ok=True)

    emotion_file = open(emotion_label_file_name, 'r')
    for data in tqdm(emotion_file.readlines()):
        file_name, emotion_label = data.split(" ")
        emotion_label = int(emotion_label) - 1
        if "train" in file_name:
            image_path = os.path.join(data_dir, file_name)
            target_path = os.path.join(target_train_dir, emotion_list[emotion_label], file_name)
            shutil.copyfile(image_path, target_path)
        elif "test" in file_name:
            image_path = os.path.join(data_dir, file_name)
            target_path = os.path.join(target_test_dir, emotion_list[emotion_label], file_name)
            shutil.copyfile(image_path, target_path)


if __name__ == '__main__':
    main()