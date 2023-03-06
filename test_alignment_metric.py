
from argparse import Namespace
import time
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import clip
from contextlib import contextmanager
from tqdm.auto import tqdm
import shutil
import cv2
import pickle
import seaborn as sns
import shutil

from global_torch.manipulate import Manipulator
from global_torch.StyleCLIP import zeroshot_classifier, imagenet_templates


def get_image_emb(image_path, model, device, preprocess):
    with torch.no_grad():
        image = Image.open(image_path)
        image_emb = model.encode_image(preprocess(image).unsqueeze(0).to(device))
        image_emb /= image_emb.norm(dim=-1, keepdim=True)
        image_emb = image_emb.cpu().detach().numpy().astype("float32")[0]
        return image_emb


def get_text_emb(text, model):
    text_features = zeroshot_classifier(text, imagenet_templates, model).t()
    text_feature = text_features[0].cpu().numpy()
    text_emb = text_feature / np.linalg.norm(text_feature)
    return text_emb


def run_score(dir, model, device, preprocess, target_prompt):
    text_emb = get_text_emb(target_prompt, model)
    print("text_emb:", text_emb.shape)

    file_list = []
    score_list = []
    for root, dirs, files in os.walk(dir):
        for file in  files:
            image_path = os.path.join(root, file)
            image_emb = get_image_emb(image_path, model, device, preprocess)
            score = np.dot(text_emb, image_emb)
            file_list.append(image_path)
            score_list.append(score)

    return file_list, score_list

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    train_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated_v2/train_class_aligned"
    test_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated_v2/test_class_aligned"

    # target_prompt = 'Laugh heartily face'
    # file_list = []
    # score_list = []
    # fl, sl = run_score(train_dir, model, device, preprocess, target_prompt)
    # file_list.extend(fl)
    # score_list.extend(sl)
    # fl, sl = run_score(test_dir, model, device, preprocess, target_prompt)
    # file_list.extend(fl)
    # score_list.extend(sl)
    # with open('laugh_heartily_face_list.pkl', 'wb') as f:
    #     pickle.dump(file_list, f)
    # np.save("score_laugh_heartily_face_list", np.array(score_list))

    with open('laugh_heartily_face_list.pkl', 'rb') as f:
        file_list = pickle.load(f)
    score_list = np.load("score_laugh_heartily_face_list.npy")

    plt.figure("score_laugh_heartily_face")
    sns.displot(score_list, kde=False)
    plt.savefig("score_laugh_heartily_face")
    plt.cla()

    score_mean = score_list.mean()
    score_std = score_list.std()
    threshold1 = score_mean - 0.5 * score_std
    threshold2 = score_mean + 0.5 * score_std
    print("score_mean:", score_mean)
    print("score_std:", score_std)
    print("t1, t2:", threshold1, threshold2)
    resample_scores = np.random.normal(loc=score_mean, scale=score_std, size=score_list.shape)
    sns.histplot(score_list, kde=False, label="org score", color="skyblue")
    sns.histplot(resample_scores, kde=False, label="resample score", color="red")
    plt.legend()
    plt.savefig("score_laugh_heartily_face_resample")
    plt.cla()

    target_train_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated_v3/train_class_aligned"
    target_test_dir = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/emotion/RAF-DB/basic-20201119T055425Z-001/basic/generated_v3/test_class_aligned"

    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_test_dir, exist_ok=True)

    for file, score in zip(file_list, score_list):
        if "train" in file:
            new_file = file.replace(train_dir, target_train_dir)
        else:
            new_file = file.replace(test_dir, target_test_dir)

        if score < threshold1:
            ev = 1
        elif threshold1 <= score < threshold2:
            ev = 2
        else:
            ev = 3

        new_dir = new_file.replace(new_file.split("/")[-1], "")
        os.makedirs(new_dir, exist_ok=True)
        new_file = new_file.split("ev_")[0] + "ev_" + str(ev) + ".jpg"
        shutil.copyfile(file, new_file)


if __name__ == '__main__':
    main()