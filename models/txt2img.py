import torch
import clip
import os
import shutil
import numpy as np

from PIL import Image


def clip_txt2img(searchpath, filepath, newfilepath):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    similar_value = []
    similar_path = []

    text = clip.tokenize(get_token()).to(device)

    with torch.no_grad():
        for parent, dirnames, filenames in os.walk(searchpath):
            for filename in filenames:
                img_path = os.path.join(parent, filename)
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                
                # 对比文本与图片特征
                text_features = model.encode_text(text)
                logits_per_image, logits_per_text = model(image, text)
                similarity_scores = str(logits_per_image)[9:13]
                similar_value.append(float(similarity_scores))
                similar_path.append(filename)

    similar_dict = dict(zip(similar_path, similar_value))
    similar_value.sort(reverse=True)
    for i in range(5):
        print(similar_path[i])
        shutil.copy(os.path.join(searchpath, similar_path[i]), os.path.join(newfilepath, similar_path[i]))


def get_token():
    # 获取关键词
    print("Please type in your keywords:")
    token = [input()]
    return token


def cosine_similarity(features1, features2):
    cosine_similarity = torch.nn.functional.cosine_similarity(features1, features2, dim=1).softmax(dim=-1)
    return cosine_similarity


if __name__ == '__main__':
    searchpath = "./data/search"
    filepath = "./data/dataset"
    newfilepath = "./data/similar"

    clip_txt2img(searchpath, filepath, newfilepath)
