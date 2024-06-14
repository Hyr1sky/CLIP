import torch
import clip
import os
import shutil
import numpy as np
import warnings

from PIL import Image

warnings.filterwarnings("ignore", "(?s).*Corrupt EXIF data.*", UserWarning)


def clip_txt2img(searchpath, filepath, newfilepath, threshold):
    """
    Image similarity search using CLIP model.
    Text to image.

    :param searchpath: images to be searched
    :param filepath: target images
    :param newfilepath: result
    :param threshold: similarity threshold
    :return: None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    newfilepath = os.path.join(newfilepath, "txt2img")
    
    similar_scores = []
    similar_paths = []

    text = clip.tokenize(get_token()).to(device)

    with torch.no_grad():
        for parent, _, filenames in os.walk(searchpath):
            for filename in filenames:
                if filename.endswith('.DS_Store'):
                    continue
                img_path = os.path.join(parent, filename)
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                similarity_score = torch.nn.functional.cosine_similarity(image_features, text_features)
                
                if similarity_score.item() >= threshold:
                    print(f"Found similar image: {img_path}, similarity: {similarity_score.item()}")
                    shutil.copy(img_path, newfilepath)

                # similar_scores.append(similarity_score.item())
                # similar_paths.append(img_path)

    # Select top 5 similar images
    # sorted_indices = sorted(range(len(similar_scores)), key=lambda i: similar_scores[i], reverse=True)
    # top5_indices = sorted_indices[:5]
    # for idx in top5_indices:
    #     img_path = similar_paths[idx]
    #     shutil.copy(img_path, newfilepath)


def get_token():
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
    threshold = 0.22

    clip_txt2img(searchpath, filepath, newfilepath, threshold)
