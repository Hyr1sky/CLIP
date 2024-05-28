import torch
import clip
import numpy as np
import shutil
import os

from PIL import Image


def clip_img2img(searchpath, filepath, newfilepath, threshold):
    """
    Image similarity search using CLIP model.
    Additional Task:
    - Select image when running similarity search.

    :param searchpath: images to be searched
    :param filepath: target images
    :param newfilepath: result
    :param threshold: similarity threshold
    :return: None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    newfilepath = os.path.join(newfilepath, "img2img")
    
    def load_and_preprocess_image(image_path):
        image = Image.open(image_path)
        image = preprocess(image).unsqueeze(0).to(device)
        return image

    def clip_img_score(img1_path, img2_path):
        image_a = load_and_preprocess_image(img1_path)
        image_b = load_and_preprocess_image(img2_path)

        with torch.no_grad():
            embedding_a = model.encode_image(image_a)
            embedding_b = model.encode_image(image_b)

        similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
        return similarity_score.item()

    for parent, _, filenames in os.walk(filepath):
        for filename in filenames:
            img2_path = os.path.join(filepath, filename)

            for _, _, search_filenames in os.walk(searchpath):
                for search_filename in search_filenames:
                    img1_path = os.path.join(searchpath, search_filename)
                    similarity = clip_img_score(img1_path, img2_path)
                    
                    if similarity >= threshold:
                        print(f"Found similar image: {img1_path}, similarity: {similarity}")
                        shutil.copy(img1_path, newfilepath)


if __name__ == '__main__':
    searchpath = "./data/search"
    filepath = "./data/dataset"
    newfilepath = "./data/similar"
    threshold = 0.8

    clip_img2img(searchpath, filepath, newfilepath, threshold)