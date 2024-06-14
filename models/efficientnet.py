import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import shutil
import os

from PIL import Image


def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image


def compute_similarity(image1, image2, model):
    with torch.no_grad():
        output1 = model(image1)
        output2 = model(image2)
        similarity_score = np.dot(output1.numpy().flatten(), output2.numpy().flatten()) / (
                    np.linalg.norm(output1.numpy()) * np.linalg.norm(output2.numpy())
                )
    return similarity_score


def efficientNetB2(searchpath, filepath, newfilepath, threshold):
    """
    Image similarity search using EfficientNet model.

    :param searchpath: images to be searched
    :param filepath: target images
    :param newfilepath: result
    :param threshold: similarity threshold
    :return: None
    """
    model = models.efficientnet_b2(pretrained=True)
    model.eval()
    newfilepath = os.path.join(newfilepath, "enetb2")

    for parent, _, filenames in os.walk(filepath):
        for filename in filenames:
            img2_path = os.path.join(filepath, filename)

            for _, _, search_filenames in os.walk(searchpath):
                for search_filename in search_filenames:
                    img1_path = os.path.join(searchpath, search_filename)
                    image1 = load_and_preprocess_image(img1_path)
                    image2 = load_and_preprocess_image(img2_path)
                    similarity = compute_similarity(image1, image2, model)

                    if similarity >= threshold:
                        print(f"Found similar image: {img1_path}, similarity: {similarity}")
                        shutil.copy(img1_path, newfilepath)


if __name__ == '__main__':
    searchpath = "./data/search"
    filepath = "./data/dataset"
    newfilepath = "./data/similar"
    threshold = 0.8

    efficientNetB2(searchpath, filepath, newfilepath, threshold)
