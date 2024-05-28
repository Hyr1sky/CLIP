import torch
import clip
import numpy as np
from PIL import Image


def clip_img2img(searchpath, filepath, newfilepath):
    """
    Next goal, maybe we can determin the target image by selecting
    file in input box.

    :param searchpath: source image path
    :param filepath: target image path
    :param newfilepath: output path
    :return: None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

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

if __name__ == '__main__':
    clip_img2img()