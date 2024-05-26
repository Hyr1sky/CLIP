import torch
import clip
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Define a function to load and preprocess an image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)
    return image

def clip_img_score(img1_path, img2_path):
    # Load and preprocess the two images
    image_a = load_and_preprocess_image(img1_path)
    image_b = load_and_preprocess_image(img2_path)

    # Calculate the embeddings for the images using the CLIP model
    with torch.no_grad():
        embedding_a = model.encode_image(image_a)
        embedding_b = model.encode_image(image_b)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
    return similarity_score.item()

img1_path = 'D:\\VScode WorkStation\\CODE\\RandomCode\\E30M3.jpg'
img2_path = 'D:\\VScode WorkStation\\CODE\\RandomCode\\Seb.jpg'
score = clip_img_score(img1_path, img2_path)
print("Similarity score:", score)

# 0.802734375
# 0.3154296875