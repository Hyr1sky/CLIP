import torch
import os
import argparse
import shutil

from PIL import Image
from torchvision import transforms
from network.elicNet import TestModel
from utils.Inference import psnr


def load_and_preprocess_image(image_path, patch_size=(256, 256)):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.CenterCrop(patch_size),
        transforms.ToTensor(),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

def ELiC(searchpath, filepath, newfilepath, threshold):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./checkpoints/checkpoint_last_1.pth.tar"
    model = TestModel()  # Assuming TestModel is your model for image compression
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    if not os.path.exists(newfilepath):
        os.makedirs(newfilepath)
    
    for filename in os.listdir(filepath):
        img2_path = os.path.join(filepath, filename)

        for search_filename in os.listdir(searchpath):
            img1_path = os.path.join(searchpath, search_filename)
            image1 = load_and_preprocess_image(img1_path)
            image2 = load_and_preprocess_image(img2_path)
            image1 = image1.to(device)
            image2 = image2.to(device)
            
            with torch.no_grad():
                output1 = model(image1)
                output2 = model(image2)
            
            psnr_val = psnr(output1['x_hat'], output2['x_hat'])
            
            if psnr_val >= threshold:
                print(f"Found similar image: {img1_path}, PSNR: {psnr_val}")
                shutil.copy(img1_path, newfilepath)

if __name__ == '__main__':
    searchpath = "./data/search"
    filepath = "./data/dataset"
    newfilepath = "./data/similar"
    threshold = 20

    ELiC(searchpath, filepath, newfilepath, threshold)
