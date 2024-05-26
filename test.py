import os 
import shutil
from PIL import Image
from matplotlib import pyplot as plt

relative_path = os.path.dirname(__file__)

print(relative_path)

img_1 = "./assets/E30M3.jpg"
# show image
plt.imshow(Image.open(img_1))