import os
import shutil
import argparse

from utils.similarity import calc_image_similarity
from args.base_args import ImageSimilarityArguments
from models.base import run_similarity
from models.txt2img import clip_txt2img


if __name__ == '__main__':

    args = ImageSimilarityArguments().parse_args()
    filepath = args.dataset_path # 搜索文件夹
    searchpath = args.search_path #待查找文件夹
    newfilepath = args.output_path # 相似图片存放路径
    threshold1 = args.threshold1 # 融合相似度阈值
    threshold2 = args.threshold2 # 最终相似度较高判断阈值
    modelstype = args.model # 模型名称

    if modelstype == 'base':
        run_similarity(searchpath, filepath, newfilepath, threshold1, threshold2)
    elif modelstype == 'clip':
        clip_txt2img(searchpath, filepath, newfilepath)

    files_num = len(os.listdir(newfilepath))
    print("There are {} similar images in the folder".format(files_num))