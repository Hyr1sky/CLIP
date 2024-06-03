import os
import shutil
import argparse

from utils.similarity import calc_image_similarity
from args.base_args import ImageSimilarityArguments
from models.base import run_similarity
from models.txt2img import clip_txt2img
from models.img2img import clip_img2img
from models.towhee import towhee_similarity


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
        files_num = len(os.listdir(newfilepath+"/base"))
    elif modelstype == 'clip-txt':
        clip_txt2img(searchpath, filepath, newfilepath, threshold2)
        files_num = len(os.listdir(newfilepath+"/txt2img"))
    elif modelstype == 'clip-img':
        clip_img2img(searchpath, filepath, newfilepath, threshold2)
        files_num = len(os.listdir(newfilepath+"/img2img"))
    elif modelstype == 'towhee':
        towhee_similarity(searchpath, filepath, newfilepath, threshold1, threshold2)
        files_num = len(os.listdir(newfilepath+"/towhee"))

    print("There are {} similar images in the folder".format(files_num))