import os
import shutil

from utils.similarity import calc_image_similarity

def run_similarity(searchpath, filepath, newfilepath, threshold1, threshold2):
    """
    Image similarity search using basic mathmatical functions.

    :param searchpath: images to be searched
    :param filepath: target images
    :param newfilepath: result
    :param threshold1: similarity threshold
    :param threshold2: similarity threshold
    :return: None
    """
    newfilepath = os.path.join(newfilepath, "base")
    
    for parent, dirnames, filenames in os.walk(searchpath):
        for srcfilename in filenames:
            img1_path = searchpath +"\\"+ srcfilename
            for parent, dirnames, filenames in os.walk(filepath):
                if srcfilename.endswith('.DS_Store'):
                    continue
                for i, filename in enumerate(filenames):
                    print("{}/{}: {} , {} ".format(i+1, len(filenames), srcfilename,filename))
                    img2_path = filepath + "\\" + filename
                    # 比较
                    kk = calc_image_similarity(img1_path, img2_path, threshold1, threshold2)
                    try:
                        if kk >= 0.5:
                            # 将两张照片同时拷贝到指定目录
                            shutil.copy(img2_path, os.path.join(newfilepath, srcfilename[:-4] + "_" + filename))  # 存储名称，可改
                    except Exception as e:
                        pass