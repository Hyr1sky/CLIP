import torch
import towhee
import os
import shutil
import numpy as np

from PIL import Image


def towhee_similarity(searchpath, filepath, newfilepath, threshold1, threshold2):
    """
    towhee image similarity search.
    It is a a cutting-edge framework designed to streamline 
    the processing of unstructured data through the use of 
    Large Language Model (LLM) based pipeline orchestration.

    :param searchpath: images to be searched
    :param filepath: target images
    :param newfilepath: result
    :param threshold1: similarity threshold
    :param threshold2: similarity threshold
    """




if __name__ == '__main__':
    searchpath = "data/search"
    filepath = "data/target"
    newfilepath = "data/output"
    threshold1 = 0.8
    threshold2 = 0.8

    towhee_similarity(searchpath, filepath, newfilepath, threshold1, threshold2)
