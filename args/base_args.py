import argparse
import yaml
import sys
import importlib

class ImageSimilarityArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Image similarity comparison')
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            '--dataset_path',
            default='./data/dataset',
            help='Path to the dataset folder'
        )
        self.parser.add_argument(
            '--search_path',
            default='./data/search',
            help='Path to the directory containing images to search'
        )
        self.parser.add_argument(
            '--output_path',
            default='./data/similar',
            help='Path to store similar images'
        )
        self.parser.add_argument(
            '--threshold1',
            type=float,
            default=0.70,
            help='Fusion similarity threshold'
        )
        self.parser.add_argument(
            '--threshold2',
            type=float,
            default=0.95,
            help='Final similarity high judgment threshold'
        )

        self.parser.add_argument(
            '--model',
            default='base', 
            help='Name of the model to use for similarity search'
        )

        self.initialized = True

    def parse_args(self):
        if not self.initialized:
            self.initialize()
        return self.parser.parse_args()

def load_model_args(model_name):
    model_module = importlib.import_module(f'models.{model_name}')
    model_args = model_module.get_args(argparse.ArgumentParser())
    return model_args

if __name__ == '__main__':
    args = ImageSimilarityArguments().parse_args()
    model_args = load_model_args(args.model)
    print(args)
    print(model_args)