import argparse
from helpers.alexnet import AlexNet

def get_trained_model(args):
    """
    Return the model with learnt weights for inference tasks
    """
    height, width = args.input_shape
    classification_threshold = 0.5
    model = AlexNet.build(width=width, height=height, depth=7, classes=1, reg=0.0002)
    model.load_weights(args.modelpath)
    return model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input-shape', '--input-shape', nargs='+', type=int, default=(512,512))
    return parser
