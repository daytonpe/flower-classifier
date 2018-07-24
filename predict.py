import argparse
import json
from predict_helpers import load_checkpoint, predict

# Basic Usage: python predict.py /path/to/image checkpoint
# Test with: python predict.py flowers/valid/19/image_06165.jpg checkpoint.pth

# parse arguments
parser = argparse.ArgumentParser(description='Train a neural network.')
parser.add_argument('img_path', action="store")
parser.add_argument('checkpoint', action="store")
parser.add_argument(
    '--cat_to_name',
    dest='cat_to_name',
    default='cat_to_name.json',
    action='store',
    help='Set category to name .json for model')
parser.add_argument(
    '--topk',
    dest='topk',
    default='1',
    action='store',
    type=int,
    help='define number of probable options returned')
parser.add_argument(
    '--gpu-off',
    dest='gpu',
    default=True,
    action='store_false',
    help='enable gpu mode')
args = parser.parse_args()

print('\nGPU? ', args.gpu)

# add .pth if not present
if args.checkpoint[-4:] != '.pth':
    args.checkpoint = args.checkpoint + '.pth'

# create a category to name from json
with open(args.cat_to_name, 'r') as f:
    cat_to_name = json.load(f)

model = load_checkpoint(args.checkpoint)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

print('\nLoaded checkpoint from ', args.checkpoint, '\n')

prob_arr, class_arr = predict(args.img_path, model, args.topk, cat_to_name, args.gpu)

prediction = class_arr[0]
probability = round(prob_arr[0]*100, 1)

print('Prediction: {} with {}% probability.'.format(prediction, probability))
print('\n')
