import torch
import numpy as np
from PIL import Image


def load_checkpoint(filepath, model):
    print('\nLoading checkpoint...')
    checkpoint = torch.load(filepath)
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    model.epochs = checkpoint['epochs']
    model.classifier = checkpoint['classifier']
    model.optimizer_state = checkpoint['optimizer_state']
    model.mapping = checkpoint['mapping']
    model.state_dict = checkpoint['state_dict']
    model.load_state_dict(checkpoint['state_dict'])
    print('Done\n')
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    width = int(image.size[0])
    height = int(image.size[1])
    ratio = float(width)/float(height)

    if (width > height):
        size = int(256*ratio), int(256)  # width, height
    elif (width < height):
        size = int(256), int(256*ratio)
    else:
        size = int(256), int(256)

    width, height = size  # reset width & height
    image = image.resize(size)

    left = (width - 224)/2
    right = (width + 224)/2
    top = (height - 224)/2
    bottom = (height + 224)/2

    image = image.crop((left, top, right, bottom))

    np_image = np.array(image)
    np_image = np_image/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std

    np_image = np_image.transpose((2, 0, 1))
    return np_image


def predict(image_path, model, topk, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print('Predicting classification for image...')
    model.eval()
    model.cpu()

    # convert from these indices to the actual class labels
    idx_to_class = {i: k for k, i in model.mapping.items()}

    # Load and Process Image
    with Image.open(image_path) as image:
        image = process_image(image)

    # Switch it to a float tensor
    image = torch.FloatTensor([image])

    # Feed it through the model
    output = model.forward(image)

    # Determine topk probabilities and labels
    topk_prob, topk_labels = torch.topk(output, topk)

    # Take exp() of image to cancel out the LogSoftMax
    topk_prob = topk_prob.exp()

    # Assemble the lists
    topk_prob_arr = topk_prob.data.numpy()[0]
    topk_indexes_list = topk_labels.data.numpy()[0].tolist()
    topk_labels_list = [idx_to_class[x] for x in topk_indexes_list]
    topk_class_arr = [cat_to_name[str(x)] for x in topk_labels_list]

    # Display topk if given as param
    if topk > 1:
        print('topk_prob_arr: ', topk_prob_arr)
        print('topk_class_arr: ', topk_class_arr)
    print('Done\n')

    return topk_prob_arr, topk_class_arr
