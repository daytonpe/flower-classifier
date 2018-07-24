import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.models as models
from torchvision import transforms
from train_helpers import check_accuracy_on_test, do_deep_learning, build_classifier

# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains

# parse arguments
parser = argparse.ArgumentParser(description='Train a neural network.')
# data_dir is required
parser.add_argument(
    'data_dir',
    action="store")
# assume the user wants to use the gpu
parser.add_argument(
    '--gpu-off',
    dest='gpu',
    default=True,
    action='store_false',
    help='enable gpu mode')
parser.add_argument(
    '--save_dir',
    dest='save_dir',
    default='/',
    action='store',
    help='choose save directory for model')
parser.add_argument(
    '--arch',
    dest='arch',
    default='densenet',
    action='store',
    help='choose architecture (densenet or vgg)')
parser.add_argument(
    '--learning_rate',
    dest='learning_rate',
    default=.0005,
    type=float,
    help='define the learning rate')
parser.add_argument(
    '--epochs',
    dest='epochs',
    default=5,
    action='store',
    type=int,
    help='define number of training epochs')
parser.add_argument(
    '--hidden_layers',
    dest='hidden_layers',
    default=3,
    action='store',
    type=int,
    help='define number of hidden layers')
parser.add_argument(
    '--dropout',
    dest='dropout',
    default=.5,
    action='store',
    type=float,
    help='define dropout rate')
args = parser.parse_args()

print('\nGPU? ', args.gpu)

# assume that the data_directory yields three folders (train, valid, test)
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
image_datasets = {}
image_datasets['train'] = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
image_datasets['test'] = torchvision.datasets.ImageFolder(test_dir, transform=test_transforms)
image_datasets['valid'] = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)


# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {}
dataloaders['trainloader'] = torch.utils.data.DataLoader(
    image_datasets['train'],
    batch_size=64,
    shuffle=True)
dataloaders['testloader'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
dataloaders['validloader'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)

# Load modelbased on args
if args.arch.upper() == "DENSENET":
    print('\nTraining with Densenet201\n')
    model = models.densenet201(pretrained=True)


elif args.arch.upper() == "VGG":
    args.arch = "VGG"
    print('\nTraining with Vgg16\n')
    model = models.vgg16(pretrained=True)

else:
    print('\nTraining with default architecture, Densenet201\n')
    model = models.densenet201(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = build_classifier(model, args.arch.upper(), args.hidden_layers, args.dropout)

# Train the classifier layers using backprop using the pre-trained network to get the features
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.gpu) else "cpu")

do_deep_learning(
    model,
    dataloaders['trainloader'],
    dataloaders['validloader'],
    args.epochs,
    25,
    criterion,
    optimizer,
    args.gpu
)

if args.gpu:
    model.to('cuda')
else:
    model.to('cpu')

# print('\nChecking accuracy with training data...')
# check_accuracy_on_test(dataloaders['trainloader'], model, args.gpu)

# CHECK ACCURACY ON VALIDATION SET
print('\nChecking accuracy with test data...')
check_accuracy_on_test(dataloaders['testloader'], model, args.gpu)

# SAVE TO CHECKPOINT
input_size = 1920 if args.arch.upper() == 'DENSENET' else 25088
checkpoint = {'input_size': input_size,
              'output_size': 102,
              'epochs': args.epochs,
              'classifier': model.classifier,
              'optimizer_state': optimizer.state_dict(),
              'mapping': image_datasets['train'].class_to_idx,
              'state_dict': model.state_dict()}

print('\nSaving checkpoint...')
torch.save(checkpoint, 'checkpoint1.pth')
print('Saved!\n')
