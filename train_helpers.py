import torch
from torch import nn
import time
import numpy as np


def check_accuracy_on_test(testloader, model, gpu):
    print('\nBeginning accuracy test...')
    t1 = time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            print('{}/{}'.format(correct, total))
            images, labels = data
            if gpu:
                images = images.to('cuda')
                labels = labels.to('cuda')
            else:
                images = images.to('cpu')
                labels = labels.to('cpu')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
    t_total = time.time() - t1
    print('Accuracy test time: {:.0f}m {:.0f}s'.format(
        t_total // 60, t_total % 60))


# Implement a function for the validation pass
def validation(model, validloader, criterion, gpu):
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')
        else:
            images, labels = images.data.cpu(), labels.data.cpu()

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def do_deep_learning(
    model,
    trainloader,
    validloader,
    epochs,
    print_every,
    criterion,
    optimizer,
    gpu
):
    print('\nBeginning deep learning...')
    t1 = time.time()
    steps = 0

    # change to cuda
    if(gpu):
        model.to('cuda')
    else:
        model.to('cpu')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if gpu:
                inputs, labels, criterion = inputs.to('cuda'), labels.to('cuda'), criterion.cuda()
            else:
                inputs, labels = inputs.data.cpu(), labels.data.cpu()

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, gpu)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "Validation Accuracy: {:.1f}%".format(100 * accuracy/len(validloader)))
                running_loss = 0

                # Make sure training is back on
                model.train()

                running_loss = 0
    t_total = time.time() - t1
    print('Train time: {:.0f}m {:.0f}s'.format(
        t_total // 60, t_total % 60))


def build_classifier(model, model_type, hidden_layers, dropout):
    # build start layer for models based on model model_type
    if model_type == 'VGG':

        # create a linear spaced array of nodes for hidden layers
        node_list = np.linspace(1920, 102, hidden_layers+2).astype(int).tolist()

        # configure our layers
        first_layer = [
            nn.Linear(25088, node_list[1]),
            nn.Dropout(p=dropout),
            nn.ReLU()
            ]

        middle_layers = []
        for idx, val in enumerate(node_list[1:-2]):
            middle_layers.append(nn.Linear(node_list[idx+1], node_list[idx+2]))
            middle_layers.append(nn.Dropout(p=dropout))
            middle_layers.append(nn.ReLU())

        last_layer = [
            nn.Linear(node_list[-2], 102),
            nn.LogSoftmax(dim=1)
            ]

        first_layer = first_layer+middle_layers+last_layer
        layers = nn.ModuleList(first_layer)
        return nn.Sequential(*layers)

    else:
        # create a linear spaced array of nodes for hidden layers
        node_list = np.linspace(1920, 102, hidden_layers+2).astype(int).tolist()

        # configure our layers
        first_layer = [
            nn.Linear(1920, node_list[1]),
            nn.Dropout(p=dropout),
            nn.ReLU()
            ]

        middle_layers = []
        for idx, val in enumerate(node_list[1:-2]):
            middle_layers.append(nn.Linear(node_list[idx+1], node_list[idx+2]))
            middle_layers.append(nn.Dropout(p=dropout))
            middle_layers.append(nn.ReLU())

        last_layer = [
            nn.Linear(node_list[-2], 102),
            nn.LogSoftmax(dim=1)
            ]

        first_layer = first_layer+middle_layers+last_layer
        layers = nn.ModuleList(first_layer)
        return nn.Sequential(*layers)
