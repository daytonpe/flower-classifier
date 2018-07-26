# Flower Classifier -- Udacity AIPND

## Description
In this project, we'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories

## Usage - Training your network.

The **train.py** file allows you to train a neural network on a dataset. Basic usage is with the following command:

```python train.py [data_directory]```

A data directory must be supplied and it must have three subdirectories `test`, `valid`, `train` housing your pictures.

## Optional Tags

```--gpu-off```

Gpu is set to true unless this tag is given

```--save_dir```

Choose save directory for model

```--arch```

Choose between vgg and densenet architectures.

```--learning rate```

Define learning rate.

```--epochs```

Define number of epochs.

```--hidden_units```

Define number of hidden units. Ex. `--hidden_units 500,300,200`

```--dropout```

Define dropout rate.

#### Example command

```python train.py flowers --arch vgg --epochs 3 --dropout .3 --cat_to_name map.json```

---

## Usage - Predicting with your network.

The **predict.py** file allows you to classify an image via the neural network trained with **train.py**. Basic usage is with the following command:

```python predict.py [img_path] [checkpoint]```

The image path points to the picture you'd like to classify. The checkpoint points to the .pth file created by **predict.py** that houses your model.

#### Optional Tags

```--gpu-off```

Gpu is set to true unless this tag is given

```--cat_to_name```

Set category to name .json for model

```--topk```

define number of probable options returned


## Example Output - Predict

```
python predict.py flowers/ddl.jpg checkpoint.pth --topk 5

GPU?  True

Loading checkpoint...
Done


Loaded checkpoint from  checkpoint.pth

topk_prob_arr:  [  9.70751166e-01   2.91926507e-02   4.20815559e-05   6.94843493e-06   5.63376852e-06]
topk_class_arr:  ['common dandelion', "colt's foot", 'english marigold', 'marigold', 'buttercup']

Prediction: common dandelion with 97.1% probability.
```
```
python predict.py flowers/ddl.jpg checkpoint.pth --topk 5 --gpu-off

GPU?  False

Loading checkpoint...
Done


Loaded checkpoint from  checkpoint.pth

topk_prob_arr:  [  9.70751584e-01   2.91927140e-02   4.20815559e-05   6.94841492e-06   5.63376352e-06]
topk_class_arr:  ['common dandelion', "colt's foot", 'english marigold', 'marigold', 'buttercup']

Prediction: common dandelion with 97.1% probability.
```
```
python predict.py flowers/ddl.jpg checkpoint.pth

GPU?  True

Loading checkpoint...
Done


Loaded checkpoint from  checkpoint.pth

Prediction: common dandelion with 97.1% probability.
```
---

## Example Output - Train

#### Densenet201

```
python train.py flowers --epochs 10 --hidden_units 500,400 --arch densenet

GPU?  True

Training with Densenet201

/opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
Downloading: "https://download.pytorch.org/models/densenet201-c1103571.pth" to /root/.torch/models/densenet201-c1103571.pth
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 81131730/81131730 [00:02<00:00, 28585382.36it/s]

Beginning deep learning...
Epoch: 1/10..  Training Loss: 4.516..  Validation Loss: 4.379..  Validation Accuracy: 5.3%
Epoch: 1/10..  Training Loss: 4.336..  Validation Loss: 4.042..  Validation Accuracy: 12.4%
Epoch: 1/10..  Training Loss: 3.955..  Validation Loss: 3.399..  Validation Accuracy: 28.4%
Epoch: 1/10..  Training Loss: 3.376..  Validation Loss: 2.725..  Validation Accuracy: 37.5%
Epoch: 2/10..  Training Loss: 2.477..  Validation Loss: 2.221..  Validation Accuracy: 51.3%
Epoch: 2/10..  Training Loss: 2.524..  Validation Loss: 1.733..  Validation Accuracy: 59.7%
Epoch: 2/10..  Training Loss: 2.192..  Validation Loss: 1.476..  Validation Accuracy: 64.8%
Epoch: 2/10..  Training Loss: 1.990..  Validation Loss: 1.226..  Validation Accuracy: 69.8%
Epoch: 3/10..  Training Loss: 1.310..  Validation Loss: 1.069..  Validation Accuracy: 73.7%
Epoch: 3/10..  Training Loss: 1.583..  Validation Loss: 0.920..  Validation Accuracy: 78.6%
Epoch: 3/10..  Training Loss: 1.425..  Validation Loss: 0.853..  Validation Accuracy: 78.9%
Epoch: 3/10..  Training Loss: 1.474..  Validation Loss: 0.744..  Validation Accuracy: 81.3%
Epoch: 4/10..  Training Loss: 0.786..  Validation Loss: 0.707..  Validation Accuracy: 82.2%
Epoch: 4/10..  Training Loss: 1.200..  Validation Loss: 0.590..  Validation Accuracy: 86.0%
Epoch: 4/10..  Training Loss: 1.175..  Validation Loss: 0.604..  Validation Accuracy: 84.8%
Epoch: 4/10..  Training Loss: 1.134..  Validation Loss: 0.514..  Validation Accuracy: 88.6%
Epoch: 5/10..  Training Loss: 0.541..  Validation Loss: 0.531..  Validation Accuracy: 88.4%
Epoch: 5/10..  Training Loss: 1.041..  Validation Loss: 0.458..  Validation Accuracy: 89.7%
Epoch: 5/10..  Training Loss: 0.985..  Validation Loss: 0.475..  Validation Accuracy: 88.8%
Epoch: 5/10..  Training Loss: 0.993..  Validation Loss: 0.414..  Validation Accuracy: 90.4%
Epoch: 6/10..  Training Loss: 0.371..  Validation Loss: 0.439..  Validation Accuracy: 89.3%
Epoch: 6/10..  Training Loss: 0.864..  Validation Loss: 0.376..  Validation Accuracy: 90.3%
Epoch: 6/10..  Training Loss: 0.829..  Validation Loss: 0.342..  Validation Accuracy: 91.0%
Epoch: 6/10..  Training Loss: 0.839..  Validation Loss: 0.371..  Validation Accuracy: 89.5%
Epoch: 7/10..  Training Loss: 0.242..  Validation Loss: 0.359..  Validation Accuracy: 90.5%
Epoch: 7/10..  Training Loss: 0.763..  Validation Loss: 0.303..  Validation Accuracy: 92.7%
Epoch: 7/10..  Training Loss: 0.723..  Validation Loss: 0.315..  Validation Accuracy: 92.3%
Epoch: 7/10..  Training Loss: 0.831..  Validation Loss: 0.326..  Validation Accuracy: 91.5%
Epoch: 8/10..  Training Loss: 0.113..  Validation Loss: 0.344..  Validation Accuracy: 92.4%
Epoch: 8/10..  Training Loss: 0.775..  Validation Loss: 0.310..  Validation Accuracy: 91.6%
Epoch: 8/10..  Training Loss: 0.707..  Validation Loss: 0.266..  Validation Accuracy: 93.1%
Epoch: 8/10..  Training Loss: 0.681..  Validation Loss: 0.317..  Validation Accuracy: 92.2%
Epoch: 9/10..  Training Loss: 0.026..  Validation Loss: 0.290..  Validation Accuracy: 92.8%
Epoch: 9/10..  Training Loss: 0.682..  Validation Loss: 0.281..  Validation Accuracy: 92.2%
Epoch: 9/10..  Training Loss: 0.672..  Validation Loss: 0.273..  Validation Accuracy: 92.8%
Epoch: 9/10..  Training Loss: 0.643..  Validation Loss: 0.266..  Validation Accuracy: 93.1%
Epoch: 9/10..  Training Loss: 0.631..  Validation Loss: 0.237..  Validation Accuracy: 94.0%
Epoch: 10/10..  Training Loss: 0.586..  Validation Loss: 0.252..  Validation Accuracy: 93.4%
Epoch: 10/10..  Training Loss: 0.634..  Validation Loss: 0.240..  Validation Accuracy: 94.0%
Epoch: 10/10..  Training Loss: 0.629..  Validation Loss: 0.245..  Validation Accuracy: 93.7%
Epoch: 10/10..  Training Loss: 0.634..  Validation Loss: 0.246..  Validation Accuracy: 92.9%
Train time: 34m 5s

Checking accuracy with test data...

Beginning accuracy test...
0/0
42/64
86/128
116/192
150/256
196/320
239/384
286/448
330/512
368/576
406/640
442/704
488/768
Accuracy of the network on test images: 62 %
Accuracy test time: 0m 17s

Saving checkpoint...
Saved!
```
#### VGG16

```
python train.py flowers --epochs 10 --hidden_units 500,400 --arch vgg

GPU?  True


Training with Vgg16

Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.torch/models/vgg16-397923af.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 553433881/553433881 [00:18<00:00, 30353209.74it/s]

Beginning deep learning...
Epoch: 1/10..  Training Loss: 4.499..  Validation Loss: 3.717..  Validation Accuracy: 22.7%
Epoch: 1/10..  Training Loss: 3.807..  Validation Loss: 2.762..  Validation Accuracy: 40.5%
Epoch: 1/10..  Training Loss: 3.138..  Validation Loss: 2.044..  Validation Accuracy: 52.3%
Epoch: 1/10..  Training Loss: 2.713..  Validation Loss: 1.588..  Validation Accuracy: 60.6%
Epoch: 2/10..  Training Loss: 2.057..  Validation Loss: 1.330..  Validation Accuracy: 66.1%
Epoch: 2/10..  Training Loss: 2.015..  Validation Loss: 1.135..  Validation Accuracy: 69.5%
Epoch: 2/10..  Training Loss: 1.931..  Validation Loss: 0.934..  Validation Accuracy: 75.7%
Epoch: 2/10..  Training Loss: 1.880..  Validation Loss: 0.915..  Validation Accuracy: 76.3%
Epoch: 3/10..  Training Loss: 1.222..  Validation Loss: 0.798..  Validation Accuracy: 78.8%
Epoch: 3/10..  Training Loss: 1.641..  Validation Loss: 0.811..  Validation Accuracy: 79.7%
Epoch: 3/10..  Training Loss: 1.597..  Validation Loss: 0.726..  Validation Accuracy: 79.7%
Epoch: 3/10..  Training Loss: 1.495..  Validation Loss: 0.694..  Validation Accuracy: 82.9%
Epoch: 4/10..  Training Loss: 0.892..  Validation Loss: 0.615..  Validation Accuracy: 83.0%
Epoch: 4/10..  Training Loss: 1.369..  Validation Loss: 0.613..  Validation Accuracy: 83.6%
Epoch: 4/10..  Training Loss: 1.367..  Validation Loss: 0.616..  Validation Accuracy: 83.2%
Epoch: 4/10..  Training Loss: 1.325..  Validation Loss: 0.570..  Validation Accuracy: 84.2%
Epoch: 5/10..  Training Loss: 0.624..  Validation Loss: 0.533..  Validation Accuracy: 84.8%
Epoch: 5/10..  Training Loss: 1.204..  Validation Loss: 0.532..  Validation Accuracy: 84.8%
Epoch: 5/10..  Training Loss: 1.170..  Validation Loss: 0.513..  Validation Accuracy: 85.7%
Epoch: 5/10..  Training Loss: 1.186..  Validation Loss: 0.490..  Validation Accuracy: 86.4%
Epoch: 6/10..  Training Loss: 0.461..  Validation Loss: 0.502..  Validation Accuracy: 87.4%
Epoch: 6/10..  Training Loss: 1.085..  Validation Loss: 0.504..  Validation Accuracy: 86.6%
Epoch: 6/10..  Training Loss: 1.123..  Validation Loss: 0.458..  Validation Accuracy: 87.9%
Epoch: 6/10..  Training Loss: 1.120..  Validation Loss: 0.467..  Validation Accuracy: 88.2%
Epoch: 7/10..  Training Loss: 0.281..  Validation Loss: 0.471..  Validation Accuracy: 87.2%
Epoch: 7/10..  Training Loss: 1.117..  Validation Loss: 0.456..  Validation Accuracy: 89.0%
Epoch: 7/10..  Training Loss: 1.085..  Validation Loss: 0.476..  Validation Accuracy: 88.4%
Epoch: 7/10..  Training Loss: 1.071..  Validation Loss: 0.421..  Validation Accuracy: 89.8%
Epoch: 8/10..  Training Loss: 0.144..  Validation Loss: 0.392..  Validation Accuracy: 90.1%
Epoch: 8/10..  Training Loss: 0.984..  Validation Loss: 0.423..  Validation Accuracy: 88.7%
Epoch: 8/10..  Training Loss: 1.047..  Validation Loss: 0.386..  Validation Accuracy: 90.0%
Epoch: 8/10..  Training Loss: 0.991..  Validation Loss: 0.386..  Validation Accuracy: 89.7%
Epoch: 9/10..  Training Loss: 0.025..  Validation Loss: 0.391..  Validation Accuracy: 88.9%
Epoch: 9/10..  Training Loss: 0.968..  Validation Loss: 0.398..  Validation Accuracy: 88.9%
Epoch: 9/10..  Training Loss: 0.964..  Validation Loss: 0.417..  Validation Accuracy: 89.0%
Epoch: 9/10..  Training Loss: 0.985..  Validation Loss: 0.401..  Validation Accuracy: 90.5%
Epoch: 9/10..  Training Loss: 1.025..  Validation Loss: 0.361..  Validation Accuracy: 91.2%
Epoch: 10/10..  Training Loss: 0.886..  Validation Loss: 0.389..  Validation Accuracy: 90.9%
Epoch: 10/10..  Training Loss: 0.980..  Validation Loss: 0.363..  Validation Accuracy: 89.6%
Epoch: 10/10..  Training Loss: 0.946..  Validation Loss: 0.343..  Validation Accuracy: 91.5%
Epoch: 10/10..  Training Loss: 0.929..  Validation Loss: 0.369..  Validation Accuracy: 89.6%
Train time: 34m 57s

Checking accuracy with test data...

Beginning accuracy test...
0/0
46/64
98/128
149/192
193/256
247/320
292/384
351/448
400/512
459/576
518/640
561/704
613/768
Accuracy of the network on test images: 79 %
Accuracy test time: 0m 17s

Saving checkpoint...
Saved!
```
