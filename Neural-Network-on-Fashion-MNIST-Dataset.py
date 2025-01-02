# -*- coding: utf-8 -*-
"""
## Part 1 - Neural Network using NumPy

### **Dataset **

The **MNIST (Modified National Institute of Standards and Technology database)** dataset contains a training set of 60,000 images and a test set of 10,000 images of handwritten digits (10 digits). The handwritten digit images have been size-normalized and centered in a fixed size of 28x28 pixels.


Download the MNIST dataset
"""

import numpy as np
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False, parser='auto')
print(X.shape, y.shape)

def min_max_norm(X):
    x_min = np.min(X)
    x_max = np.max(X)
    norm_x = (X - x_min) / (x_max - x_min)
    return norm_x


X = min_max_norm(X)

## TODO: Split the data into Train set (80%) and Test set (20%)

np.random.seed(42)

# Shuffle the indices
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_shuffled = X[indices]
y_shuffled = y[indices]

# Split the shuffled data into training and test sets
split_index = int(0.8 * X.shape[0])
X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:]
y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

"""### **Useful functions**"""

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Negative Log Likelihood loss function for multiclass
# Receive y_pred, which is a (10,) vector containing probability predictions, and y_true,
# which is a one-hot (10,) vector with the value of 1 only at the correct class index and zeros elsewhere.
def nll_loss(y_pred, y_true):
  loss = -np.sum(y_true * np.log(y_pred), dtype=np.float64)
  return loss / y_pred.shape[0]

"""### **Multilayer Perceptron**

Hyper-Parameters
"""

# Note that the MNIST dataset has 10 classes.
input_layer = 784  # 28X28 = 784
hidden_layer = 128
output_layer = 10
learning_rate = 0.01
epochs = 5

"""Multilayer Perceptron class with train and test functions."""

class MultilayerPerceptron:
    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate, epochs):


        self.W1 = np.random.randn(hidden_layer, input_layer) * 0.01
        self.b1 = np.zeros((hidden_layer, 1))
        self.W2 = np.random.randn(output_layer, hidden_layer) * 0.01
        self.b2 = np.zeros((output_layer, 1))

        # Store dimensions
        self.input_size = input_layer
        self.hidden_size = hidden_layer
        self.output_size = output_layer
        # Hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs

    def forward(self, X):

        Z1 = np.dot(self.W1, X) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = softmax(Z2)


        return Z1, A1, Z2, A2

    # Back propagation - compute the gradients of each parameter
    def backward(self, X, Y, Z1, A1, Z2, A2):

        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1



    def train(self, X_train, y_train):
        train_size = len(X_train)

        for epoch in range(self.epochs):
            total_epoch_loss = 0
            for i in range(train_size):
                # Reshape image i to be (784x1)
                x_i = X_train[i].reshape(-1, 1)
                Z1, A1, Z2, A2 = self.forward(x_i)


                y_true = np.zeros((10, 1))
                y_true[int(y_train[i])] = 1


                loss = nll_loss(A2, y_true)
                total_epoch_loss += loss
                self.backward(x_i, y_true, Z1, A1, Z2, A2)
            print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {total_epoch_loss / train_size}")

    def test(self, X_test, y_test):
        true_predictions = 0

        test_size = X_test.shape[0]  # Your code here
        for i in range(test_size):
            X_sample = X_test[i, :].reshape(-1, 1)
            Z1, A1, Z2, A2 = self.forward(X_sample)
            predicted_label = np.argmax(A2)
            true_label = int(y_test[i])
            if predicted_label == true_label:
                true_predictions += 1

        accuracy = true_predictions / test_size  # Your code here


        print(f"Accuracy: {accuracy}")

"""#### **Train**"""

mlp = MultilayerPerceptron(input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer,
                           learning_rate=learning_rate, epochs=epochs)

# Train your model
mlp.train(X_train, y_train)

"""#### **Test**

Accuracy should be more than **0.8** !
"""

mlp.test(X_test, y_test)

"""## Part 2 - Neural Network in PyTorch

____________
"""

# check python version
!python --version

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

# Enter the foldername in your Drive where you have saved the code and datasets.
# Recommended path: 'machine_learning/assignments/assignment5/'
FOLDERNAME = 'machine_learning/assignments/'
ASSIGNMENTNAME = 'assignment5'

# %cd drive/My\ Drive
# %cp -r $FOLDERNAME/$ASSIGNMENTNAME ../../
# %cd ../../

"""###  Dataset

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes - **'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'**.


Let's visualize the data before working with it.  
1. We can use the "torchvision" package to download the trainset. Set ```transform``` as to be the transform function below (It normalizes each image) and ```train=True```.
2. We use torch.utils.data.DataLoader to load the data. Set ```batch_size=64```.
"""

# load packages
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt

data_path = "./FashionMNIST_data"

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the data
mnist_data = datasets.FashionMNIST(data_path, download=True, train=True, transform=transform)
mnist_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=64, shuffle=True)

# get single batch
dataiter = iter(mnist_dataloader) # create itreable func
batch_images, batch_labels = next(dataiter) #get the next item in date

num_sampels=len(mnist_data)
print("the num of samples are:", num_sampels)

print("the num of samples in every batch is:", batch_images.size(0))

image_height, image_width = batch_images.shape[2], batch_images.shape[3]
print(f'the shape of images in the data is: {image_height} * {image_width}')

labels = mnist_dataloader.dataset.targets
unique_labels = set(labels.numpy())
print(f"Number of labels in the dataset: {len(unique_labels)}")

# 5. plot three images and print their labels
idx = np.random.choice(range(64),3) # three random indices
plt.subplot(1,3,1)
plt.imshow(batch_images[idx[0]].numpy().squeeze(), cmap='Greys_r')
plt.subplot(1,3,2)
plt.imshow(batch_images[idx[1]].numpy().squeeze(), cmap='Greys_r')
plt.subplot(1,3,3)
plt.imshow(batch_images[idx[2]].numpy().squeeze(), cmap='Greys_r')

# Define the label names array, where each label corresponds to its class, which is also its index
label_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print("Labels:", [label_names[i] for i in batch_labels[idx]])

"""### **3. Neural Network - Architecture **"""

from torch import nn, optim
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        '''
        Declare layers for the model
        '''
        super().__init__()

        self.fc1=nn.Linear(784,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        ''' Forward pass through the network, returns log_softmax values '''

        x1=F.relu(self.fc1(x))
        x2=F.relu(self.fc2(x1))
        x3=F.log_softmax(self.fc3(x2),dim=1)
        return x3

model = NeuralNetwork()
model

"""Now that we have a network, let's see what happens when we pass in an image.  
We'll choose a random image and pass it through the network. It should return a prediction - confidences for each class. The class with the highest confidence is the prediction of the model for that image.   
We visualize the results using the ```view_classify``` function below.
"""

def view_classify(img, ps, version="MNIST"):
    '''
    Function for viewing an image and its predicted classes.
    img - the input image to the network
    ps - the class confidences (network output)
    '''
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap='Greys_r')
    ax1.axis('off')

    # Setting the y-ticks for the bar chart
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))  # Numeric y-ticks for positioning

    # Using set_yticklabels to assign string labels to y-ticks
    if version == "MNIST":
        ax2.set_yticklabels(label_names)

    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()

def random_prediction_example(data_loader, model):
  '''
  The function sample an image from the data, pass it through the model (inference)
  and show the prediction visually. It returns the predictions confidences.
  '''
  # take a batch and randomly pick an image
  dataiter = iter(data_loader)
  images, labels = next(dataiter)
  images.resize_(64, 1, 784)
  img = images[0]

  # Forward pass through the network
  # we use torch.no_grad() for faster inference and to avoid gradients from
  # moving through the network.
  with torch.no_grad():
      ps = model(img)
      # the network outputs log-probabilities, so take exponential for probabilities
      ps = torch.exp(ps)

  # visualize image and prediction
  view_classify(img.view(1, 28, 28), ps)
  return ps

# you can run this cell multiple times for different images
preds_conf = random_prediction_example(mnist_dataloader, model)

"""Print the prediction"""

# TODO: Print the prediction of the network for that sample (preds_conf)
######################
### YOUR CODE HERE ###
######################
y=np.argmax(preds_conf)
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
print ("the predication of the network for the sample is :",class_labels[y])

"""### 4. Neural Network - Training"""

from torch.utils import data
from torch.utils.data import random_split

train_size = int(0.8 * len(mnist_data))
train_val=len(mnist_data)-train_size
train_dataset, val_dataset = random_split(mnist_data, [train_size,train_val])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

# 3. set hyper parameters
learning_rate = 0.005
nepochs = 5

model = NeuralNetwork()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

# 6.1. Train the model. (Fill empty code blocks)
def train_model(model, optimizer, criterion,
                nepochs, train_loader, val_loader, is_image_input = False):
  '''
  Train a pytorch model and evaluate it every epoch.
  Params:
  model - a pytorch model to train
  optimizer - an optimizer
  criterion - the criterion (loss function)
  nepochs - number of training epochs
  train_loader - dataloader for the trainset
  val_loader - dataloader for the valset
  is_image_input (default False) - If false, flatten 2d images into a 1d array.
                                Should be True for Neural Networks
                                but False for Convolutional Neural Networks.
  '''
  train_losses, val_losses = [], []
  for e in range(nepochs):
      running_loss = 0
      running_val_loss = 0
      for images, labels in train_loader:
          if is_image_input:
            # Flatten Fashion-MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

          # Training pass
          model.train() # set model in train mode


          optimizer.zero_grad()
          output=model(images)
          loss=criterion(model(images),labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
      else:
          val_loss = 0
          # 6.2 Evalaute model on validation at the end of each epoch.
          with torch.no_grad():
              for images, labels in val_loader:
                  if is_image_input:
                    # Flatten Fashion-MNIST images into a 784 long vector
                    images = images.view(images.shape[0], -1)
                  model.eval() # set model in evaluation mode


                  val_outputs = model(images)
                  val_loss = criterion(val_outputs,labels)
                  ######################

                  running_val_loss += val_loss.item()

          train_losses.append(running_loss/len(train_loader))
          val_losses.append(running_val_loss/len(val_loader))

          print("Epoch: {}/{}.. ".format(e+1, nepochs),
                "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                "Validation Loss: {:.3f}.. ".format(running_val_loss/len(val_loader)))
  return train_losses, val_losses

train_losses, val_losses = train_model(model, optimizer, criterion, nepochs,
                                       train_loader, val_loader, is_image_input=True)

"""If you implemented everything correctly, you should see the training loss drop with each epoch.

8. Plot train loss and validation loss as a function of epoch. **On the same graph!**
"""

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

"""With the network trained, we can check out it's predictions:"""

# you can run this cell multiple times for different images
ps = random_prediction_example(mnist_dataloader, model)

"""Calculate the model's accuracy on the validation-set."""

def evaluate_model(model, val_loader, is_image_input=False):
  '''
  Evaluate a model on the given dataloader.
  Params:
  model - a pytorch model to train
  val_loader - dataloader for the valset
  is_image_input (default False) - If false, flatten 2d images into a 1d array.
                                   Should be True for Neural Networks
                                   but False for Convolutional Neural Networks.
  '''
  validation_accuracy = 0
  with torch.no_grad():
      model.eval()
      for images, labels in val_loader:
          if is_image_input:
            # flatten Fashion-MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
          # forward pass
          test_output = model(images)
          ps = torch.exp(test_output)
          top_p, top_class = ps.topk(1, dim=1)
          # count correct predictions
          equals = top_class == labels.view(*top_class.shape)
          validation_accuracy += torch.sum(equals.type(torch.FloatTensor))
  res = validation_accuracy/len(val_loader.dataset)
  return res

print(f"Validation accuracy: {evaluate_model(model, val_loader, is_image_input=True)}")


class ConvolutionalNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x) # conv 1
        x = F.max_pool2d(x, 2) # max pooling 1
        x = F.relu(x) # relu
        x = self.conv2(x) # conv 2
        x = F.max_pool2d((x), 2) # max pooling 2
        x = F.relu(x) # relu
        x = x.view(-1, 320) # flatten input
        x = self.fc1(x) # hidden layer 1
        x = F.relu(x) # relu
        x = self.fc2(x) # hidden layer 2
        return F.log_softmax(x, dim=1) #output

cnn_model = ConvolutionalNet()
print(cnn_model)

"""We can now train the model on the train set."""

# set hyperparameters
cnn_nepochs = 3
cnn_learning_rate = 0.03

# train the conv model
cnn_model = ConvolutionalNet()
# create sgd optimizer
cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=cnn_learning_rate)
# create negative log likelihood loos
cnn_criterion = nn.NLLLoss()

train_losses, val_losses = train_model(cnn_model, cnn_optimizer, cnn_criterion,
                                       cnn_nepochs, train_loader, val_loader, is_image_input=False)

# evaluate on the validation set
print(f"Validation accuracy: {evaluate_model(cnn_model, val_loader, is_image_input=False)}")

"""___________

Now it's your turn: Use the code above and create a convolutional neural network that achieves the best accuracy on the validation set.
Note that you must use only the Fashion-MNIST training set as we did earlier:
```mnist_data = datasets.FashionMNIST(data_path, download=True, train=True, transform=transform)```

**Using any other or additional data from the test set will result in point deductions (without any bonus)!**

You should consider changing (change at least 4 things):  
1. Network architecture:
  - Number of convolutional layers
  - Number of kernels (filters) for each convolutional layer
  - Size of each kernel on each layer
  - Number of hidden layers (fully connected)
  - Number of units for each hidden layer
  - Usage of layers like BatchNormalization and Dropout.
  - Usage of max pooling (or maybe other pooling strategies)

2. Training hyperparameters:
  - Learning rate
  - Optimizer (SGD with momentum, adam, etc)
  - Number of epochs

To get full points in this part: make sure you implement the model architecture correctly, train the model properly, use the optimizer effectively, and ensure your validation accuracy is greater than 80%. **(4 points for this part)**
"""

######################
data_path = "./FashionMNIST_data_CNN"

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the data
mnist_data = datasets.FashionMNIST(data_path, download=True, train=True, transform=transform)
mnist_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=64, shuffle=True)

# split trainset into train (80%) and validation (20%)

train_size = int(0.8 * len(mnist_data))
train_val=len(mnist_data)-train_size
train_dataset, val_dataset = random_split(mnist_data, [train_size,train_val])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

class ConvolutionalNetworks(nn.Module):
    def __init__(self):
        super(ConvolutionalNetworks, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3) # change kernal size
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(20 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
       #padding,stride
        x = self.conv1(x) # conv 1
        x = self.bn1(x) # add batch normalizion
        x = F.relu(x) # relu
        x = F.max_pool2d(x, 2) # max pooling 1
        x = self.dropout(x)
        x = self.conv2(x) # conv 2
        x=  self.bn2(x)
        x = F.relu(x) # relu
        x = F.max_pool2d((x), 2) # max pooling 2
        x = self.dropout(x) # add dropout
        x = x.view(-1, 500) # flatten input
        x = self.fc1(x) # hidden layer 1
        x = F.relu(x) # relu
        x = self.dropout(x) # add dropout
        x = self.fc2(x) # hidden layer 2
        return F.log_softmax(x, dim=1) #output

cnn_nepochs = 5
learning_rate = 0.05
best_model= ConvolutionalNetworks()
optimizer = optim.SGD(best_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
train_losses, val_losses = train_model(best_model, optimizer, criterion,
                                       cnn_nepochs, train_loader, val_loader, is_image_input=False)
print(f"Validation accuracy: {evaluate_model(best_model,val_loader, is_image_input=False)}")

def predict_and_save(model, test_path, file_name):
  # load mnist test data
  mnist_test_data = torch.utils.data.TensorDataset(torch.load(test_path))

  transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
  # create a dataloader
  mnist_test_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=32, shuffle=False)
  # make a prediction for each batch and save all predictions in total_preds
  total_preds = torch.empty(0, dtype=torch.long)
  for imgs in mnist_test_loader:
    log_ps = model(imgs[0])
    ps = torch.exp(log_ps)
    _, top_class = ps.topk(1, dim=1)
    total_preds = torch.cat((total_preds, top_class.reshape(-1)))
  total_preds = total_preds.cpu().numpy()
  # write all predictions to a file
  with open(file_name,"w") as pred_f:
    for pred in total_preds:
      pred_f.write(str(pred) + "\n")

print(f"Validation accuracy: {evaluate_model(model, val_loader, is_image_input=True)}")