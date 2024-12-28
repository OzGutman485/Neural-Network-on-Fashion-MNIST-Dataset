import numpy as np
from sklearn.datasets import fetch_openml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

X, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False, parser='auto')
print(X.shape, y.shape)


## TODO: Normalize the dataset according to Min-Max normalization.
def min_max_norm(X):
    x_min = np.min(X)
    x_max = np.max(X)
    norm_x = (X - x_min) / (x_max - x_min)
    return norm_x


X = min_max_norm(X)

## TODO: Split the data into Train set (80%) and Test set (20%)
X_train, X_test = X[:56000, :], X[56000:70000, :]  # Your code here
y_train, y_test = y[:56000], y[56000:70000]  # Your code here

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

## TODO: Implement the sigmoid activation function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

## TODO: Implement the softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Negative Log Likelihood loss function for multiclass
# Receive y_pred, which is a (10,) vector containing probability predictions, and y_true,
# which is a one-hot (10,) vector with the value of 1 only at the correct class index and zeros elsewhere.
def nll_loss(y_pred, y_true):
    loss = -np.sum(y_true * np.log(y_pred), dtype=np.float64)
    return loss / y_pred.shape[0]

## TODO: Define the hyper-parameters that you will need for your model
# Note that the MNIST dataset has 10 classes.
input_layer = 784  # 28X28 = 784
hidden_layer = 128
output_layer = 10
learning_rate = 0.01
epochs = 5

# Your code here

class MultilayerPerceptron:
    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate, epochs):
        ## TODO: Initialize the parameters.
        ## Note that the MNIST dataset has 10 classes.

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
        ## TODO: Implement the forward pass.

        ######################
        ### YOUR CODE HERE ###
        ######################
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = softmax(Z2)
        # Your code here
        ######################

        return Z1, A1, Z2, A2

    # Back propagation - compute the gradients of each parameter
    def backward(self, X, Y, Z1, A1, Z2, A2):
        ## TODO: Implement the backward pass, Back propagation - compute the gradients of each parameter.
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

    ######################

    def train(self, X_train, y_train):
        train_size = len(X_train)

        for epoch in range(self.epochs):
            total_epoch_loss = 0
            for i in range(train_size):
                # Reshape image i to be (784x1)
                x_i = X_train[i].reshape(-1, 1)
                Z1, A1, Z2, A2 = self.forward(x_i)

                # TODO: Define y_true as a zero vector with the shape (10, 1),
                # and place a '1' at the index specified by int(y_train[i]).
                # For example when int(y_train[i])=4, then y_true should be (0,0,0,0,1,0,0,0,0,0)
                y_true = np.zeros((10, 1))
                y_true[int(y_train[i])] = 1
                ######################

                loss = nll_loss(A2, y_true)
                total_epoch_loss += loss
                self.backward(x_i, y_true, Z1, A1, Z2, A2)
            print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {total_epoch_loss / train_size}")

    def test(self, X_test, y_test):
        true_predictions = 0
        # TODO: test your model and print the accuracy on the test set
        ######################
        ### YOUR CODE HERE ###
        ######################
        test_size = X_test.shape[0]  # Your code here
        for i in range(test_size):
            X_sample = X_test[i, :].reshape(-1, 1)
            Z1, A1, Z2, A2 = self.forward(X_sample)
            predicted_label = np.argmax(A2)
            true_label = int(y_test[i])
            if predicted_label == true_label:
                true_predictions += 1
            # Your code here
        accuracy = true_predictions / test_size  # Your code here
        ######################

        print(f"Accuracy: {accuracy}")


# TODO: Define your mlp model with all the parameters

mlp = MultilayerPerceptron(input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer,
                           learning_rate=learning_rate, epochs=epochs)

# Train your model
mlp.train(X_train, y_train)

mlp.test(X_test, y_test)

# check python version
!python --version

from google.colab import drive

drive.mount('/content/drive', force_remount=True)

# Enter the foldername in your Drive where you have saved the code and datasets.
# Recommended path: 'machine_learning/assignments/assignment5/'
FOLDERNAME = 'machine_learning/assignments/'
ASSIGNMENTNAME = 'assignment5'

%cd drive/My\ Drive
%cp -r $FOLDERNAME/$ASSIGNMENTNAME ../../
%cd ../../

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
dataiter = iter(mnist_dataloader)
batch_images, batch_labels = next(dataiter)

# TODO: 1. Print the number of samples in the whole dataset.
######################
### YOUR CODE HERE ###
######################

# TODO: 2. Print the number of samples in a single batch.
######################
### YOUR CODE HERE ###
######################

# TODO: 3. Print the shape of images in the data (image dimensions).
######################
### YOUR CODE HERE ###
######################

# TODO: 4. Print the number of labels in the whole dataset (using the targets in the dataloader.dataset).
######################
### YOUR CODE HERE ###
######################


# 5. plot three images and print their labels
idx = np.random.choice(range(64), 3)  # three random indices
plt.subplot(1, 3, 1)
plt.imshow(batch_images[idx[0]].numpy().squeeze(), cmap='Greys_r')
plt.subplot(1, 3, 2)
plt.imshow(batch_images[idx[1]].numpy().squeeze(), cmap='Greys_r')
plt.subplot(1, 3, 3)
plt.imshow(batch_images[idx[2]].numpy().squeeze(), cmap='Greys_r')

# Define the label names array, where each label corresponds to its class, which is also its index
label_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print("Labels:", [label_names[i] for i in batch_labels[idx]])

from torch import nn, optim
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        '''
        Declare layers for the model
        '''
        super().__init__()
        # TODO
        ######################
        ### YOUR CODE HERE ###
        ######################
        pass

    def forward(self, x):
        ''' Forward pass through the network, returns log_softmax values '''
        # TODO
        ######################
        ### YOUR CODE HERE ###
        ######################
        return None


model = NeuralNetwork()
model

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

# TODO: Print the prediction of the network for that sample (preds_conf)
######################
### YOUR CODE HERE ###
######################

from torch.utils import data

# TODO: 1. split trainset into train and validation (use torch.utils.data.random_split())
######################
### YOUR CODE HERE ###
######################

# TODO: 2.1. create data loader for the trainset (batch_size=64, shuffle=True)
######################
### YOUR CODE HERE ###
######################
# train_loader = ...

# TODO: 2.2. create data loader for the valset (batch_size=64, shuffle=False)
######################
### YOUR CODE HERE ###
######################
# val_loader = ...

# 3. set hyper parameters
learning_rate = 0.005
nepochs = 5

model = NeuralNetwork()

# TODO: 4. create sgd optimizer. It should optimize our model parameters with
#    learning_rate defined above
######################
### YOUR CODE HERE ###
######################
# optimizer = ...

# TODO: 5. create a criterion object. It should be negative log-likelihood loss since the task
#    is a multi-task classification (digits classification)
######################
### YOUR CODE HERE ###
######################
# criterion = ...

# 6.1. Train the model. (Fill empty code blocks)
def train_model(model, optimizer, criterion,
                nepochs, train_loader, val_loader, is_image_input=False):
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
        model.train()  # set model in train mode

        # TODO: Compute the loss and do the backward and optimizer step
        ######################
        ### YOUR CODE HERE ###
        ######################

        running_loss += loss.item()
    else:
        val_loss = 0
        # 6.2 Evalaute model on validation at the end of each epoch.
        with torch.no_grad():
            for images, labels in val_loader:
                if is_image_input:
                    # Flatten Fashion-MNIST images into a 784 long vector
                    images = images.view(images.shape[0], -1)
                model.eval()  # set model in evaluation mode

                # TODO: compute the Validation Loss
                ######################
                ### YOUR CODE HERE ###
                ######################
                # val_loss = ...
                ######################

                running_val_loss += val_loss.item()

        # 7. track train loss and validation loss
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(running_val_loss / len(val_loader))

        print("Epoch: {}/{}.. ".format(e + 1, nepochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(running_val_loss / len(val_loader)))
return train_losses, val_losses

  # 6.1. Train the model.
## NOTE: Do not run this cell continuously without running the two cells above!
##       Otherwise, you might train a model you have already trained.
##       So make sure to run the two cells above (to first initialize the model
##       and optimizer), every time, before running this cell!
train_losses, val_losses = train_model(model, optimizer, criterion, nepochs,
                                       train_loader, val_loader, is_image_input=True)

# TODO: plot train and validation loss as a function of # epochs
######################
### YOUR CODE HERE ###
######################

# you can run this cell multiple times for different images
ps = random_prediction_example(mnist_dataloader, model)

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
    res = validation_accuracy / len(val_loader.dataset)
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
        x = self.conv1(x)  # conv 1
        x = F.max_pool2d(x, 2)  # max pooling 1
        x = F.relu(x)  # relu
        x = self.conv2(x)  # conv 2
        x = F.max_pool2d((x), 2)  # max pooling 2
        x = F.relu(x)  # relu
        x = x.view(-1, 320)  # flatten input
        x = self.fc1(x)  # hidden layer 1
        x = F.relu(x)  # relu
        x = self.fc2(x)  # hidden layer 2
        return F.log_softmax(x, dim=1)  #output


cnn_model = ConvolutionalNet()
print(cnn_model)

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

## TODO: Prepocess
######################
### YOUR CODE HERE ###
######################
data_path = "./FashionMNIST_data_CNN"

# Define a transform to normalize the data
# transform = transforms.Compose([transforms.ToTensor(),
#                               ...
#                               ])

# Download and load the data
mnist_data = datasets.FashionMNIST(data_path, download=True, train=True, transform=transform)
mnist_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=64, shuffle=True)

# split trainset into train (80%) and validation (20%)
# ...
# train_loader = ...
# val_loader = ...

# TODO: Find network and hyperparams that achieve best validation accuracy as possible
######################
### YOUR CODE HERE ###
######################

# TODO: Save the best model in this variable and evaluate on the validation set
######################
### YOUR CODE HERE ###
######################
#best_model = ...
#print(f"Validation accuracy: {evaluate_model(best_model, val_loader, is_image_input=False)}")

def predict_and_save(model, test_path, file_name):
    # load mnist test data
    mnist_test_data = torch.utils.data.TensorDataset(torch.load(test_path))

    ## TODO: Prepocess the test set (i.e apply the transform to normalize the test set as you did to your train set)
    ######################
    ### YOUR CODE HERE ###
    ######################

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
    with open(file_name, "w") as pred_f:
        for pred in total_preds:
            pred_f.write(str(pred) + "\n")


predict_and_save(best_model, test_path=f"{ASSIGNMENTNAME}/FashionMNIST_test.pth", file_name="predictions.txt")