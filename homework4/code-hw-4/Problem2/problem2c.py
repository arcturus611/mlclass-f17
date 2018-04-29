#%% Load packages. 
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%% Load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%
def eval_on_data(datatype):
    # Function: Evaluate network performance on input data set
    # Input: Nothng, since we use loaded data available in workspace
    # Output: accuracy (in percentage)
    if (datatype=="train"):
        loader = trainloader
    else:
        loader = testloader
        
    correct = 0
    total = 0
    for data in loader:
        images, labels = data   
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    
    accu = 100 * correct / total
    print('Accuracy of the network on the 10000' + datatype + ' images: %d %%' %(accu))
    return accu

#%% Show some arbitrary images
def imshow(img):
    img = img / 2 + 0.5      
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#%% Define the conv neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, 5) #M = 100, p = 5
        self.pool = nn.MaxPool2d(14, 14)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(400, 10) 
        #self.fc2 = nn.Linear(100, 10)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 400)
        x = self.fc1(x)
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

#%% Instantiate the class
net = Net()

#%% Define a loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#%% Train and observe model on test set after each epoch
# store accuracies in arrays
num_epochs = 12
train_accu = np.empty(num_epochs)
test_accu = np.empty(num_epochs)

for epoch in range(num_epochs):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # print statistics
    train_accu[epoch] = eval_on_data("train")
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, train_accu))
        #running_loss = 0.0
    # print statistics
    test_accu[epoch] = eval_on_data("test")
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, test_accu))
        #running_loss = 0.0            

print('Finished Training')

#%% Test the model
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#%% Test on a few
outputs = net(Variable(images))
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

#%% plot stuff 
# final stuff (for part b)
sns.set()
fig = plt.figure(1)
fig, ax = plt.subplots(1, 1)
x_data = np.arange(num_epochs)+1
ax.plot(x_data, train_accu, 'b-.')
ax.plot(x_data, test_accu, 'r-.')
ax.set_xlim([1, 12])
#ax.set_ylim([-1.5, 41.5])
ax.legend(['train data', 'test data'])
ax.set_title('Neural Net: $W_2 \cdot \mathrm{vec} ( \mathrm{MaxPool} ( \mathrm{relu} (\mathrm{Conv2d}(x^{in}, W_1) + b_1) ) ) + b_2$')
ax.set_xlabel('Epoch #')
ax.set_ylabel('Accuracy in Percentage')



