import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

def MyModel(x):
    return -4 * x * x + x * 3 + 4

x = torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1)  # x data (tensor), shape=(100, 1)
y = MyModel(x) + 0.4*torch.rand(x.size())              # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidde2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = F.relu(self.hidde2(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        a = torch.rand(1)
        b = torch.rand(1)
        c = torch.rand(1)
        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)
        self.c = torch.nn.Parameter(c)

    def forward(self, x):
        x = x * x * self.a + x * self.b + self.c
        return x
    
    def __str__(self):
        return f'a={self.a.data}, b={self.b.data}, c={self.c.data}'

# net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
net = Model()     # define the network
# print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

loss_list = []
t = 0
while True:
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    loss_list.append(loss.data.numpy())
    print(t, loss.data, net)
    t = t + 1
    loss_decreased = True
    if len(loss_list) > 20:
        loss_sum = (np.array(loss_list[-20:-10]) - np.array(loss_list[-10:])).sum()
        print(loss_sum)
        loss_decreased = abs(loss_sum) > 1e-7
    print(loss_decreased)
    if not loss_decreased or t > 1000:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.01)
        break


plt.ioff()
plt.show()