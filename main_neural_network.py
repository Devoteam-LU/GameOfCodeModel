# -*- coding: utf-8 -*-
"""devolution.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PGxSs7O4m2W47ukUW0hrTZsgwuA3dNeb
"""

import pandas as pd
import pyodbc
import torch
import torch.utils.data as data_utils
from sklearn.tree import DecisionTreeRegressor
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from devolution_network import DevolutionNetwork
from setup_data import create_data
from utils import save_model, load_model

mean,std,n_train,input_size, trainset, testset, Xtrain, Ytrain, Xtest, Ytest = create_data()

mini_batch_size = n_train

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=mini_batch_size,
    shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=mini_batch_size,
    shuffle=False, num_workers=1)

hidden_layer_size = [32]
net = DevolutionNetwork(input_size, hidden_layer_size, 1, mean, std)
print(net)

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 0.2)
stopping_loss = 0.001

# load data from loader
writer = SummaryWriter()

for epoch in range(10000):  # loop over the dataset multiple times

    for i, (inputs, labels) in enumerate(trainloader):  # loop over minibatches
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        minibatch_loss = loss.item()

        save_model(net, "model")
        writer.add_scalar("minibatch_loss", minibatch_loss, epoch)

    if epoch % 100 == 0:
        for label, dataloader in [("train", trainloader), ("test", testloader)]:
            print("============ {} ===========".format(label))
            loss = 0
            for i, (inputs, labels) in enumerate(dataloader):  # loop over minibatches
                optimizer.zero_grad()
                outputs = net(inputs)
                for i_sample, (p, l) in enumerate(zip(outputs, labels)):
                    print("[epoch={}][{}] y={} y_pred={} loss={}"
                          .format(epoch, i_sample, l, p, criterion(l, p)))
                loss += criterion(outputs, labels)
            loss /= (i + 1)
            writer.add_scalar("Loss/{}".format(label), loss, epoch)
        save_model(net, "model")

        # if running_loss < stopping_loss:
        #     break
writer.flush()

print('Finished Training')
save_model(net, "model")
net = load_model("model")

writer.close()

# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.scatter(x.data.numpy(), y.data.numpy(), color = "orange")
# plt.plot(x.data.numpy(),net(x).data.numpy())
#
# plt.show()

# from google.colab import files
# files.download('model.pth')