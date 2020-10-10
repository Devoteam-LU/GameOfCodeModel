# -*- coding: utf-8 -*-
"""devolution.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PGxSs7O4m2W47ukUW0hrTZsgwuA3dNeb
"""

import pandas as pd
import pyodbc

conn = pyodbc.connect(DRIVER='{ODBC Driver 17 for SQL Server}',
                      SERVER='devohackaton.database.windows.net',
                      DATABASE='privilege',
                      UID='aadmin',
                      PWD='9rTuUjFcssBUGws2')

cursor = conn.cursor()

bdd_data = pd.read_sql_query('SELECT * FROM [privilege].[dbo].[userdetailtrains]', conn)
credit_scores = pd.read_sql_query(
    'SELECT [UserId],[Amount],[Probability] FROM [privilege].[dbo].[RepaymentProbabilities]', conn)

import torch
import torch.utils.data as data_utils

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# merge user data and creditscore
data = []
for index, row in credit_scores.iterrows():
    # print(row)
    # break
    client_detail = bdd_data[bdd_data.UserId == row["UserId"]]
    new_row = [client_detail.iloc[0][col] for col in bdd_data.head()]
    new_row += [row["Amount"], row["Probability"]]
    data.append(new_row)

data = pd.DataFrame(data, columns=list(bdd_data.head()) + ["Amount", "Probability"])
del data["CreatedByUserId"]
del data["UserId"]
del data["CreationDate"]
del data["Id"]

# preprocessing the data
data = data.replace("M", 0)
data = data.replace("F", 1)
data = data.replace(True, 1)
data = data.replace(False, 0)

normalize_data = False
if normalize_data:
    # normalisation
    from sklearn import preprocessing

    x = data.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled, columns=list(data.head()))

ratio = 0.75
n_train = int(ratio * data.shape[0])

input_size = data.shape[1] - 1

train_df = data.iloc[:n_train]
test_df = data.iloc[n_train:]


def create_dataset(dataframe):
    y = torch.tensor(dataframe['Probability'].values.astype(np.float32))
    x = torch.tensor(dataframe.drop('Probability', axis=1).values.astype(np.float32))
    dataset = data_utils.TensorDataset(x, y)
    return dataset


trainset = create_dataset(train_df)
testset = create_dataset(test_df)

mini_batch_size = n_train

trainloader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=mini_batch_size, shuffle=False, num_workers=2)


class Net(torch.nn.Module):
    def __init__(self, n_feature, h_sizes, out_size):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=n_feature)

        self.input = nn.Linear(n_feature, h_sizes[0])
        self.hidden = []
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        self.output = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.input(x))

        for layer in self.hidden:
            x = F.relu(layer(x))
        # x = F.softmax(self.output(x), dim=1)
        x = self.output(x)
        return x


def save_model():
    PATH = './model.pth'
    torch.save(net.state_dict(), PATH)


hidden_layer_size = [32, 16]
net = Net(input_size, hidden_layer_size, 1)
print(net)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # 0.2)
stopping_loss = 0.001

# load data from loader
writer = SummaryWriter()

for epoch in range(1000):  # loop over the dataset multiple times

    for i, (inputs, labels) in enumerate(trainloader):  # loop over minibatches
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        minibatch_loss = loss.item()

        PATH = './model.pth'
        torch.save(net.state_dict(), PATH)
        writer.add_scalar("minibatch_loss", minibatch_loss, epoch)


    if epoch % 100 == 0:
        for label, dataloader in [("train", trainloader), ("test", testloader)]:
            print("============ {} ===========".format(label))
            loss = 0
            for i, (inputs, labels) in enumerate(dataloader):  # loop over minibatches
                optimizer.zero_grad()
                outputs = net(inputs)
                for i_sample, (p, l) in enumerate(zip(outputs, labels)):
                    print("[epoch={}][{}] y={} y_pred={} loss={}".format(epoch, i_sample, l, p,criterion(l,p)))
                loss += criterion(outputs, labels)
            loss /= (i + 1)
            writer.add_scalar("Loss/{}".format(label), loss, epoch)
        save_model()

        # if running_loss < stopping_loss:
        #     break
writer.flush()

print('Finished Training')
save_model()

net = Net(input_size, hidden_layer_size, 1)
net.load_state_dict(torch.load(PATH))

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