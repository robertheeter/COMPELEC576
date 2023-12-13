# for finalized model?

import os
import numpy as np
import pandas as pd
import scipy as sp
import torch
import torch.nn as nn
import torch_geometric as pyg
import matplotlib.pyplot as plt

import sklearn.metrics as skm
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score

from utils import *

from dataset import *

import random

class GCN(nn.Module):
    
    def __init__(self, hidden_channels, edge_dim):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = pyg.nn.GATv2Conv(39, hidden_channels, edge_dim=edge_dim)
        self.conv2 = pyg.nn.GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv3 = pyg.nn.GATv2Conv(hidden_channels, 2, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = nn.functional.dropout(x, p=0.2, training=True)
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = nn.functional.dropout(x, p=0.2, training=True)
        x = self.conv3(x, edge_index, edge_attr)
        
        return x


def main():

    data_dir = "data/processed"
    working_dir = os.getcwd()

    data_path = "data/processed/data_paths.csv"

    # # ignore split
    graph_path_list = make_dataset(data_path, data_dir, working_dir, data_type='data', radius_ncov=10, split=[0.8, 0.0, 0.2], num_process=64)

    random.shuffle(graph_path_list)
    train_graph_path_list = random.sample(graph_path_list, int(len(graph_path_list) * 0.75))
    test_graph_path_list = list(set(graph_path_list) - set(train_graph_path_list))
    # train_graph_path_list = graph_path_list[:40]
    # test_graph_path_list = graph_path_list[40:]

    train_list = []
    for train_path in train_graph_path_list:
        train_list.append(torch.load(train_path))

    test_list = []
    for test_path in test_graph_path_list:
        test_list.append(torch.load(train_path))

    data_example = train_list[0]

    train_loader = pyg.loader.DataLoader(train_list, batch_size=2, shuffle=False)
    test_loader = pyg.loader.DataLoader(test_list, batch_size=2, shuffle=False)


    model = GCN(hidden_channels=256, edge_dim=data_example.num_edge_features)
    print(model)


    model = GCN(hidden_channels=256, edge_dim=data_example.num_edge_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5) # add weight_decay?

    class_weight = torch.tensor([1.0, 9.0])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

    # TRAINING
    epochs = 300
    batch_loss = AverageMeter()
    batch_recall = AverageMeter()
    list_epoch_loss = []
    list_epoch_recall = []

    model.train()

    for epoch in range(epochs):
        for data in train_loader:

                pred = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(pred, data.y)
                recall = skm.recall_score(data.y, pred.argmax(dim=1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.update(loss.item(), data.num_nodes)
                batch_recall.update(recall, data.num_nodes)

                # print(f'- step_loss = {loss.item():.4f}')
        
        epoch_loss = batch_loss.get_average()
        epoch_recall = batch_recall.get_average()
        list_epoch_loss.append(epoch_loss)
        list_epoch_recall.append(epoch_recall)
        epoch_rmse = np.sqrt(epoch_loss)
        batch_loss.reset()
        batch_recall.reset()

        # add assessment of validation set performance

        print(f'epoch = {epoch:03d}, epoch_loss = {epoch_loss:.4f}')

    plt.figure()
    plt.plot(list_epoch_loss)
    plt.title('training loss v. epoch')
    plt.savefig('TRAINING_LOSS.png')

    plt.figure()
    plt.plot(list_epoch_recall)
    plt.title('training recall v. epoch')
    plt.savefig('TRAINING_RECALL.png')

    # TESTINGGGGG
    model.eval()
    y_pred = torch.empty((0), dtype=torch.int)
    y_true = torch.empty((0), dtype=torch.int)

    for data in test_loader:
        out = model(data.x, data.edge_index, data.edge_attr)
        
        y_pred = torch.cat((y_pred, out.argmax(dim=1)), axis=0) # use the class with highest probability
        y_true = torch.cat((y_true, data.y), axis=0)

# y_pred, y_true = test(test_loader)
# test_correct = (y_pred == y_true) # check against ground-truth labels
# test_accuracy = np.sum(test_correct) / len(test_correct) # derive ratio of correct predictions
# print(f'Accuracy:\n{test_accuracy}')
    test_confusion = skm.confusion_matrix(y_true, y_pred)
    test_recall = skm.recall_score(y_true, y_pred)
# balanced accuracy
# recall score
# f1 score
# precision

#75% train/25% validation/%0 test

# ONLY RECALL MATTERS AT THE MOMENT --> REPORT THE RECALL

# recall should increase for validation set

    print(f'CONFUSION MATRIX:\n{test_confusion}')
    print(f'RECALL:\n{test_recall}')

    with open("TEST_METRICS.txt", 'w') as txt:
        txt.write(f"confusion:\n{test_confusion}\n\nrecall:\n{test_recall}")

if __name__ == "__main__":
    main()