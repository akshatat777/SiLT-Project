import torch
from torch import nn
import numpy as np
from data_processing import read_test_data
from data_processing import read_batch_train
from sign_recog_cnn import SignRecogCNN
from torch.utils.tensorboard import SummaryWriter

loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter()
minLoss = 1e9

def train(model, optimizer, batch_size, epochs):
    counter = 0
    test_x, test_y = read_test_data()
    test_x = np.moveaxis(test_x,-1,1).astype(np.float32)/255
    for epoch in range(epochs):
        loss = 0
        acc = 0
        gen = read_batch_train(batch_size)
        # creates a generator for batch data and iterates through it below
        for batch_i,(train_x_batch, truth) in enumerate(gen):
            train_x_batch = torch.tensor(train_x_batch).to('cpu')
            truth = torch.tensor(truth).to('cpu')
            #print(train_x_batch.shape)
            preds = model(train_x_batch)
            # (N, 24)
            loss = loss_fn(preds, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                #print(preds.shape)
                #print(preds.dtype)
                acc = torch.mean((torch.argmax(preds, dim=1) == truth).float())
                writer.add_scalar('loss/train', loss.detach().item(), counter)
                writer.add_scalar('acc/train', acc, counter)
                # calculates accuracy, graphes on tensorboard
                if batch_i%5==0:
                    print(f'train {epoch}, loss: {loss}, acc: {acc}')
                counter += 1
        with torch.no_grad():
            model.eval()
            eval(model, test_x, test_y, epoch)
            model.train()

def eval(model, test_x, test_y, epoch):
    # evaluates the loss using test data
    global minLoss
    preds = model(test_x) 
    loss = loss_fn(preds, test_y)
    acc = torch.mean((torch.argmax(preds, dim=1) == test_y).float())
    writer.add_scalar('loss/eval', loss.detach().item(), epoch)
    writer.add_scalar('acc/eval', acc, epoch)
    # calculates accuracy, graphes on tensorboard
    print(f'eval {epoch}, loss: {loss}, acc: {acc}')
    if loss < minLoss:
        minLoss = loss
        torch.save(model.state_dict(), 'sign_recogn_cnn')
        # saves the model params whenever the loss goes below minLoss

model = SignRecogCNN().to('cpu')
epochs = 1
batch_size = 64
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
train(model, optimizer, batch_size, epochs)