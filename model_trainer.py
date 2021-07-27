import torch
from torch import nn
import numpy as np
from data_processing import read_data
from sign_recog_cnn import SignRecogCNN
from torch.utils.tensorboard import SummaryWriter

loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter()
minLoss = 1e9

def train(model, optimizer, batch_size, epochs):
    train_x, train_y, test_x, test_y = read_data()
    train_x = torch.tensor(train_x[:,None,...]).to('cpu')
    train_y = torch.tensor(train_y).to('cpu')
    test_x = torch.tensor(test_x[:,None,...]).to('cpu')
    test_y = torch.tensor(test_y).to('cpu')
    for epoch in range(epochs):
        indices = np.arange(len(train_y))
        loss = 0
        acc = 0
        for batch_i  in range(len(train_y)//batch_size):
            idx = indices[batch_i*batch_size : (batch_i+1)*batch_size]
            train_x_batch = train_x[idx]
            truth = train_y[idx]
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
                writer.add_scalar('loss/train', loss.detach().item(), epoch*(len(train_y)//batch_size)+batch_i)
                writer.add_scalar('acc/train', acc, epoch*(len(train_y)//batch_size)+batch_i)
                if batch_i%5==0:
                    print(f'train {epoch}, loss: {loss}, acc: {acc}')
        with torch.no_grad():
            model.eval()
            eval(model, test_x, test_y, epoch)
            model.train()

def eval(model, test_x, test_y, epoch):
    global minLoss
    preds = model(test_x) 
    loss = loss_fn(preds, test_y)
    acc = torch.mean((torch.argmax(preds, dim=1) == test_y).float())
    writer.add_scalar('loss/eval', loss.detach().item(), epoch)
    writer.add_scalar('acc/eval', acc, epoch)
    print(f'eval {epoch}, loss: {loss}, acc: {acc}')
    if loss < minLoss:
        minLoss = loss
        torch.save(model.state_dict(), 'sign_recogn_cnn')

model = SignRecogCNN().to('cpu')
epochs = 2
batch_size = 64
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
train(model, optimizer, batch_size, epochs)