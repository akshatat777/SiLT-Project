import torch
from torch import nn
import numpy as np
from data_processing import read_batch, data_augmentation
from sign_recogn_joint import SignRecogJoint
from torch.utils.tensorboard import SummaryWriter

loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter()
minLoss = 1e9

def train(model, optimizer, batch_size, epochs):
    counter = 0
    for epoch in range(epochs):
        loss = 0
        acc = 0
        train_gen = read_batch(batch_size,True)
        test_gen = read_batch(batch_size,False)
        # creates a generator for batch data and iterates through it below
        for batch_i,(train_x_batch, truth) in enumerate(train_gen):
            train_x_batch = data_augmentation(train_x_batch).astype(np.float32)
            train_x_batch = torch.flatten(torch.tensor(train_x_batch).to('cpu'),start_dim=1)
            truth = torch.tensor(truth).to('cpu')
            preds = model(train_x_batch)
            # (N, 26)
            loss = loss_fn(preds, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                acc = torch.mean((torch.argmax(preds, dim=1) == truth).float())
                writer.add_scalar('loss/train', loss.detach().item(), counter)
                writer.add_scalar('acc/train', acc, counter)
                # calculates accuracy, graphes on tensorboard
                if batch_i%5==0:
                    print(f'train {epoch}, loss: {loss}, acc: {acc}')
                counter += 1
        with torch.no_grad():
            model.eval()
            eval(model, test_gen, epoch)
            model.train()

def eval(model, test_gen, epoch):
    # evaluates the loss using test data
    global minLoss
    loss, count, acc = 0,0,0
    for test_x_batch, truth in test_gen:
        test_x_batch = torch.flatten(torch.tensor(test_x_batch).to('cpu'),start_dim=1)
        truth = torch.tensor(truth).to('cpu')
        preds = model(test_x_batch) 
        loss += loss_fn(preds, truth)
        acc += torch.mean((torch.argmax(preds, dim=1) == truth).float())
        count += 1
    loss /= count
    acc /= count
    writer.add_scalar('loss/eval', loss.detach().item(), epoch)
    writer.add_scalar('acc/eval', acc, epoch)
    # calculates accuracy, graphes on tensorboard
    print(f'eval {epoch}, loss: {loss}, acc: {acc}')
    if loss < minLoss:
        minLoss = loss
        torch.save(model.state_dict(), 'sign_recogn_joint')
        # saves the model params whenever the loss goes below minLoss

     
epochs = 100
batch_size = 64
model = SignRecogJoint().to('cpu')
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
train(model, optimizer, batch_size, epochs)