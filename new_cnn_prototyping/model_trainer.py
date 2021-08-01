from sign_recogn_cnn import SignRecogCNN
import torch
from torch import nn
import numpy as np
from img_proc_help import normalize, crop_hand_cnn


def read_batch(batch_size : int = 64, train : bool = True):
    if train:
        total_x, total_y = np.load('/kaggle/input/signrecogasl2/train_img.npy'), np.load('/kaggle/input/signrecogasl2/train_lab.npy')
    else:
        total_x, total_y = np.load('/kaggle/input/signrecogasl2/test_img.npy'), np.load('/kaggle/input/signrecogasl2/test_lab.npy')
    indices = np.arange(len(total_y))
    np.random.shuffle(indices)
    for batch_i in range(len(total_y)//batch_size):
        idx = indices[batch_i*batch_size : (batch_i+1)*batch_size]
        yield normalize(total_x[idx]), total_y[idx]

def data_aug(images):
    # N,3,224,224
    shifth = np.random.randint(-35,35)
    shiftv = np.random.randint(-35,35)
    images = np.roll(images,shifth,axis=-2)
    images = np.roll(images,shiftv,axis=-1)
    cshift1 = np.random.normal(0,0.2)
    cshift2 = np.random.normal(0,0.2)
    cshift3 = np.random.normal(0,0.2)
    images[:,0,:,:] += cshift1
    images[:,1,:,:] += cshift2
    images[:,2,:,:] += cshift3
    images = np.clip(images, 0, 1)
    return images

def eval(model, epoch, minLoss):
    # evaluates the loss using test data
    test_gen = read_batch(train=False)
    loss, count, acc = 0,0,0
    for test_x_batch, truth in test_gen:
        test_x_batch = torch.tensor(test_x_batch).to('cuda')
        truth = torch.tensor(truth).to('cuda')
        preds = model(test_x_batch) 
        loss += loss_fn(preds, truth)
        acc += torch.mean((torch.argmax(preds, dim=1) == truth).float())
        count += 1
    loss /= count
    acc /= count
    # calculates accuracy, graphes on tensorboard
    print(f'eval {epoch}, loss: {loss}, acc: {acc}')
    if loss < minLoss:
        minLoss = loss
        torch.save(model.state_dict(), 'sign_recogn_joint')
        # saves the model params whenever the loss goes below minLoss
    return minLoss

model = SignRecogCNN().to('cuda')
optimizer = torch.optim.Adam(model.parameters(),lr=2*1e-3)
loss_fn = nn.CrossEntropyLoss()
minLoss = 1e9
for epoch in range(100):
    print(f'training epoch {epoch}')
    train_gen = read_batch()
    for i, (imgs, labs) in enumerate(train_gen):
        imgs = data_aug(imgs)
        imgs = torch.tensor(imgs).to('cuda')
        labs = torch.tensor(labs).to('cuda')
        preds = model(imgs)
        # N, 26
        loss = loss_fn(preds, labs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            acc = torch.mean((torch.argmax(preds.detach(), dim=1) == labs).float()).detach().item()
            print('train loss,acc: ',loss.detach().item(), acc)
            if (i+1) % 10 == 0:
                model.eval()
                minLoss = eval(model, epoch, minLoss)
                model.train()
