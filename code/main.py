from util.data_preprocessing import preprocessing
from util.dataset import CVR
from torch.utils.data import DataLoader
from model.model import LogisticRegression
from config import configure
import torch.nn.functional as F
import torch

## train function
def train(model, train_loader, optimizer, CUDA=False):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        if CUDA:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.binary_cross_entropy(output.squeeze(), target)
        loss.retain_grad()
        loss.backward()
        with torch.no_grad():
            for k, p in enumerate(model.parameters()):
                update = optimizer.step(k, p, p.grad, loss)

## test function
def test(model, test_loader, CUDA):
    model.eval()
    corrects, total_loss = 0, 0
    for batch_index, (data, target) in enumerate(test_loader):
        if CUDA:
            data, target = data.cuda(), target.cuda()
        logit = model(data)
        loss = F.binary_cross_entropy(logit.squeeze(), target, reduction='sum')
        total_loss += loss.item()
        corrects += (torch.round(logit).view(target.size()).data == target.data).sum()
    size = len(test_loader.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

## load data
if configure['TRAIN']:
    print("Train Data Loading ...")
    train_data, train_label = preprocessing(data_range=configure['TRAIN_DATA'],augmentation=True)
    train_dataset = CVR(train_data,train_label)
    train_loader = DataLoader(train_dataset, batch_size=configure['BATCH'], shuffle=True)
    print("Train Data Loading Success!!")
print("Test Data Loading ...")
test_data, test_label = preprocessing(data_range=configure['TEST_DATA'],augmentation=True)
test_dataset = CVR(test_data,test_label)
test_loader = DataLoader(test_dataset, batch_size=configure['BATCH'], shuffle=True)
print("Test Data Loading Success!!")

## define model
model = LogisticRegression(22,1).cuda() if configure['CUDA'] else LogisticRegression(22,1)
if configure['LOAD']:
    model.load_state_dict(torch.load(configure['MODEL_LOAD_PATH']+configure['MODEL_LOAD_FILE']))
optimizer = torch.opti.Adam(model.parameters(), lr=0.0001)

## train
if configure['TRAIN']:
    for epoch in range(configure['EPOCH']):
        train(model, train_loader, optimizer, configure['CUDA'])
        test_loss, test_accuracy = test(model, test_loader, configure['CUDA'])
        print('[{}] Test Loss : {:.4f}, Accuracy : {:.2f}%'.format(epoch, test_loss, test_accuracy))

## test
else:
    test_loss, test_accuracy = test(model, test_loader, configure['CUDA'])
    print('Test Loss : {:.4f}, Accuracy : {:.2f}%'.format(test_loss, test_accuracy))

## save
if configure['SAVE']:
    torch.save(model.state_dict(),configure['MODEL_SAVE_PATH']+configure['MODEL_SAVE_FILE'])
