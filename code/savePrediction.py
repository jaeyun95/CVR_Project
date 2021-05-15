from util.data_preprocessing import preprocessing
from util.dataset import CVR
from model.model import LogisticRegression
from config import configure
import torch
import json

## load data
train_data, train_label = preprocessing(data_range=configure['TRAIN_DATA'],augmentation=False)
test_data, test_label = preprocessing(data_range=configure['TEST_DATA'],augmentation=False)
train_dataset, test_dataset = CVR(train_data,train_label), CVR(test_data,test_label)

## load model
model = LogisticRegression(22,1).cuda() if configure['CUDA'] else LogisticRegression(22,1)
model.load_state_dict(torch.load(configure['MODEL_LOAD_PATH']+configure['MODEL_LOAD_FILE']))

## set file name and dictionary
train_result = configure['MODEL_SAVE_PATH']+'train_result.txt'
test_result = configure['MODEL_SAVE_PATH']+'test_result.txt'
train_list = []
test_list = []

## get train data result
model.eval()
for i in range(configure['TRAIN_DATA'][0],configure['TRAIN_DATA'][1]):
    data, label = train_dataset[i]
    logit = model(data.cuda())
    if (torch.round(logit).view(label.size()).data == label.data): predicted_probability = float(logit[0].cpu().detach().item())
    else: predicted_probability = 1 - float(logit[0].cpu().detach().item())
    print('<Line Number>', i + 1, '<Target Label>', int(label.detach().item()), '<Predicted Probability>',predicted_probability)
    train_list.append({'<Line Number>':i+1,'<Target Label>':int(label.detach().item()),'<Predicted Probability>':predicted_probability})

## save
with open(train_result,'w') as file:
    file.write(json.dumps(train_list))

## get test data result
for i in range(configure['TEST_DATA'][0],configure['TEST_DATA'][1]):
    data, label = test_dataset[i-configure['TEST_DATA'][0]]
    logit = model(data.cuda())
    if (torch.round(logit).view(label.size()).data == label.data): predicted_probability = float(logit[0].cpu().detach().item())
    else: predicted_probability = 1 - float(logit[0].cpu().detach().item())
    print('<Line Number>',i+1,'<Target Label>',int(label.detach().item()),'<Predicted Probability>',predicted_probability)
    test_list.append({'<Line Number>':i+1,'<Target Label>':int(label.detach().item()),'<Predicted Probability>':predicted_probability})

## save
with open(test_result,'w') as file:
    file.write(json.dumps(train_list))