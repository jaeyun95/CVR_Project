## This is jaeyun's CVR Prediction Report.

### 0. Directory structure
```
CVR_Prediction
└code
    └model
        └model.py
        └__init__.py
    └util
        └data_preprocessing.py
        └dataset.py
        └__init__.py
    └config.py
    └main.py
    └requirements.txt
└data
    └CriteoSearchData  #input data here 
└save
README.md
```

### 1. DataSet
[[Download here]](https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/)

### 2. Setting up environment
```
conda create -n test python=3.6
source activate test
pip install -r requirements.txt
```

### 3. Variable setting

    - config.py 
```
      configure = {
      'DATA_PATH' : "$PATH/data/", ## data path
      'DATA_FILE' : "CriteoSearchData", ## data file
      'TRAIN_DATA' : [0,1000000], ## split train data range
      'TEST_DATA' : [1000000,1500000], ## split test data range
      'MODEL_SAVE_PATH' : "$PATH/save/", ## save path
      'MODEL_SAVE_FILE' : "models.dat", ## save model name
      'MODEL_LOAD_PATH' : "$PATH/save/", ## load path
      'MODEL_LOAD_FILE' : "models.dat", ## load model name
      'BATCH' : 100000, ## batch size
      'EPOCH' : 200, ## epoch
      'CUDA' : True, ## gpu use or not
      'SAVE' : True, ## model save or not
      'LOAD' : False, ## model load or not
      'TRAIN' : True ## train or not
}
```

### 4. Model training
```
# Change "LOAD" to False and "TRAIN" to True in the /CVR_Prediction/code/config.py file.
# location is /CVR_Prediction/code/
python main.py
```


### 5. Model evaluating
```
# Change "LOAD" to True and "TRAIN" to False in the /CVR_Prediction/code/config.py file.
# location is /CVR_Prediction/code/
python main.py
```

### 6. Result

#### (1) Evaluation index
I evaluated the model using accuracy. 
Accuracy = (True Positives + True Negatives) / (True Positivies + True Negatives + False Positives + False Negatives)
- True Positives : The model predicted to be 1, and label is 1.
- True Negatives : The model predicted to be 1, and label is 0.
- False Positivies : The model predicted to be 0, and label is 0.
- Flase Negatives : The model predicted to be 0, and label is 1.

#### (2) Accuracy

|Train Data|Test Data|
|:------:|:---:|
|81.67%|82.09%|


#### (3) Loss
|Train Data|Test Data|
|:------:|:---:|
|0.4647|0.4580|


