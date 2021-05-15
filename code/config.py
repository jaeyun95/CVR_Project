
configure = {
      'DATA_PATH' : "$PATH/data/", ## data path
      'DATA_FILE' : "CriteoSearchData", ## data file
      'TRAIN_DATA' : [0,5000000], ## split train data range
      'TEST_DATA' : [5000000,6500000], ## split test data range
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