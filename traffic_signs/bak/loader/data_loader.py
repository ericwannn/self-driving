import pickle
import cv2
import numpy as np

def convert_to_grayscale(data):
  pdata = np.ndarray((data.shape[0], 32, 32, 1), dtype=np.float32)
  for i, img in enumerate(data):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    pdata[i] = np.expand_dim(img, axis=2)
  return pdata

def normalize(data, mean, stddev):
  return (data.astype(np.float32) - mean) / stddev

def data_loader(dir):
  train_files = dir[train]
  valid_files = dir[valid]
  test_files  = dir[test]

  with open(train_files, mode='rb') as f:
    train = pickle.load(f)

  with open(valid_files, mode='rb') as f:
    valid = pickle.load(f)

  with open(test_files, mode='rb') as f:
    test = pickle.load(f)


  X_train_raw, y_train = train['features'], train['labels']
  X_valid_raw, y_valid = valid['features'], valid['labels']
  X_test_raw,  y_test  = test['features'],  test['labels']

  _MEAN = np.mean(X_train_raw)  
  _STD  = np.std(X_train_raw)

  X_train = normalize(convert_to_grayscale(X_train_raw), _MEAN, _STD)
  X_valid = normalize(convert_to_grayscale(X_valid_raw), _MEAN, _STD)
  X_test  = normalize(convert_to_grayscale(X_test_raw), _MEAN, _STD)

  return X_train, y_train, X_valid, y_valid, X_test, y_test