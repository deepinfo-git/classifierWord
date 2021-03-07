import os
import re
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from transformers import *

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd
import matplotlib.pyplot as plt


# 시각화

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

#random seed 고정
tf.random.set_seed(1234)
np.random.seed(1234)

BATCH_SIZE = 32
NUM_EPOCHS = 3
VALID_SPLIT = 0.2
MAX_LEN = 39 # EDA에서 추출된 Max Length
# DATA_IN_PATH = 'data_in/KOR'
# DATA_OUT_PATH = "data_out/KOR"
DATA_IN_PATH = "20200501~20200531"
# DATA_IN_PATH = "20190501~20190531"
DATA_IN_PATH = "20190501~20190531"

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir='bert_ckpt', do_lower_case=False)

# 데이터 전처리 준비
DATA_TRAIN_PATH = os.path.join(DATA_IN_PATH, "[사회]articles_UTF-8.txt")
# DATA_TEST_PATH = os.path.join(DATA_IN_PATH, "ratings_test.txt")

# train_data = pd.read_csv(DATA_TRAIN_PATH, header = 0, delimiter = '\t', quoting = 3)
train_data = pd.read_csv(DATA_TRAIN_PATH, delimiter = '\t', header=None, dtype="string")
train_data = train_data.dropna()
# print(train_data.head())
#print("train_data", type(train_data))
s = train_data

train_data_np = pd.DataFrame(train_data).to_numpy()
train_data_np = train_data_np
# print(train_data_np[0])
#따옴표 제거
def dequote(s):
    """
    If a string has single or double quotes around it, remove them.
    Make sure the pair of quotes match.
    If a matching pair of quotes is not found, return the string unchanged.
    """
    if (s[0] == s[-1]) and s.startswith(("'", '"')):
        return s[1:-1]
    return s

def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

def ret(train_data_np):
        ds3 = dequote(to_str(train_data_np[i]))
        ds4 = ds3.replace("\\n", '').replace('\\', '')
        return ds4

import kss
from kss import split_chunks
from kss import split_sentences
list1 = []

import csv

for i in range(0, len(train_data_np)):
    #print(ret(train_data_np))
    #for sent in kss.split_sentences(ret(train_data_np)):
    #for sent in split_chunks(ret(train_data_np), max_length=128):
    for sent in split_sentences(ret(train_data_np), safe=True, max_recover_length=20):
        print(sent)
        list1.append([sent])


print("sent: ", list1[:][0])
# csv 모듈 import하기

import csv

csvfile = open('[사회]articles_UTF-8_ex.txt', 'w', newline="")

csvwirter = csv.writer(csvfile)
for row in list1:
    csvwirter.writerow(row)

'''
with open('GFG.txt', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    for i in range(0, len(list1)):
        write.writerow(list1[:][i])
'''
'''
import kss
#print(dequote(to_str(train_data_np[0])))
#ds3 = dequote(to_str(train_data_np))
#ds4 = ds3.replace("\\n", '').replace('\\', '')
#print(ds4)

result = []
print(len(train_data_np))
#for i in range(0, len(to_str(train_data_np))):
for i in range(0, len(to_str(train_data_np))):

    # for sent in kss.split_sentences(dequote(to_str(train_data_np[i])).replace("\\n", '').replace('\\', '')):
    for sent in kss.split_sentences(dequote(to_str(train_data_np[i]))):
        print(sent)
        result.append(sent)

print(result)
'''