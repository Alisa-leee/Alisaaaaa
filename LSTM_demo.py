
import random
#
# Core Keras libraries
#
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras import regularizers
#
# For data conditioning
#
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
#
# Make results reproducible
#
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import glob
# 
# Other essential libraries
#
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import numpy as np

#
# Set input number of timestamps and training days
#
n_timestamp = 2
n_epochs = 25
filter_on = 1
model_type = 1
buffer = np.zeros([1, 2, 2])
buffer1 = np.zeros([1, 2])
buffer2 = np.zeros([1, 2, 2])
buffer3 = np.zeros([1, 2])

#url1 = "D:/predicted/ca/12-nu11.csv"
url1 = "D:/predicted/mo/1-1-nu17.csv"
dataset1 = pd.read_csv(url1)
'''
if filter_on == 1:
    dataset1['X'] = medfilt(dataset1['X'], 1)
    dataset1['X'] = gaussian_filter1d(dataset1['X'],0.4)

if filter_on == 1:
    dataset1['Y'] = medfilt(dataset1['Y'], 1)
    dataset1['Y'] = gaussian_filter1d(dataset1['Y'], 0.4)
'''
n = len(dataset1)
train_days1 = round(n*(2/3))  
testing_days1 =round(n*(1/3))  
#
# Set number of training and testing data
# 
data1 = np.array(dataset1)
train_set1 = data1[0:train_days1,:]
test_set1 = data1[train_days1: train_days1+testing_days1,:]
training_set1 = train_set1[0:]
testing_set1 = test_set1[0:]

#
# Normalize data first(歸一化)
#
'''
sc1 = MinMaxScaler(feature_range = (0, 1))
training_set_scaled1 = sc1.fit_transform(training_set1)
testing_set_scaled1 = sc1.fit_transform(testing_set1)
'''
#
# Split data into n_timestamp
#
def data_split(sequence1, n_timestamp):
    X1= []
    y1 = []
    for i in range(len(sequence1)):
        end_ix = i + n_timestamp
        if end_ix > len(sequence1)-1:
            break
        seq_x, seq_y = sequence1[i:end_ix], sequence1[end_ix]
        X1.append(seq_x)
        y1.append(seq_y)
    return array(X1), array(y1)

X_train1, y_train1 = data_split(training_set1, n_timestamp)
X_test1, y_test1 = data_split(testing_set1, n_timestamp) 

def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
dirname ='D:\predicted\mo1'
filenames=glob.glob(dirname+'\*.csv') 
for i in filenames:
    dataset = pd.read_csv(i)
    '''
    if filter_on == 1:
        dataset['X'] = medfilt(dataset['X'], 1)
        dataset['X'] = gaussian_filter1d(dataset['X'],0.4)
    
    if filter_on == 1:
        dataset['Y'] = medfilt(dataset['Y'], 1)
        dataset['Y'] = gaussian_filter1d(dataset['Y'], 0.4)
    '''
    n = len(dataset)
    train_days = round(n*(2/3))  
    testing_days =round(n*(1/3))  
#
# Set number of training and testing data
# 
    data = np.array(dataset)
    train_set = data[0:train_days,:]
    test_set = data[train_days: train_days+testing_days,:]
    training_set = train_set[0:]
    testing_set = test_set[0:]
    
    X_train, y_train = data_split(training_set, n_timestamp)
    X_test, y_test = data_split(testing_set, n_timestamp)       
    buffer=np.vstack((buffer,X_train))
    buffer1=np.vstack((buffer1,y_train))

#
# Normalize data first(歸一化)
#
    '''
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    testing_set_scaled = sc.fit_transform(testing_set)
    
    X_train, y_train = data_split(training_set_scaled, n_timestamp)
    X_test, y_test = data_split(testing_set_scaled, n_timestamp)       
    buffer=np.vstack((buffer,X_train))
    buffer1=np.vstack((buffer1,y_train))
    '''
#
# Split data into n_timestamp
#
if model_type == 1:
    # Stacked LSTM
    model = Sequential()
    model.add(BatchNormalization())
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(2, 2)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(2))
'''
seed=1
np.random.seed(seed)
np.random.shuffle(buffer) #X_train
np.random.seed(seed)
np.random.shuffle(buffer1) #y_train
'''
#
# Start training
#
#model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')

history = model.fit(buffer,  buffer1, epochs = n_epochs, batch_size = 32)
loss = history.history['loss']
epochs = range(len(loss)) 
z = np.c_[epochs,loss]
#
# Get predicted data
#
'''
y_predicted = model.predict(X_test1)
#
# 'De-normalize' the data
#
sc = MinMaxScaler(feature_range = (0, 1))
y_predicted_descaled = sc.inverse_transform(y_predicted) #預測值
y_test_descaled = sc.inverse_transform(y_test1)
#y_train_descaled = sc.inverse_transform(buffer1)
'''
#
# Get predicted data
#
y_predicted = model.predict(X_test1)
#正規化
sc = MinMaxScaler(feature_range = (0, 1))
aaa = sc.fit_transform(y_predicted)
bbb = sc.fit_transform(y_test1)

y_predicted_descaled = sc.inverse_transform(aaa) #預測值
y_test_descaled = sc.inverse_transform(bbb)


plt.figure(1)
plt.plot(y_predicted_descaled[:,0],y_predicted_descaled[:,1],color = 'red');
plt.plot(y_test_descaled[:,0],y_test_descaled[:,1],color = 'black');


plt.figure(2)
plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()
