# https://keras.io/guides/writing_a_training_loop_from_scratch/
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import seaborn as sns
from utils import custom_KDE
import time
sns.set()
tf.keras.backend.set_floatx('float64')
np.random.seed(10)

#%% =========================================== %%#
##### Stochastic signal (SDE) #####

np.random.seed(10)

sigma = 2.  # Standard deviation.
mu = 2.  # Mean.
tau = .5  # Time constant.

dt = .001  # Time step.
T = 1.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.

sigma_bis = sigma * np.sqrt(2. / tau)
sqrtdt = np.sqrt(dt)

x = np.zeros(n)

for i in range(n - 1):
    x[i + 1] = x[i] + dt * (-(.25*x[i] - mu) / tau) + sigma_bis * sqrtdt * np.random.randn()

y = np.atleast_2d(x).T
x = t

plt.figure()
plt.plot(x, y)
plt.xlabel('time')
plt.ylabel('y')

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(y)
training_set_scaled.reshape(-1, 1)
    
training_set_scaled = training_set_scaled.flatten().reshape(-1, 1)
#%%

history_window = 10
horizon_window = 0
prediction_window = 5
stride_window = 1

x_train = []
y_train = []
for i in range(0, len(training_set_scaled) - history_window - prediction_window - horizon_window + 1, stride_window):
    x_train.append(training_set_scaled[i:(i+history_window)])
    y_train.append(training_set_scaled[(i+(history_window+horizon_window)):(i+history_window+prediction_window+horizon_window)])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

y_train = np.reshape(y_train, (len(x_train), prediction_window))

true_pdf = custom_KDE(y_train, bw=.01)
true_pdf_y, true_pdf_py = true_pdf.evaluate()
pdf_x = y_train
pdf_y = np.interp(pdf_x, true_pdf_y, true_pdf_py)
pdf_y = pdf_y*(pdf_y>=0)

pdf_y_train = np.reshape(pdf_y, (len(y_train), prediction_window))

plt.figure()
plt.subplot(1,2,1)
plt.semilogy(true_pdf_y, true_pdf_py, 'k')
plt.subplot(1,2,2)
plt.plot(y_train)
plt.plot(1/pdf_y)

#%% Preprocess training data by projection on to POD basis

X_train_preproc = x_train[:,:,0]
Y_train_preproc = y_train
Full_train_Xy = np.hstack((X_train_preproc, Y_train_preproc))

u, s, v = scipy.linalg.svd(Full_train_Xy.T, full_matrices=(False), lapack_driver='gesvd')

Full_train_Xy_coeff_mat = (np.diag(s) @ v).T
Full_train_Xy_coeff_mat_scaled1 = (np.diag(s*(np.array(range(len(s)))+1)) @ v).T
Full_train_Xy_coeff_mat_scaled2 = (np.diag(s*(np.array(range(len(s)))+1)**2) @ v).T
# # Full_train_Xy_coeff_mat_scaled3 = (np.diag(s*(np.array(range(len(s)))+1)**3) @ v).T

#%% POD coefficient distributions

# mode_no_hist = 2
# bins = 20
# plt.figure()
# plt.hist(Full_train_Xy_coeff_mat[:,mode_no_hist], bins=bins)

#%% Create initial informative dataset

Size = 100
resample_mode_no = 5
X_train_res = []
Y_train_res = []
for resample_mode in range(resample_mode_no):
    coeff_pdf = custom_KDE(Full_train_Xy_coeff_mat[:, resample_mode], bw=.05)
    coeff_pdf_x, coeff_pdf_y = coeff_pdf.evaluate()
    p = np.interp(Full_train_Xy_coeff_mat[:, resample_mode].flatten(), coeff_pdf_x, coeff_pdf_y)
    p = 1/p
    p = p/np.sum(p)
    
    y_resampled = np.random.choice(Full_train_Xy_coeff_mat[:, resample_mode].flatten(), size=Size, p=p)
    x_train_res = []
    y_train_res = []
    for i in range(len(y_resampled)):
        y_train_res.append(y_train[y_resampled[i] == Full_train_Xy_coeff_mat[:, resample_mode], :])
        x_train_res.append(x_train[y_resampled[i] == Full_train_Xy_coeff_mat[:, resample_mode], :, 0])
        
    X_train_res.append(x_train_res)
    Y_train_res.append(y_train_res)
    
X_train_res = np.array(X_train_res).reshape((-1, x_train.shape[1], x_train.shape[2]))
Y_train_res = np.array(Y_train_res).reshape((-1, y_train.shape[1]))

X_train_res = np.unique(X_train_res, axis=0)
Y_train_res = np.unique(Y_train_res, axis=0)

true_pdf = custom_KDE(Y_train_res, bw=.05)
true_pdf_y, true_pdf_py = true_pdf.evaluate()
pdf_x = Y_train_res
pdf_y = np.interp(pdf_x, true_pdf_y, true_pdf_py)
pdf_y = pdf_y*(pdf_y>=0)

pdf_y_train = np.reshape(pdf_y, (len(Y_train_res), prediction_window))


rand_idx = np.random.permutation(X_train_res.shape[0])
X_train = X_train_res[rand_idx,:]
Y_train = Y_train_res[rand_idx,:]
pdf_Y_train = pdf_y_train[rand_idx,:]

#%% Training the intial model

lr1 = 1e-2

inputs = keras.Input(shape=(X_train.shape[1],1))
x = layers.Dense(4)(inputs)
x = layers.Dense(8)(x)
x = layers.Dense(16)(x)
x = layers.LSTM(units=32)(x)
x = layers.Dense(16)(x)
x = layers.Dense(16)(x)
outputs = layers.Dense(units=prediction_window)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    
    optimizer=keras.optimizers.Adam(learning_rate=lr1),
    loss=keras.losses.MeanSquaredError(),
    metrics=['mse']
    
    )


epochs = 100
train_acc_rec = []
T0 = time.time()
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=16, verbose=2)
print('Training took', np.round_((time.time() - T0)/60), 'minutes!')

#%% Training data accuracy of the initial model

y_test = model.predict(x_train)

plt.figure()
plt.subplot(2,2,1)
plt.plot(np.mean(y_train, axis=1).flatten(), '-', label='True')
plt.plot(np.mean(y_test, axis=1), '--r', label='Model')
plt.legend()

plt.subplot(2,2,3)
plt.plot(np.abs(np.mean(y_train, axis=1).flatten() - np.mean(y_test, axis=1).flatten()))
plt.title('abs error')

pred_pdf = custom_KDE(model.predict(x_train), bw=.01)
pred_pdf_y, pred_pdf_py = pred_pdf.evaluate()

true_pdf = custom_KDE(y_train, bw=.01)
true_pdf_y, true_pdf_py = true_pdf.evaluate()

plt.subplot(2,2,2)
plt.semilogy(true_pdf_y, true_pdf_py, 'k', label='True')
plt.semilogy(pred_pdf_y, pred_pdf_py, 'r', label='Model')
plt.legend()

#%% Sequential training and dataset augmentation

lr2 = 1e-2
x_train_temp = x_train
y_train_temp = y_train
epochs = 100
iterations = 20
new_smps = 20
total_MAE_iter = []
training_smp_no = []
y_test = model.predict(x_train_temp)
total_MAE_iter.append(np.mean(np.abs(y_test - y_train)))
plt.figure()
for iters in range(iterations):
    training_smp_no.append(len(X_train))
    output_abs_err_smpNo = np.sum(np.abs(y_test - y_train_temp), axis = 1)
    
    y_err_idx_sorted = np.argsort(output_abs_err_smpNo)
    y_err_idx_sorted = y_err_idx_sorted[::-1]
    
    x_train_add = x_train_temp[y_err_idx_sorted[:new_smps], ::]
    x_train_temp = np.delete(x_train_temp, y_err_idx_sorted[:new_smps], axis=0)
    y_train_add = y_train_temp[y_err_idx_sorted[:new_smps], :]
    y_train_temp = np.delete(y_train_temp, y_err_idx_sorted[:new_smps], axis=0)
    
    X_train = np.vstack((X_train, x_train_add))
    Y_train = np.vstack((Y_train, y_train_add))
    
    T0 = time.time()
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=512, verbose=0)
    print('Iter', iters, 'took', np.round_((time.time() - T0)/60), 'minutes, with loss:', history.history['loss'][-1])
    plt.semilogy(iters*epochs + np.array(range(epochs))+1, history.history['loss'], label='iter '+str(iters+1))
    
    y_test = model.predict(x_train_temp)
    total_MAE_iter.append(np.mean(np.abs(y_test - y_train_temp)))

plt.legend()

plt.figure()
plt.plot(np.array(total_MAE_iter).flatten())
plt.ylabel('MAE')
plt.xlabel('iteration')

#%% Test model performance

y_test = model.predict(x_train)

plt.figure()
plt.subplot(2,2,1)
plt.plot(np.mean(y_train, axis=1).flatten(), '-', label='True')
plt.plot(np.mean(y_test, axis=1), '--r', label='Model')
plt.legend()

plt.subplot(2,2,3)
plt.plot(np.abs(np.mean(y_train, axis=1).flatten() - np.mean(y_test, axis=1).flatten()))
plt.title('abs error')

pred_pdf = custom_KDE(model.predict(x_train), bw=.01)
pred_pdf_y, pred_pdf_py = pred_pdf.evaluate()

true_pdf = custom_KDE(y_train, bw=.01)
true_pdf_y, true_pdf_py = true_pdf.evaluate()

plt.subplot(2,2,2)
plt.semilogy(true_pdf_y, true_pdf_py, 'k', label='True')
plt.semilogy(pred_pdf_y, pred_pdf_py, 'r', label='Model')
plt.legend()


