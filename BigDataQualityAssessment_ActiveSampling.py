import sys
sys.path.append('core/')
import tensorflow as tf
from tensorflow import keras
from gpsearch import UniformInputs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
from core.utils import custom_KDE
from core.acquisition_fcn import acquisition_fcn
from core.ensemble_model import UQ_NN
import time
sns.set()
tf.keras.backend.set_floatx('float64')
   
#%% Setting up the initial training points

ndim = 2

domain = [ [-1, 1] ] * ndim

inputs = UniformInputs(domain)

pts = inputs.draw_samples(n_samples=100, sample_method="grd")

y = pts[:,0]**3 - pts[:,0] + pts[:,1]**2 + .5*np.sin(8*pts[:,0]*pts[:,1])

X1 = pts[:,0].reshape((100, 100))
X2 = pts[:,1].reshape((100, 100))
Y = y.reshape((100, 100))

bw = .05
y_pdf_obj = custom_KDE(y, bw=bw)
y_pdf_x, y_pdf = y_pdf_obj.evaluate()

plt.figure()
plt.subplot(1,2,1)
plt.contourf(X1, X2, Y)
plt.title('Original dataset')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()
plt.subplot(1,2,2)
plt.semilogy(y_pdf_x, y_pdf)
plt.xlabel('y')
plt.ylabel('p(y)')

#%%  Selecting the initial training dataset
Size = 100
p = np.interp(y, y_pdf_x, y_pdf)
p = 1/p
p = p**1
p = p/np.sum(p)

y_resampled = np.random.choice(y.flatten(), size=Size, p=p)

bins = np.linspace(np.min(y), np.max(y), 10)
plt.figure()
plt.hist(y, bins, density=True, alpha=0.5, label='original data')
plt.hist(y_resampled, bins, density=True, alpha=0.5, label='resampled data')
plt.title('Density function')
plt.legend()

x_train = []
y_train = []
for i in range(len(y_resampled)):
    y_train.append(y_resampled[i])
    x_train.append(pts[y_resampled[i] == y][int(len(pts[y_resampled[i] == y])/2)])

y_train = np.array(y_train)
x_train = np.array(x_train)

y_train_pdf_obj = custom_KDE(y_train, bw=bw)
y_train_pdf_x, y_train_pdf = y_train_pdf_obj.evaluate()

init_x_train = x_train

plt.figure()
plt.subplot(1,2,1)
triang = tri.Triangulation(x_train[:,0].flatten(), x_train[:,1].flatten())
interpolator = tri.LinearTriInterpolator(triang, y_train.flatten())
plt.tricontourf(x_train[:,0].flatten(), x_train[:,1].flatten(), y_train.flatten(), levels=30)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.colorbar()
plt.scatter(x_train[:,0], x_train[:,1], c='w')
plt.subplot(1,2,2)
plt.semilogy(y_train_pdf_x, y_train_pdf)
plt.xlabel('y')
plt.ylabel('p(y)')

#%% Train the first model

class MyModel(keras.Model):
    def model(self, activation='swish', ker_initializer=None):
        inputs = keras.Input(shape=(2,))
        x = keras.layers.Dense(8, activation=activation, kernel_initializer=ker_initializer)(inputs)
        x = keras.layers.Dense(8, activation=activation, kernel_initializer=ker_initializer)(x)
        x = keras.layers.Dense(8, activation=activation, kernel_initializer=ker_initializer)(x)
        outputs = keras.layers.Dense(1)(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model

batch_size = 1
Train_DS = [x_train, y_train]

ens_model = UQ_NN(model_class=MyModel, train=Train_DS, lr=.001, epochs=300, batch_size=batch_size)


y_test_pred, y_test_var = ens_model._predict_mean_var(pts)
squared_error_field = (y_test_pred.reshape(y.shape) - y)**2

y_test_pred_pdf_obj = custom_KDE(y_test_pred, bw=bw)
y_test_pred_pdf_x, y_test_pred_pdf = y_test_pred_pdf_obj.evaluate()

plt.figure()
plt.subplot(2,2,1)
plt.contourf(X1, X2, Y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Original full dataset')
plt.colorbar()
plt.subplot(2,2,2)
plt.contourf(X1, X2, y_test_pred.reshape(Y.shape))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Model prediction')
plt.colorbar()
plt.subplot(2,2,3)
plt.semilogy(y_pdf_x, y_pdf, label='Original full dataset')
plt.semilogy(y_test_pred_pdf_x, y_test_pred_pdf, label='Model prediction')
plt.xlabel('y')
plt.ylabel('p(y)')
plt.ylim([.01, 10])
plt.legend()
plt.subplot(2,2,4)
plt.contourf(X1, X2, squared_error_field.reshape(Y.shape))
plt.title('SE field')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()

#%% Active sampling of the informative data

lr1 = .00025

for model_no in range(len(ens_model.m_list)):
    ens_model.m_list[model_no].compile(optimizer=keras.optimizers.Adam(lr=lr1),
                                       loss=keras.losses.MeanSquaredError())


n_iter = 20
epochs = 2000
MSE = []
smp_no = []
log_pdf_error = []
log_pdfs = []
x_evals = []
model_pred_field = []
MSE_field = []
var_field = []
uslw_field = []
MSE_trained_field = []
var_trained_field = []
uslw_trained_field = []
MVar = []

possible_acq_list = ['se', 'us ', 'us_lw']
acq_list = possible_acq_list[:]

pts_temp = pts
y_temp = y

se_xdata = []
se_ydata = []
us_xdata = []
us_ydata = []
uslw_xdata = []
uslw_ydata = []

plt.figure(dpi=100)
for i in range(n_iter):
    
    t0 = time.time()
    y_test_pred_total, y_test_var_total = ens_model._predict_mean_var(pts)
    squared_error_field_total = (y_test_pred_total.reshape(y.shape) - y)**2
    MSE.append(np.mean(squared_error_field_total))
    MVar.append(np.mean(y_test_var_total))
    smp_no.append(len(x_train))
    model_pred_field.append(y_test_pred_total)
    MSE_field.append(squared_error_field_total)
    var_field.append(y_test_var_total)
    uslw_field.append(acquisition_fcn(acquisition='us_lw', ens_model=ens_model, pts=pts).eval_acq())
    MSE_trained_field.append((ens_model._predict_mean_var(x_train)[0].reshape(y_train.shape) - y_train)**2)
    var_trained_field.append(ens_model._predict_mean_var(x_train)[1])
    uslw_trained_field.append(acquisition_fcn(acquisition='us_lw', ens_model=ens_model, pts=x_train).eval_acq())
        
    y_test_pred_pdf_obj = custom_KDE(y_test_pred_total, bw=bw)
    
    x_min = min( y_test_pred_pdf_obj.data.min(), y_pdf_obj.data.min() )
    x_max = max( y_test_pred_pdf_obj.data.max(), y_pdf_obj.data.max() )
    rang = x_max-x_min
    x_eva = np.linspace(x_min - 0.01*rang,
                        x_max + 0.01*rang, 1024)
    
    yb, yt = y_test_pred_pdf_obj.evaluate(x_eva), y_pdf_obj.evaluate(x_eva)
    
    x_evals.append(x_eva)
    log_pdfs.append(yb)
    
    log_yb, log_yt = np.log(yb), np.log(yt)
    np.clip(log_yb, -6, None, out=log_yb)
    np.clip(log_yt, -6, None, out=log_yt)
        
    log_diff = np.abs(log_yb-log_yt)
    noInf = np.isfinite(log_diff)
    
    log_pdf_error.append(np.trapz(log_diff[noInf], x_eva[noInf]))
    
    
    if 'se' in acq_list:
        y_test_pred, _ = ens_model._predict_mean_var(pts_temp)
        squared_error_field = (y_test_pred.reshape(y_temp.shape) - y_temp)**2
        
        max_arg = np.argmax(squared_error_field)
        y_train_add = y_temp[max_arg]
        x_train_add = pts_temp[max_arg]
        
        se_xdata.append(x_train_add)
        se_ydata.append(y_train_add)
        
        y_train = np.hstack((y_train, y_train_add))
        x_train = np.vstack((x_train, x_train_add))
        
        y_temp = np.delete(y_temp, max_arg)
        pts_temp = np.delete(pts_temp, max_arg, axis=0)
    
    if 'us ' in acq_list:
        acq_fcn = acquisition_fcn(acquisition='us', ens_model=ens_model, pts=pts_temp).eval_acq()
        
        max_arg = np.argmax(acq_fcn)
        y_train_add = y_temp[max_arg]
        x_train_add = pts_temp[max_arg]
        
        us_xdata.append(x_train_add)
        us_ydata.append(y_train_add)
        
        y_train = np.hstack((y_train, y_train_add))
        x_train = np.vstack((x_train, x_train_add))
        
        y_temp = np.delete(y_temp, max_arg)
        pts_temp = np.delete(pts_temp, max_arg, axis=0)
    
    if 'us_lw' in acq_list:
        acq_fcn = acquisition_fcn(acquisition='us_lw', ens_model=ens_model, pts=pts_temp).eval_acq()
        
        max_arg = np.argmax(acq_fcn)
        y_train_add = y_temp[max_arg]
        x_train_add = pts_temp[max_arg]
        
        uslw_xdata.append(x_train_add)
        uslw_ydata.append(y_train_add)
        
        y_train = np.hstack((y_train, y_train_add))
        x_train = np.vstack((x_train, x_train_add))
        
        y_temp = np.delete(y_temp, max_arg)
        pts_temp = np.delete(pts_temp, max_arg, axis=0)
    
    batch_size = int(len(y_train))
    for j in range(len(ens_model.m_list)):
        history = ens_model.m_list[j].fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    plt.semilogy(i*epochs + np.array(range(epochs))+1, history.history['loss'], label='iter '+str(i+1))
    
    t1 = time.time()
    print('Iteration', i+1, 'completed in', int(t1-t0), 'seconds.')
  
MSE = np.array(MSE)
smp_no = np.array(smp_no)
log_pdf_error = np.array(log_pdf_error)

log_pdfs = np.array(log_pdfs)
x_evals = np.array(x_evals)
model_pred_field = np.array(model_pred_field)
MSE_field = np.array(MSE_field)
var_field = np.array(var_field)
uslw_field = np.array(uslw_field)
MSE_trained_field = np.array(MSE_trained_field)
var_trained_field = np.array(var_trained_field)
uslw_trained_field = np.array(uslw_trained_field)
MVar = np.array(MVar)

se_xdata = np.array(se_xdata)
se_ydata = np.array(se_ydata)
us_xdata = np.array(us_xdata)
us_ydata = np.array(us_ydata)
uslw_xdata = np.array(uslw_xdata)
uslw_ydata = np.array(uslw_ydata)

# =================== Some Graphs =================== #
plt.figure()
plt.subplot(1,3,1)
plt.semilogy(smp_no/len(y)*100, MSE)
plt.xlabel('Data pct.')
plt.ylabel('MSE for acq. fcn. :'+str(acq_list))
plt.subplot(1,3,2)
plt.semilogy(smp_no/len(y)*100, MVar)
plt.xlabel('Data pct.')
plt.ylabel('Mean variance for acq. fcn. :'+str(acq_list))
plt.subplot(1,3,3)
plt.semilogy(smp_no/len(y)*100, log_pdf_error)
plt.xlabel('Data pct.')
plt.ylabel('log pdf error for acq. fcn. :'+str(acq_list))

y_test_pred, y_test_var = ens_model._predict_mean_var(pts)
squared_error_field = (y_test_pred.reshape(y.shape) - y)**2

y_test_pred_pdf_obj = custom_KDE(y_test_pred, bw=bw)
y_test_pred_pdf_x, y_test_pred_pdf = y_test_pred_pdf_obj.evaluate()

plt.figure()
plt.subplot(3,2,1)
plt.contourf(X1, X2, Y)
plt.plot(init_x_train[:,0], init_x_train[:,1], '*c')
plt.plot(se_xdata[:,0], se_xdata[:,1], '.w')
plt.plot(us_xdata[:,0], us_xdata[:,1], 'xw')
plt.plot(uslw_xdata[:,0], uslw_xdata[:,1], '2w')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Original full dataset')
plt.colorbar()
plt.subplot(3,2,2)
plt.contourf(X1, X2, y_test_pred.reshape(Y.shape))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Model prediction')
plt.colorbar()
plt.subplot(3,2,(3, 4))
plt.semilogy(y_pdf_x, y_pdf, label='Original full dataset')
plt.semilogy(y_test_pred_pdf_x, y_test_pred_pdf, label='Model prediction')
plt.ylim([.01, 10])
plt.legend()
plt.subplot(3,2,5)
plt.contourf(X1, X2, squared_error_field.reshape(Y.shape))
plt.title('SE field')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()
plt.subplot(3,2,6)
plt.contourf(X1, X2, y_test_var.reshape(Y.shape))
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()
plt.title('var. field')