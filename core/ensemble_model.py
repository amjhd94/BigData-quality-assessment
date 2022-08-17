import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

class UQ_NN():
    """

    Parameters
    ----------
    model_class : The individual NN model of the ensemble - refer to "base_model.py".
    
    train : Training dataset of form [X_train, y_train].
        
    val_pct : (optional: The default is 0) ratio of the "train" to be used as validation dataset.
    
    N : (optional: The default is 2) ensemble size.
    
    lr : (optional: The default is 0.001) Learning rate.
    
    epochs : (optional: The default is 3000) Training epochs at each iteration.
            
    batch_size : (optional: The default is 512) Batch size.
    
    fit_verbose : (optional: The default is 0) Each NN training log.
    
    ens_verbose : (optional: The default is True) Ensemble training log.

    """
    def __init__(self, model_class, train, val_pct=0, N=2, lr=0.001, epochs=3000, 
                 batch_size=512, fit_verbose=0, ens_verbose=True, model_prefix='Model'):
        self.model_class = model_class
        self.train = train
        self.val_pct = val_pct
        self.N = N
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.fit_verbose = fit_verbose
        self.ens_verbose = ens_verbose
        self.model_prefix = model_prefix
        
        
        self.m_list = []
        for i in range(self.N):
            t0 = time.time()

            Model = self.model_class().model()
            Model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                          loss=keras.losses.MeanSquaredError(),
                          metrics=['mse'])
            Model.fit(train[0], train[1], validation_split=val_pct, epochs=self.epochs, batch_size=self.batch_size, verbose=self.fit_verbose)

                
            self.m_list.append(Model)
            if self.ens_verbose == True:
                print(f'Model {i+1} is trained in', int(time.time()-t0), 'seconds!')
        
    def _predict_mean_var(self, x):
        Y = []
        for i in range(self.N):
            # Y.append(self.m_list[i].predict(x, batch_size=self.batch_size))
            Y.append(self.m_list[i](x)) # Doesn't give you the "@tf.function" call error
            
        Y = np.array(Y)
        mean = np.mean(Y, axis=0)
        var = np.var(Y, axis=0)
        return mean, var
    
    def _predictive_jac_hess(self, x, compute_hess=False):
        
        if compute_hess:
            if x.shape[0] <= 1000:
                if x.shape[1] == 1:
                    xq = tf.constant(x)
                else:
                    xq = tf.constant(list(x))
                
                
                Jac_list = []
                Hess_list = []
                for i in range(self.N):
                    with tf.GradientTape() as t2:
                        t2.watch(xq)
                        with tf.GradientTape() as t1:
                            t1.watch(xq)
                            yp = self.m_list[i](xq)
                    
                        jac = t1.gradient(yp, xq)
                        
                    hess = t2.jacobian(jac, xq)
                        
                    jac_vec = tf.reshape(jac, [xq.shape[0], xq.shape[1]])
                    hess_mat = tf.reduce_sum(hess, axis=2)
                    
                    Jac_list.append(jac_vec)
                    Hess_list.append(hess_mat)
                    
                mean_jac = np.mean(np.array(Jac_list), axis=0)
                mean_hess = np.mean(np.array(Hess_list), axis=0)
                    
            else:
                mean_jac = np.zeros((x.shape[0], x.shape[1]))
                mean_hess = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
                
                if x.shape[0]%1000 == 0:
                    batch_numbers = x.shape[0]//1000
                else: 
                    batch_numbers = x.shape[0]//1000 + 1
                    
                for batch_counter in range(batch_numbers):
                    if (batch_counter+1)*1000 < x.shape[0]:
                        x_temp = x[batch_counter*1000:(batch_counter+1)*1000,:]
                    else:
                        x_temp = x[batch_counter*1000:,:]
                    
                    if x_temp.shape[1] == 1:
                        xq = tf.constant(x_temp)
                    else:
                        xq = tf.constant(list(x_temp))
                    
                    if compute_hess:
                        Jac_list = []
                        Hess_list = []
                        for i in range(self.N):
                            with tf.GradientTape() as t2:
                                t2.watch(xq)
                                with tf.GradientTape() as t1:
                                    t1.watch(xq)
                                    yp = self.m_list[i](xq)
                            
                                jac = t1.gradient(yp, xq)
                                
                            hess = t2.jacobian(jac, xq)
                                
                            jac_vec = tf.reshape(jac, [xq.shape[0], xq.shape[1]])
                            hess_mat = tf.reduce_sum(hess, axis=2)
                            
                            Jac_list.append(jac_vec)
                            Hess_list.append(hess_mat)
                        
                            
                        if (batch_counter+1)*1000 < x.shape[0]:
                            mean_jac[batch_counter*1000:(batch_counter+1)*1000,:] = np.mean(np.array(Jac_list), axis=0)
                            mean_hess[batch_counter*1000:(batch_counter+1)*1000,:,:] = np.mean(np.array(Hess_list), axis=0)
                        else:
                            mean_jac[batch_counter*1000:,:] = np.mean(np.array(Jac_list), axis=0)
                            mean_hess[batch_counter*1000:,:,:] = np.mean(np.array(Hess_list), axis=0)
            
            return mean_jac, mean_hess
        
        else:
            
            if x.shape[0] <= 1000:
                if x.shape[1] == 1:
                    xq = tf.constant(x)
                else:
                    xq = tf.constant(list(x))
                
                Jac_list = []
                for i in range(self.N):
                    with tf.GradientTape() as t2:
                        t2.watch(xq)
                        with tf.GradientTape() as t1:
                            t1.watch(xq)
                            yp = self.m_list[i](xq)
                    
                        jac = t1.gradient(yp, xq)
                        
                    jac_vec = tf.reshape(jac, [xq.shape[0], xq.shape[1]])
                    
                    Jac_list.append(jac_vec)
                    
                mean_jac = np.mean(np.array(Jac_list), axis=0)
                    
            else:
                mean_jac = np.zeros((x.shape[0], x.shape[1]))
                
                if x.shape[0]%1000 == 0:
                    batch_numbers = x.shape[0]//1000
                else: 
                    batch_numbers = x.shape[0]//1000 + 1
                    
                for batch_counter in range(batch_numbers):
                    if (batch_counter+1)*1000 < x.shape[0]:
                        x_temp = x[batch_counter*1000:(batch_counter+1)*1000,:]
                    else:
                        x_temp = x[batch_counter*1000:,:]
                    
                    if x_temp.shape[1] == 1:
                        xq = tf.constant(x_temp)
                    else:
                        xq = tf.constant(list(x_temp))
                    
                    if compute_hess:
                        Jac_list = []
                        Hess_list = []
                        for i in range(self.N):
                            with tf.GradientTape() as t2:
                                t2.watch(xq)
                                with tf.GradientTape() as t1:
                                    t1.watch(xq)
                                    yp = self.m_list[i](xq)
                            
                                jac = t1.gradient(yp, xq)
                              
                            jac_vec = tf.reshape(jac, [xq.shape[0], xq.shape[1]])
                            
                            Jac_list.append(jac_vec)
                        
                            
                        if (batch_counter+1)*1000 < x.shape[0]:
                            mean_jac[batch_counter*1000:(batch_counter+1)*1000,:] = np.mean(np.array(Jac_list), axis=0)
                        else:
                            mean_jac[batch_counter*1000:,:] = np.mean(np.array(Jac_list), axis=0)
                    
            return mean_jac
