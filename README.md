# Big Data Quality Assessment
In the era of abundance data, and high velocity and variety information assets a data analyst's job has become more difficult to examine the qulity of such datasets, identify their redundancies, figure out where the datasets lack vital information and how to break them down into managable sub-dataset. Hence, the objective in this project is to employ Bayesian statistics and create parallel exploration algorithms that identify the most informative subsets of large datasets and eliminate their redundancies. The new data subset is then used to train a deep neural network model which is able to capture the rare and extreme events, has minimum prediction uncertainty and mean squared error (MSE) compared to the original dataset.

## Getting Started
The codes was written, run and tested by Spyder IDE version 4.2.5.
The following is the required packages for this project:
```bash
pip install numpy==1.21.2
pip install scipy==1.4.1
pip install tensorflow==2.9.0
pip install keras==2.9.0
pip install gpy==1.10.0
pip install pydoe==0.3.8
pip install kdepy==1.1.0
pip install matplotlib==3.4.3
pip install seaborn==0.11.2
```
With the packages installed, the code is ready to run.

## Tutorial
In the demo code `BigDataQualityAssessment_ActiveSampling.py` we consider a scenario in which we are given a dataset of size 10000 and are asked to create a regression model, using a fully connected deep neural network, that takes a 2-dimensional input, **X**, and predicts the scalar output, $y$. The figure below depicts the full dataset, {**X**, $y$}, and the posterior, $p(y)$, in log scale. 
<img src="https://user-images.githubusercontent.com/110791799/185191503-8be0ed0c-3a7b-4984-8a65-f73d75f742f4.png" alt="orig_ds" width="500"/>

However, the question is "Do we need all 10000 training data to create an accurate model?". It is far more preferable to use as little data as possible to train such model because it **speeds up** the training process and we can **decrease memory usage** by discarding unimportant data. Therefore, we employ three exploratory algorithms that identify data subsets that 

i) Most significantly decrease **squared error** metric between the model prediction and the original dataset;

ii) Significantly decrease the prediction uncertainty at the discarded data locations;

iii) Minimizes the difference between the model posterior and the original dataset output density functions in **log-scale**, thus ensuring that the rare and extreme events in the original dataset are accounted for in the new data subset and the model is able to predict them.


1- We first begin by importing the required modules:
```py
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
```

2- Next, we import (actually create, since this is a demo problem) the "large" dataset that was shown in the figure above:
```py
ndim = 2
domain = [ [-1, 1] ] * ndim
inputs = UniformInputs(domain)
pts = inputs.draw_samples(n_samples=100, sample_method="grd")
y = pts[:,0]**3 - pts[:,0] + pts[:,1]**2 + .5*np.sin(8*pts[:,0]*pts[:,1])
```

3- In this step, we select an initial small subset of the original dataset. The small subset can be randomly sampled from the original dataset or with any other arbitrary weight. I chose the weight to be $1/p(y)$ so that I'm sure that I have included at least some of the rare events in the initial data subset. This is not necessary since the algorithms eventually find the rare events. After the initial data subset is sampled, we create the initial training dataset with it.

```py
Size = 10
p = np.interp(y, y_pdf_x, y_pdf)
p = 1/p
p = p**1
p = p/np.sum(p)
y_resampled = np.random.choice(y.flatten(), size=Size, p=p)

x_train = []
y_train = []
for i in range(len(y_resampled)):
    y_train.append(y_resampled[i])
    x_train.append(pts[y_resampled[i] == y][int(len(pts[y_resampled[i] == y])/2)])

y_train = np.array(y_train)
x_train = np.array(x_train)
```

4- Now, we create and train an ensemble of fully connected deep neural networks on the small training dataset from the previous step.
```py
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
```

5- At this step, we are ready to run the exploration algorithms over the original dataset and add 3\*n_iter new data points to our previous initial data subset. Note that instead of prescribing the size of the data subset, we can simply add data until certain convergence criteria (that are defined by the user) are met.
```py
for i in range(n_iter):

    # ============== Squared Error explorer ============== #
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
    
    # ============== Prediction uncertainty explorer ============== #
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
    
    # ============== Rare event explorer ============== #
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
        ens_model.m_list[j].fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
```

At the end of this step, the dataset `[x_train, y_train]`, which is 2% as large as the original dataset represents the most important subset of the original dataset. A regression neural network trained with this data subset is as acuurate as a model trained with the original dataset.

The figure below shows the progression of the model trained on the small data subset and compares it with the original dataset as the explorers identify informative points.
<img src="https://user-images.githubusercontent.com/110791799/185217303-36731f05-89ed-44ac-84fe-6de8a647b037.gif" alt="model_prog" width="500"/>

(_Blue stars_ represent the data points in the initially sampled training dataset, _white dots_ are the data chosen by the **squared error** explorer, _white crosses_ are the data chosen by the **prediction uncertainty** explorer, and _white tri-ups_ are the data chosen by the **rare event** explorer.)

<img src="https://user-images.githubusercontent.com/110791799/185217610-b11261cb-b270-4bc7-8aa9-ab45219bb7aa.png" alt="conv" width="500"/>

(In the figure above, the log pdf error metric is computed by <img src="https://user-images.githubusercontent.com/110791799/185176409-7e8d3751-1027-41ae-8618-86f96f408c23.png" alt="equ1" width="300"/>)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
