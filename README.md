# Big Data Quality Assessment
### Parallel exploration algorithms to identify the most informative subsets of large datasets and eliminate redundancy
Exploration is done to identfy the smallest subset of the data with which a deep neural network model could be trained such that it captures the rare extreme events and tipping points, and minimizes the overall uncertainty of the model predictions and mean squared error (MSE).

The code "BigDataQualityAssessment_ActiveSampling.py" contains a quick demo of how the algorithms work in a regression problem.

The code "SDE_forecast_ActiveSampling.py" contains a quick demo of how the algorithms work in forecasting a problem involving a stochastic signal.


# AI-based Active Learning
The agents take advantage of likelihood and output weighted acquisition functions to discover the most informative data to train neural network models. The resulting dataset includes frequent events, rare and extreme events, tipping points and topological extrema. 

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
The demo code `RareEvent_TippingPoint_discovery_demo.py` shows the process of discovering important training data to create a deep neural network that most accurately identifies and models [Hopf bifurcation](https://www.math.colostate.edu/~shipman/47/volume3b2011/M640_MunozAlicea.pdf) diagram. The bifurcation point, $\mu = 0$, corresponds to an extreme event, a tipping point and a topological extremum - Hopf bifurcation diagram, $r$ vs $\mu$ and its posterior (in log scale), $p(r)$ are depicted below. 
<img src="https://user-images.githubusercontent.com/110791799/185156764-5ebc179b-6612-4943-a1b2-fb3edb305bd9.png" alt="obj_fcn" width="500"/>

The AI agent explores the domain and learns the location(s) that corresponds to the extreme events in the output, i.e., the bifurcation point. 
Note: In this case, both the AI and the black-box model are deep neural networks.

1- We first begin by importing the required modules:
```py
import sys
sys.path.append('core/')
from active_sampling import active_sampling
from inputs import *
from core.utils import *
import numpy as np
```
2- Next, we define the objective function class. This class contains the "expensive" experiment that we cannot afford to run numerous times due to its high cost. In this demo, the `evaluate(x)` method in `obj_fcn()` class evaluates the value of output, $r$, corresponding to input `x`, by using the true bifurcation diagram formula. 
```py
class obj_fcn():
    def evaluate(self, x):
        y = 0*(x<0) + np.sqrt(np.abs(x))*(x>=0)
        return y
```

3- Next, we define the exploration domain, the input probability density function (in this case it is assumed input domain is sampled from uniformly) and a reference dataset (test data) to compare our model against:
```py
domain = [ [-1, 1] ]
inputs = UniformInputs(domain)
pts = inputs.draw_samples(n_samples=int(1e3), sample_method="grd")
x = pts
y = obj_fcn().evaluate(x)
```

4- In this step we initialize our exploration by creating an initial training dataset of size 2. This dataset is used by the AI to begin domain exploration.
```py
n_init = 2
x_init = np.random.uniform(low=domain[0][0], high=domain[0][1], size=(n_init,1))
y_init =  0*(x_init<0) + np.sqrt(np.abs(x_init))*(x_init>=0)
```

5- For this demo, we stop the AI after it has explored and found the best 30 training data points. In practice, exploration can continue until an accuracy metric meets a criterion.
```py
n_iter = 30
data_init = [x_init, y_init]

NN_active_sampling = active_sampling(data_init, obj_fcn, inputs=inputs, epochs=1500, batch_size=1)
ens_model_list = NN_active_sampling.optimize(acquisition='us_lw', n_iter=n_iter)
```
Note that `NN_active_sampling` intance is initialized with the initial training dataset, `data_init` and the objective function class, `obj_fcn`. (The rest of the parameters are defined in details in `active_sampling.py`). Next, we begin exploration by calling the `optimize` method with "likelihood weighted uncertainty sampling" criterion. Below we see how the agent discovers the corresponding input region to the extreme event and how accurately it replicates the objective function, i.e., the bifurcation diagram, using small size training dataset. 
<img src="https://user-images.githubusercontent.com/110791799/185178026-7fad2c7f-c25d-4a3a-a96d-c0d859bdd3c3.gif" alt="model_iter" width="500"/>

Below we see two convergence accuracy metrics:
  
  1- log pdf error which compares the log of the posterior predicted by our trained model and that of the actual objective function using Monte-Carlo sampling:

<img src="https://user-images.githubusercontent.com/110791799/185176409-7e8d3751-1027-41ae-8618-86f96f408c23.png" alt="equ1" width="300"/>
    
  2- coefficitne of determination, $R^2$, by comparing model prediction and the objective function.

<img src="https://user-images.githubusercontent.com/110791799/185178193-919b8fdb-9231-4f44-a231-4d5530d68815.png" alt="conv" width="500"/>

These graphs show that with even 12 sample we create an accurate black-box model of the objective function.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
