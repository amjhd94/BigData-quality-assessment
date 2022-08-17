from likelihood import likelihood

class acquisition_fcn():
    """
    
    Parameters
    ----------
    acquisition : Type of acquisition function for domain exploration.
    
        "us": Uncertainty Sampling 
        (acquisition function for discovering data 
         to reduce the prediction uncertainty of the 
         final trained model)
        
        "us_lw" Likelihood-weighted uncertainty sampling 
        (acquisition function for discovering data and 
         sub-domains corresponding to extreme and rare outputs)
        
        "us_lgw" Likelihood/geometrically-weighted uncertainty sampling 
        (acquisition function for discovering 
         data and sub-domains corresponding to 
         tipping points and topological extrema 
         of the output).
    
    ens_model : The NN ensemble model.
    
    inputs : Input object containing the features of the exploration domain - refer to inputs.py
    
    pts : points at which the values of the acquisition functions are to be computed.
    
    """
    
    def __init__(self, acquisition, ens_model, inputs=None, pts=None):
        self.acquisition = acquisition
        self.model = ens_model
        self.likelihood = likelihood
        self.inputs = inputs
        self.pts = pts
        if self.pts.all()==None:
            self.x = inputs.draw_samples(n_samples=int(1e3), sample_method="lhs") # sample_method = "uni" or "lhs"
        else:
            self.x = pts
        
    def us(self):
        acq = self.likelihood(self.model, inputs=self.inputs, pts=self.pts, weight_type='nominal')._evaluate_raw(self.x).flatten()*self.model._predict_mean_var(self.x)[1].flatten()
        return acq
        
    def us_lw(self):
        acq = self.likelihood(self.model, inputs=self.inputs, pts=self.pts, weight_type='importance')._evaluate_raw(self.x).flatten()*self.model._predict_mean_var(self.x)[1].flatten()
        return acq
    
    def us_lgw(self):
        acq = self.likelihood(self.model, inputs=self.inputs, pts=self.pts, weight_type='importance_ho')._evaluate_raw(self.x).flatten()*self.model._predict_mean_var(self.x)[1].flatten()
        return acq
    
    def eval_acq(self):
        if self.acquisition == "us":
            self.acq_val = self.us()
            
        elif self.acquisition == "us_lw":
            self.acq_val = self.us_lw()
            
        elif self.acquisition == "us_lgw":
            self.acq_val = self.us_lgw()
            
        else:
            print('Acquisition function not available!')
        
        return self.acq_val
    
    @staticmethod
    def check_acquisition(acquisition):
        assert(acquisition.lower() in ["us", "us_lw", "us_lgw"])
        return acquisition.lower()
