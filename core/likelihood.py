import numpy as np
from core.utils import custom_KDE
from scipy.interpolate import InterpolatedUnivariateSpline

class likelihood():
    """
        
    Parameters
    ----------
    ens_model : The NN ensemble model.
    
    inputs : Input object containing the features of the exploration domain - refer to inputs.py
    
    weight_type : (optional: The default is "importance") acquisition weight type.
                    Can be one of "nominal", "importance" and "importance_ho" (High order importance)

    """
    def __init__(self, ens_model, inputs=None, weight_type="importance", 
                  c_w2=1, c_w3=1, tol=1e-5, pts=None, bw=.05):

        self.model = ens_model
        self.inputs = inputs
        self.weight_type = self.check_weight_type(weight_type)
        self.c_w2 = c_w2
        self.c_w3 = c_w3
        self.tol = tol
        self.pts = pts
        self.bw = bw

    def update_model(self, model):
        self.model = model
        self._prepare_likelihood()
        return self

    def evaluate(self, x=None):
        w = self._evaluate_raw(x)
        return w

    def _evaluate_raw(self, x=None):
        if self.pts.all()==None:
            fx = self.inputs.pdf(x)
        else:
            fx = np.ones((self.pts.shape[0],)).flatten()
        
        if self.weight_type == "nominal":
            w = fx
            
        elif self.weight_type == "importance":
            if self.pts.all()==None:
                mu = self.model._predict_mean_var(x)[0].flatten()
                x2, y = custom_KDE(mu, weights=fx, bw=self.bw).evaluate()
            else:
                mu = self.model._predict_mean_var(self.pts)[0].flatten()
                x2, y = custom_KDE(mu, bw=self.bw).evaluate()
                
            self.fy_interp = InterpolatedUnivariateSpline(x2, y, k=1)
            fy = self.fy_interp(mu) 
            w = fx/fy
            
            
        elif self.weight_type == "importance_ho":
            if self.pts==None:
                mu = self.model._predict_mean_var(x)[0].flatten()
                x2, y = custom_KDE(mu, weights=fx, bw=self.bw).evaluate()
                Jac_vecs, Hess_mat = self.model._predictive_jac_hess(x, compute_hess=True)
            else:
                mu = self.model._predict_mean_var(self.pts)[0].flatten()
                x2, y = custom_KDE(mu, bw=self.bw).evaluate()
                Jac_vecs, Hess_mat = self.model._predictive_jac_hess(self.pts, compute_hess=True)
            
            self.fy_interp = InterpolatedUnivariateSpline(x2, y, k=1)
            fy = self.fy_interp(mu) 
            fy_jac = self.fy_interp.derivative()(mu)
            term_temp = np.array([np.sum(np.array([Jac_vecs[:,i]*Hess_mat[:,i,j] for i in range(Jac_vecs.shape[1])]), axis=0) for j in range(Jac_vecs.shape[1])])
            term = np.sum(np.array([Jac_vecs[:,i]*term_temp.T[:,i] for i in range(Jac_vecs.shape[1])]),axis=0)
            
            term2 = fx*np.abs(fy_jac)/(2*self.fy_interp(mu)**2)*term/(np.linalg.norm(Jac_vecs, axis=1)**4 + self.c_w3*self.tol)
            w =  self.c_w3*np.abs(term2)
                
        return w[:,None]
    

    @staticmethod
    def check_weight_type(weight_type):
        assert(weight_type.lower() in ["nominal", "importance", "importance_ho"])
        return weight_type.lower()
