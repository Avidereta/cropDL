from model import Model
import numpy as np

class RandomModel(Model):
    def __init__(self,name=None):

        super(RandomModel, self).__init__(name)

    def predict(self, X, pred_shape, *args,**kwargs):
        """ :returns: random predictions"""
        predicted = np.random.rand(X.shape[0], 1, pred_shape[0], pred_shape[1])
        return predicted