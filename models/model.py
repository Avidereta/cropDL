class Model(object):
    ##not required here
    def __init__(self,name=None):

        self.name = name or str(self.__class__)

    def predict(self, X, *args,**kwargs):
        """ :returns: numpy tensor4[batch,time,lat,lon] of float32 of ONE-STEP predictions
        (e.g. probabilities or actual precipitation)"""
        predicted = 1000
        return predicted

