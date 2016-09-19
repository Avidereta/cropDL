import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer
from lasagne.regularization import regularize_network_params, l2


from agentnet.agent import Recurrence
from model import Model

from lasagne.layers import batch_norm

class ClassiCNN(Model):

    def __init__(self,
                 X_layer,
                 n_hidden=16,
                 name="ClassiCNN",
                 filter_size=[5, 5]):

        DATA_TENSOR_SHAPE = 4 #[batch_size, nmb_channels, h, w]
        assert len(X_layer.output_shape) == DATA_TENSOR_SHAPE

        super(ClassiCNN, self).__init__(name)

        input_shape = X_layer.output_shape
        print "input_shape", input_shape
        self.build_graph(input_shape, n_hidden, filter_size=filter_size)
        self.weights = lasagne.layers.get_all_params(self.cnn_out, trainable=True)

        self.pred_y_value = lasagne.layers.get_output(self.cnn_out, X_layer.input_var)
        # [none, 1] none is batch_size
        # get rid of "1": [none]
        self.pred_y_value = self.pred_y_value[:, 0]

        # hidden states
        # [none, n_hidden, input_shape[0], input_shape[1]]
        self.hidden_maps_batch = lasagne.layers.get_output(self.last_conv, X_layer.input_var)


        if isinstance(X_layer, InputLayer):
            self.predict_fun = theano.function([X_layer.input_var], self.pred_y_value)

        if isinstance(X_layer,InputLayer):
            self.hidden_fun = theano.function([X_layer.input_var], self.hidden_maps_batch)


    def build_graph(self, input_shape, n_hidden, filter_size=[5, 5]):
        input_shape = tuple(input_shape)

        self.Xi_inp = InputLayer(input_shape, name='x_i')

        # TODO add dropouts
        # Convolution with Xi_inp: [None,nmb_channels,input_shape] --> [None,n_hidden,input_shape]
        conv1_from_inp = Conv2DLayer(self.Xi_inp,
                                    num_filters=n_hidden,
                                    filter_size=filter_size,  # arbitrary
                                    nonlinearity=lasagne.nonlinearities.rectify,
                                    W = lasagne.init.GlorotNormal(gain='relu'),
                                    # pad='same',
                                    name='conv1 from input')

        pool = MaxPool2DLayer(batch_norm(conv1_from_inp),
                              pool_size = 2,
                              stride=None,
                              pad=(0, 0),
                              ignore_border=True)

        conv2 = Conv2DLayer(batch_norm(pool),
                                    num_filters=n_hidden,
                                    filter_size=filter_size,  # arbitrary
                                    nonlinearity=lasagne.nonlinearities.rectify,
                                    W=lasagne.init.GlorotNormal(gain='relu'),
                                    # pad='same',
                                    name='conv2')

        pool = MaxPool2DLayer(batch_norm(conv2),
                              pool_size = 2,
                              stride=None,
                              pad=(0, 0),
                              ignore_border=True)

        conv3 = Conv2DLayer(batch_norm(pool),
                                    num_filters=n_hidden,
                                    filter_size=filter_size,  # arbitrary
                                    b=None,  # only apply bias once in from_hid
                                    nonlinearity=lasagne.nonlinearities.rectify,
                                    W=lasagne.init.GlorotNormal(gain='relu'),
                                    # pad='same',
                                    name='conv3')

        pool = MaxPool2DLayer(batch_norm(conv3),
                              pool_size = 2,
                              stride=None,
                              pad=(0, 0),
                              ignore_border=True)

        conv4 = Conv2DLayer(batch_norm(pool),
                                    num_filters=n_hidden,
                                    filter_size=filter_size,  # arbitrary
                                    nonlinearity=lasagne.nonlinearities.rectify,
                                    W=lasagne.init.GlorotNormal(gain='relu'),
                                    # pad='same',
                                    name='conv4')

        self.last_conv = Conv2DLayer(batch_norm(conv4),
                                    num_filters=n_hidden,
                                    filter_size=filter_size,  # arbitrary
                                    b=None,  # only apply bias once in from_hid
                                    nonlinearity=lasagne.nonlinearities.rectify,
                                    W=lasagne.init.GlorotNormal(gain='relu'),
                                    # pad='same',
                                    name='last_conv')

        self.cnn_out = lasagne.layers.DenseLayer(batch_norm(self.last_conv),
                                                    num_units=1,
                                                    nonlinearity=lasagne.nonlinearities.rectify,
                                                    name='dense layer')



    def predict(self, X_layer, *args,**kwargs):
        """
        Input:  X_layer: len(X_layer.output_shape) = 4
        """
        if not hasattr(self, 'predict_fun'):
            raise ValueError("you should add a valid predict_fun to this class"
                             "(no automatic predict generates since X_layer isn't an InputLayer")
        return self.predict_fun(X_layer)


    def predict_hidden(self, X_layer):
        """
        Input:  X_layer: len(X_layer.output_shape) = 4
        """
        if not hasattr(self,'hidden_fun'):
            raise ValueError("you should add a valid hidden_fun to this class"
                             "(no automatic predict generates since X_seq isn't an InputLayer")
        return self.hidden_fun(X_layer)


    def get_loss_components(self, Y):

        # print "SHAPE pred_y_value", self.pred_y_value.get_output_shape(), Y.shape
        mae = T.abs_(self.pred_y_value - Y).mean()
        mse = T.sqr(self.pred_y_value - Y).mean()

        reg_l2 = regularize_network_params(self.cnn_out, l2) * 10 ** -5


        return mae, mse, reg_l2
