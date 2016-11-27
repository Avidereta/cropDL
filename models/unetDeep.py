import theano
import lasagne
from models.model import Model
from collections import OrderedDict
from lasagne.layers import Conv2DLayer, InputLayer, ConcatLayer, Pool2DLayer, \
    ReshapeLayer, DimshuffleLayer, NonlinearityLayer, DropoutLayer, Deconv2DLayer
import theano.tensor as T
from lasagne.layers import batch_norm
from lasagne.regularization import regularize_network_params, l2

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, recall_score, precision_score

from metrics import roc_auc_loss

PRED_MIN = 1e-10
PRED_MAX = 1.

class uNet(Model):
    def __init__(self,
                 X_layer,
                 n_filters=64,
                 filter_size=3,
                 name='unet',
                 pad='valid',
                 nmb_out_classes=2,
                 do_dropout=False):

        super(uNet, self).__init__(name)

        input_shape = X_layer.output_shape

        self.build_graph(input_shape, n_filters, filter_size, nmb_out_classes, do_dropout, pad)
        self.weights = lasagne.layers.get_all_params(self.net['output_flattened'], trainable=True)

        self.pred_y = lasagne.layers.get_output(self.net['output_flattened'], X_layer.input_var)
        self.outlayer_for_loss = self.net['output_flattened']
        print 'self.outlayer_for_loss.output_shape', self.outlayer_for_loss.output_shape

        self.outlayer_seg = self.net['output_segmentation']
        self.pred_img_shape = self.outlayer_seg.output_shape[2:]
        seg_output = lasagne.layers.get_output(self.outlayer_seg, X_layer.input_var)
        if isinstance(X_layer, lasagne.layers.InputLayer):
            self.predict_fun = theano.function([X_layer.input_var], seg_output)

        self.input_img_shape = input_shape[2:]
        self.pred_img_shape = self.outlayer_seg.output_shape[2:]
        
        
        self.before_sigm = lasagne.layers.get_output(self.net['output_flattened_before_sigm'], X_layer.input_var)

    def build_graph(self, input_shape, n_filters, filter_size, nmb_out_classes, do_dropout, pad):

        nonlinearity = lasagne.nonlinearities.rectify
        self.net = OrderedDict()
        self.net['input'] = InputLayer(input_shape, name='input')
        print "Input: self.net['input']", self.net['input'].output_shape

        ### Conv1
        self.net['conv_1_1'] = Conv2DLayer(self.net['input'],
                                           num_filters=n_filters,
                                           filter_size=filter_size,
                                           nonlinearity=nonlinearity,
                                           W=lasagne.init.GlorotNormal(),
                                           pad=pad)
        print "\nConv11: self.net['conv_1_1']", self.net['conv_1_1'].output_shape

        self.net['dropout1'] = DropoutLayer(self.net['conv_1_1'], p=0.2)
        print "Dropout1: self.net['dropout1']", self.net['dropout1'].output_shape

        self.net['conv_1_2'] = Conv2DLayer(batch_norm(self.net['dropout1']),
                                           num_filters=n_filters,
                                           filter_size=filter_size,
                                           nonlinearity=nonlinearity,
                                           W=lasagne.init.GlorotNormal(),
                                           pad=pad)
        print "Conv12: self.net['conv_1_2']", self.net['conv_1_2'].output_shape

        self.net['pool1'] = Pool2DLayer(batch_norm(self.net['conv_1_2']), 2)
        print "\nPool1: self.net['pool1']", self.net['pool1'].output_shape

        ### Conv2
        self.net['conv_2_1'] = Conv2DLayer(batch_norm(self.net['pool1']),
                                           num_filters=n_filters * 2,
                                           filter_size=filter_size,
                                           nonlinearity=nonlinearity,
                                           W=lasagne.init.GlorotNormal(),
                                           pad=pad)
        print "Conv21: self.net['conv_2_1']", self.net['conv_2_1'].output_shape

        self.net['dropout2'] = DropoutLayer(batch_norm(self.net['conv_2_1']), p=0.2)
        print "Dropout2: self.net['dropout2']", self.net['dropout2'].output_shape

        self.net['conv_2_2'] = Conv2DLayer(batch_norm(self.net['dropout2']),
                                           num_filters=n_filters * 2,
                                           filter_size=filter_size,
                                           nonlinearity=nonlinearity,
                                           W=lasagne.init.GlorotNormal(),
                                           pad=pad)
        print "Conv22: self.net['conv_2_2']", self.net['conv_2_2'].output_shape

        self.net['pool2'] = Pool2DLayer(self.net['conv_2_2'], 2)
        print "\nPool2: self.net['pool2']", self.net['pool2'].output_shape

        ### Conv3
        self.net['conv_3_1'] = Conv2DLayer(batch_norm(self.net['pool2']),
                                           num_filters=n_filters * 4,
                                           filter_size=filter_size,
                                           nonlinearity=nonlinearity,
                                           W=lasagne.init.GlorotNormal(),
                                           pad=pad)
        print "Conv31: self.net['conv_3_1']", self.net['conv_3_1'].output_shape

        self.net['dropout3'] = DropoutLayer(self.net['conv_3_1'], p=0.2)
        print "Dropout3: self.net['dropout3']", self.net['dropout3'].output_shape

        self.net['conv_3_2'] = Conv2DLayer(batch_norm(self.net['dropout3']),
                                           num_filters=n_filters * 4,
                                           filter_size=filter_size,
                                           nonlinearity=nonlinearity,
                                           W=lasagne.init.GlorotNormal(),
                                           pad=pad)
        print "Conv32: self.net['conv_3_2']", self.net['conv_3_2'].output_shape

        self.net['pool3'] = Pool2DLayer(self.net['conv_3_2'], 2)
        print "\nPool3: self.net['pool3']", self.net['pool3'].output_shape

        ### Conv4
        self.net['conv_4_1'] = Conv2DLayer(batch_norm(self.net['pool3']),
                                           num_filters=n_filters * 8,
                                           filter_size=filter_size,
                                           nonlinearity=nonlinearity,
                                           W=lasagne.init.GlorotNormal(),
                                           pad=pad)
        print "Conv41: self.net['conv_4_1']", self.net['conv_4_1'].output_shape

        self.net['conv_4_2'] = Conv2DLayer(batch_norm(self.net['conv_4_1']),
                                           num_filters=n_filters * 8,
                                           filter_size=filter_size,
                                           nonlinearity=nonlinearity,
                                           W=lasagne.init.GlorotNormal(),
                                           pad=pad)
        print "Conv42: self.net['conv_4_2']", self.net['conv_4_2'].output_shape

        self.net['dropout4'] = DropoutLayer(self.net['conv_4_2'], p=0.5)
        print "Dropout4: self.net['dropout4']", self.net['dropout4'].output_shape

        self.net['pool4'] = Pool2DLayer(self.net['dropout4'], 2)
        print "\nPool4: self.net['pool4']", self.net['pool4'].output_shape

        ### Conv5
        self.net['conv_5_1'] = Conv2DLayer(batch_norm(self.net['pool4']),
                                           num_filters=n_filters * 16,
                                           filter_size=filter_size,
                                           nonlinearity=nonlinearity,
                                           W=lasagne.init.GlorotNormal(),
                                           pad=pad)
        print "Conv51: self.net['conv_5_1']", self.net['conv_5_1'].output_shape

        self.net['conv_5_2'] = Conv2DLayer(batch_norm(self.net['conv_5_1']),
                                           num_filters=n_filters * 16,
                                           filter_size=filter_size,
                                           nonlinearity=nonlinearity,
                                           W=lasagne.init.GlorotNormal(),
                                           pad=pad)
        print "Conv52: self.net['conv_5_2']", self.net['conv_5_2'].output_shape

        self.net['dropout5'] = DropoutLayer(self.net['conv_5_2'], p=0.5)
        print "Dropout5: self.net['dropout5']", self.net['dropout5'].output_shape

        ### Deconv1
        self.net['deconv1'] = Deconv2DLayer(batch_norm(self.net['dropout5']),
                                            num_filters=n_filters * 8,
                                            filter_size=2,
                                            nonlinearity=nonlinearity,
                                            W=lasagne.init.GlorotNormal(),
                                            stride=2,
                                            crop='valid')
        print "\nDeconv1: self.net['deconv1']", self.net['deconv1'].output_shape

        self.net['concat1'] = ConcatLayer([self.net['deconv1'], self.net['dropout4']],
                                          cropping=(None, None, "center", "center"))
        print "Concat1: self.net['concat1']", self.net['concat1'].output_shape

        self.net['convde_1_1'] = Conv2DLayer(self.net['concat1'],
                                             num_filters=n_filters * 8,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             W=lasagne.init.GlorotNormal(),
                                             pad=pad)
        print "Convde11: self.net['convde_1_1']", self.net['convde_1_1'].output_shape

        self.net['convde_1_2'] = Conv2DLayer(batch_norm(self.net['convde_1_1']),
                                             num_filters=n_filters * 8,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             W=lasagne.init.GlorotNormal(),
                                             pad=pad)
        print "Convde12: self.net['convde_1_2']", self.net['convde_1_2'].output_shape

        ### Deconv2
        self.net['deconv2'] = Deconv2DLayer(batch_norm(self.net['convde_1_2']),
                                            num_filters=n_filters * 4,
                                            filter_size=2,
                                            nonlinearity=nonlinearity,
                                            W=lasagne.init.GlorotNormal(),
                                            stride=2,
                                            crop=pad)
        print "\nDeconv2: self.net['deconv2']", self.net['deconv2'].output_shape

        self.net['concat2'] = ConcatLayer([self.net['deconv2'], self.net['conv_3_2']],
                                          cropping=(None, None, "center", "center"))
        print "Concat2: self.net['concat2']", self.net['concat2'].output_shape

        self.net['convde_2_1'] = Conv2DLayer(self.net['concat2'],
                                             num_filters=n_filters * 4,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             W=lasagne.init.GlorotNormal(),
                                             pad=pad)
        print "Convde21: self.net['convde_2_1']", self.net['convde_2_1'].output_shape

        self.net['convde_2_2'] = Conv2DLayer(batch_norm(self.net['convde_2_1']),
                                             num_filters=n_filters * 4,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             W=lasagne.init.GlorotNormal(),
                                             pad=pad)
        print "Convde22: self.net['convde_2_2']", self.net['convde_2_2'].output_shape

        ### Deconv3
        self.net['deconv3'] = Deconv2DLayer(batch_norm(self.net['convde_2_2']),
                                            num_filters=n_filters * 2,
                                            filter_size=2,
                                            nonlinearity=nonlinearity,
                                            W=lasagne.init.GlorotNormal(),
                                            stride=2,
                                            crop=pad)
        print "\nDeconv3: self.net['deconv3']", self.net['deconv3'].output_shape

        self.net['concat3'] = ConcatLayer([self.net['deconv3'], self.net['conv_2_2']],
                                          cropping=(None, None, "center", "center"))
        print "Concat3: self.net['concat3']", self.net['concat3'].output_shape

        self.net['convde_3_1'] = Conv2DLayer(self.net['concat3'],
                                             num_filters=n_filters * 2,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             W=lasagne.init.GlorotNormal(),
                                             pad=pad)
        print "Convde31: self.net['convde_3_1']", self.net['convde_3_1'].output_shape

        self.net['convde_3_2'] = Conv2DLayer(batch_norm(self.net['convde_3_1']),
                                             num_filters=n_filters * 2,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             W=lasagne.init.GlorotNormal(),
                                             pad=pad)
        print "Convde32: self.net['convde_3_2']", self.net['convde_3_2'].output_shape

        ### Deconv4
        self.net['deconv4'] = Deconv2DLayer(batch_norm(self.net['convde_3_2']),
                                            num_filters=n_filters,
                                            filter_size=2,
                                            nonlinearity=nonlinearity,
                                            W=lasagne.init.GlorotNormal(),
                                            stride=2,
                                            crop=pad)
        print "\nDeconv4: self.net['deconv4']", self.net['deconv4'].output_shape

        self.net['concat4'] = ConcatLayer([self.net['deconv4'], self.net['conv_1_2']],
                                          cropping=(None, None, "center", "center"))
        print "Concat4: self.net['concat4']", self.net['concat4'].output_shape

        self.net['convde_4_1'] = Conv2DLayer(self.net['concat4'],
                                             num_filters=n_filters,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             W=lasagne.init.GlorotNormal(),
                                             pad=pad)
        print "Convde41: self.net['convde_4_1']", self.net['convde_4_1'].output_shape

        self.net['convde_4_2'] = Conv2DLayer(batch_norm(self.net['convde_4_1']),
                                             num_filters=n_filters,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             W=lasagne.init.GlorotNormal(),
                                             pad=pad)
        print "Convde42: self.net['convde_4_2']", self.net['convde_4_2'].output_shape

        ####
        self.net['output'] = Conv2DLayer(self.net['convde_4_2'],
                                         num_filters=nmb_out_classes,
                                         filter_size=1,
                                         nonlinearity=None)
        print "\nself.net['output']", self.net['output'].output_shape

        ####
        self.net['dimshuffle'] = DimshuffleLayer(self.net['output'], (1, 0, 2, 3))
        shuffled_output_shape = self.net['dimshuffle'].output_shape
        print "self.net['dimshuffle']", self.net['dimshuffle'].output_shape

        self.net['output_flattened_before_sigm'] = ReshapeLayer(self.net['dimshuffle'], (nmb_out_classes, -1))
        print "self.net['output_flattened_before_sigm']", self.net['output_flattened_before_sigm'].output_shape

        self.net['output_flattened'] = NonlinearityLayer(self.net['output_flattened_before_sigm'],
                                                         nonlinearity=lasagne.nonlinearities.sigmoid)

        self.net['reshaped_flattened'] = ReshapeLayer(self.net['output_flattened'], shuffled_output_shape)
        print "self.net['reshaped_flattened']", self.net['reshaped_flattened'].output_shape

        self.net['output_segmentation'] = DimshuffleLayer(self.net['reshaped_flattened'], (1, 0, 2, 3))
        print "self.net['output_segmentation']", self.net['output_segmentation'].output_shape

    def predict(self, X, *args, **kwargs):
        """
        Input:  X
        """
        if not hasattr(self, 'predict_fun'):
            raise ValueError("you should add a valid predict_fun to this class"
                             "(no automatic predict generates since X isn't an InputLayer")
        return self.predict_fun(X)

    def get_loss_components(self, target, weights):
        """
        @param: target: theano vector
        @param: weights: theano vector
        """
        
        target = T.transpose(target)      
        
        ce = lasagne.objectives.binary_crossentropy(T.clip(self.pred_y,PRED_MIN,PRED_MAX), target)
        # put weights = weights if want
        ce_weighed = lasagne.objectives.aggregate(ce, weights=None, mode='mean')

        reg_l2 = regularize_network_params(self.outlayer_for_loss, l2) * 10 ** -5
        
        max_pred = self.before_sigm.max()
        min_pred = self.before_sigm.min()
        
        return ce_weighed, reg_l2, max_pred, min_pred
