import theano
import lasagne
from models.model import Model
from collections import OrderedDict
from lasagne.layers import Conv2DLayer, InputLayer, ConcatLayer, Pool2DLayer, \
    ReshapeLayer, DimshuffleLayer, NonlinearityLayer, DropoutLayer, Upscale2DLayer
import theano.tensor as T
from lasagne.regularization import regularize_network_params, l2


class uNet(Model):
    def __init__(self,
                 X_layer,
                 n_filters=8,
                 filter_size=3,
                 name='unet',
                 pad='same',
                 nmb_out_classes=2,
                 do_dropout=False):

        super(uNet, self).__init__(name)

        input_shape = X_layer.output_shape
        self.build_graph(input_shape, n_filters, filter_size, nmb_out_classes, do_dropout, pad)
        self.weights = lasagne.layers.get_all_params(self.net['output_flattened'], trainable=True)

        print self.net['output_segmentation'].output_shape, self.net['output_flattened'].output_shape

        self.pred_y = lasagne.layers.get_output(self.net['output_flattened'], X_layer.input_var)

        self.outlayer_for_loss = self.net['output_flattened']
        self.outlayer_seg = self.net['output_segmentation']

        if isinstance(X_layer, lasagne.layers.InputLayer):
            self.predict_fun = theano.function([X_layer.input_var], self.pred_y)

    def build_graph(self, input_shape, n_filters, filter_size, nmb_out_classes, do_dropout, pad='same'):

        nonlinearity = lasagne.nonlinearities.rectify
        self.net = OrderedDict()
        self.net['input'] = InputLayer(input_shape, name='input')

        self.net['contr_1_1'] = Conv2DLayer(self.net['input'],
                                            num_filters=n_filters,
                                            filter_size=filter_size,
                                            nonlinearity=nonlinearity,
                                            pad=pad)

        self.net['contr_1_2'] = Conv2DLayer(self.net['contr_1_1'],
                                            num_filters=n_filters,
                                            filter_size=filter_size,
                                            nonlinearity=nonlinearity,
                                            pad=pad)
        self.net['pool1'] = Pool2DLayer(self.net['contr_1_2'], 2)

        self.net['contr_2_1'] = Conv2DLayer(self.net['pool1'],
                                            num_filters=n_filters * 2,
                                            filter_size=filter_size,
                                            nonlinearity=nonlinearity,
                                            pad=pad)
        self.net['contr_2_2'] = Conv2DLayer(self.net['contr_2_1'],
                                            num_filters=n_filters * 2,
                                            filter_size=filter_size,
                                            nonlinearity=nonlinearity,
                                            pad=pad)
        self.net['pool2'] = Pool2DLayer(self.net['contr_2_2'], 2)

        self.net['contr_filter_size_1'] = Conv2DLayer(self.net['pool2'],
                                            num_filters=n_filters * 4,
                                            filter_size=filter_size,
                                            nonlinearity=nonlinearity,
                                            pad=pad)
        self.net['contr_filter_size_2'] = Conv2DLayer(self.net['contr_filter_size_1'],
                                            num_filters=n_filters * 4,
                                            filter_size=filter_size,
                                            nonlinearity=nonlinearity,
                                            pad=pad)
        self.net['poolfilter_size'] = Pool2DLayer(self.net['contr_filter_size_2'], 2)

        self.net['contr_4_1'] = Conv2DLayer(self.net['poolfilter_size'],
                                            num_filters=n_filters * 8,
                                            filter_size=filter_size,
                                            nonlinearity=nonlinearity,
                                            pad=pad)
        self.net['contr_4_2'] = Conv2DLayer(self.net['contr_4_1'],
                                            num_filters=n_filters * 8,
                                            filter_size=filter_size,
                                            nonlinearity=nonlinearity,
                                            pad=pad)
        l = self.net['pool4'] = Pool2DLayer(self.net['contr_4_2'], 2)

        # TODO reunderstand deconvolutions

        # the paper does not really describe where and how dropout is added. Feel free to try more options
        if do_dropout:
            l = DropoutLayer(l, p=0.4)

        self.net['encode_1'] = Conv2DLayer(l,
                                           num_filters=n_filters * 16,
                                           filter_size=filter_size,
                                           nonlinearity=nonlinearity,
                                           pad=pad)
        self.net['encode_2'] = Conv2DLayer(self.net['encode_1'],
                                           num_filters=n_filters * 16,
                                           filter_size=filter_size,
                                           nonlinearity=nonlinearity,
                                           pad=pad)
        self.net['deconv1'] = Upscale2DLayer(self.net['encode_2'], 2)
        print "self.net['deconv1'].shape, self.net['contr_4_2'].shape", self.net['deconv1'].output_shape, self.net[
            'contr_4_2'].output_shape

        self.net['concat1'] = ConcatLayer([self.net['deconv1'], self.net['contr_4_2']],
                                          cropping=(None, None, "center", "center"))
        self.net['expand_1_1'] = Conv2DLayer(self.net['concat1'],
                                             num_filters=n_filters * 8,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             pad=pad)
        self.net['expand_1_2'] = Conv2DLayer(self.net['expand_1_1'],
                                             num_filters=n_filters * 8,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             pad=pad)
        self.net['deconv2'] = Upscale2DLayer(self.net['expand_1_2'], 2)

        self.net['concat2'] = ConcatLayer([self.net['deconv2'], self.net['contr_filter_size_2']],
                                          cropping=(None, None, "center", "center"))
        self.net['expand_2_1'] = Conv2DLayer(self.net['concat2'],
                                             num_filters=n_filters * 4,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             pad=pad)
        self.net['expand_2_2'] = Conv2DLayer(self.net['expand_2_1'],
                                             num_filters=n_filters * 4,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             pad=pad)
        self.net['deconvfilter_size'] = Upscale2DLayer(self.net['expand_2_2'], 2)

        self.net['concatfilter_size'] = ConcatLayer([self.net['deconvfilter_size'], self.net['contr_2_2']],
                                          cropping=(None, None, "center", "center"))
        self.net['expand_filter_size_1'] = Conv2DLayer(self.net['concatfilter_size'],
                                             num_filters=n_filters * 2,
                                             filter_size=filter_size,
                                             nonlinearity=nonlinearity,
                                             pad=pad)
        self.net['expand_filter_size_2'] = Conv2DLayer(self.net['expand_filter_size_1'],
                                             n_filters * 2,
                                             filter_size,
                                             nonlinearity=nonlinearity,
                                             pad=pad)
        self.net['deconv4'] = Upscale2DLayer(self.net['expand_filter_size_2'], 2)

        self.net['concat4'] = ConcatLayer([self.net['deconv4'], self.net['contr_1_2']],
                                          cropping=(None, None, "center", "center"))
        self.net['expand_4_1'] = Conv2DLayer(self.net['concat4'],
                                             n_filters,
                                             filter_size,
                                             nonlinearity=nonlinearity,
                                             pad=pad)
        self.net['expand_4_2'] = Conv2DLayer(self.net['expand_4_1'],
                                             n_filters,
                                             filter_size,
                                             nonlinearity=nonlinearity,
                                             pad=pad)

        self.net['output_segmentation'] = Conv2DLayer(self.net['expand_4_2'], nmb_out_classes, 1, nonlinearity=None)
        print "self.net['output_segmentation']", self.net['output_segmentation'].output_shape

        self.net['dimshuffle'] = DimshuffleLayer(self.net['output_segmentation'], (1, 0, 2, filter_size))
        print "self.net['dimshuffle']", self.net['dimshuffle'].output_shape

        self.net['reshapeSeg'] = ReshapeLayer(self.net['dimshuffle'], (nmb_out_classes, -1))
        print "self.net['reshapeSeg']", self.net['reshapeSeg'].output_shape

        self.net['dimshuffle2'] = DimshuffleLayer(self.net['reshapeSeg'], (1, 0))
        print "self.net['dimshuffle2']", self.net['dimshuffle2'].output_shape

        self.net['output_flattened'] = NonlinearityLayer(self.net['reshapeSeg'],
                                                         nonlinearity=lasagne.nonlinearities.softmax)
        print "self.net['output_flattened']", self.net['output_flattened'].output_shape

    def predict(self, X, *args, **kwargs):
        """
        Input:  X
        """
        if not hasattr(self, 'predict_fun'):
            raise ValueError("you should add a valid predict_fun to this class"
                             "(no automatic predict generates since X_layer isn't an InputLayer")
        return self.predict_fun(X)

    def get_loss_components(self, target, weights):

        target = T.transpose(target)
        ce = lasagne.objectives.binary_crossentropy(self.pred_y, target)
        ce_weighed = lasagne.objectives.aggregate(ce, weights=weights, mode='mean')

        reg_l2 = regularize_network_params(self.outlayer_for_loss, l2) * 10 ** -5
        acc = T.mean(T.eq(T.argmax(self.pred_y, axis=0), target), dtype=theano.config.floatX)

        return ce_weighed, reg_l2, acc
