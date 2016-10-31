import lasagne
import theano.tensor as T


class RepeatLayer(lasagne.layers.Layer):

    """
    Layer, which repeats its in n_times along axis = 0
    """

    def __init__(self, layer, n_times):
        self.n_times = n_times
        super(RepeatLayer,self).__init__(layer)

    def get_output_for(self, input, **kwargs):
        return T.repeat(input, self.n_times, axis=0)

    def get_output_shape_for(self, input_shape, **kwargs):
        if (type(self.n_times) is int) and (input_shape[0] is not None):
            return (self.n_times*input_shape[0],) +  tuple(input_shape[1:])
        else:
            return (None,)+ tuple(input_shape[1:])

class BroadcastableSumLayer(lasagne.layers.MergeLayer):

    """
    Layer, which gets two layers of shape which differs only along axis 0:, so that
    [n, :] and [1, :], repeats last layers n times along axis=0 and then applies
    elementwise sum
    """

    def __init__(self, l_weather, l_relief):
        super(BroadcastableSumLayer, self).__init__([l_weather, l_relief])

    def get_output_for(self, inputs, **kwargs):
        w, r = inputs
        r = T.repeat(r, w.shape[0], axis=0)
        return w + r

    def get_output_shape_for(self, input_shapes, **kwargs):
        w_shape, r_shape = input_shapes
        return w_shape


class FlipLayer(lasagne.layers.Layer):

    """
    Makes a flip for the last 2 dimensions
    """

    def __init__(self, l_weather, l_relief):
        super(BroadcastableSumLayer, self).__init__([l_weather, l_relief])

    def get_output_for(self, inputs, **kwargs):
        w, r = inputs
        r = T.repeat(r, w.shape[0], axis=0)
        return w + r

    def get_output_shape_for(self, input_shapes, **kwargs):
        w_shape, r_shape = input_shapes
        return w_shape


