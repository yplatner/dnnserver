from django.db import models
from django.contrib.auth.models import User

class Update(models.Model):
    UPDATE_STRATEGIES = (
        ('sgd',                 'Stochastic Gradient Descent'),
        ('momentum',            'Stochastic Gradient Descent with Momentum'),
        ('nesterov_momentum',   'Stochastic Gradient Descent with Noesterov Momentum'),
        ('adagrad',             'Adagrad'),
        ('rmsprop',             'RMSProp'),
        ('adadelta',            'Adadelta'),
        ('adam',                'Adam'),
    )
    name                = models.CharField(max_length=200, default='')
    strategy            = models.CharField(max_length=50, choices=UPDATE_STRATEGIES, default="sgd")
    learning_rate       = models.FloatField(default=1)
    momentum            = models.FloatField(default=0.9)
    rho                 = models.FloatField(default=0.9)
    epsilon             = models.FloatField(default=0.000001)
    beta1               = models.FloatField(default=0.9)
    beta2               = models.FloatField(default=0.999)
    def __str__(self):
        return self.name

class Network(models.Model):
    LOSS_OBJECTIVES = (
        ('binary_crossentropy',         'Binary Crossentropy'),
        ('categorical_crossentropy',    'Categorical Crossentropy'),
        ('squared_error',               'Squared Error'),
        ('binary_hinge_loss',           'Binary Hinge Loss'),
        ('multiclass_hinge_loss',       'Multiclass Hinge Loss'),
    )
    PENALTIES = (
        ('l1',                          'L1'),
        ('l2',                          'L2'),
    )
    name                = models.CharField(max_length=200, default='')
    user                = models.ForeignKey(User, blank=True, null=True, on_delete=models.SET_NULL)
    update              = models.ForeignKey(Update, blank=True, null=True, on_delete=models.SET_NULL)
    loss_objective      = models.CharField(max_length=50, choices=LOSS_OBJECTIVES, default="binary_crossentropy")
    loss_delta          = models.FloatField(default=1)
    regularization      = models.BooleanField(default=False)
    penalty             = models.CharField(max_length=50, choices=PENALTIES, default="l1")
    coefficient         = models.FloatField(default=1)
    generated           = models.BooleanField(default=False)
    creation_date       = models.DateTimeField(null=True)
    def __str__(self):
        return self.name

class Weight(models.Model):
    INITIALIZERS = (
        ('Constant',                    'Initialize weights with constant value.'),
        ('Normal',                      'Sample initial weights from the Gaussian distribution.'),
        ('Uniform',                     'Sample initial weights from the uniform distribution.'),
        #('Glorot',                      'Glorot weight initialization.'),
        ('GlorotNormal',                'Glorot with weights sampled from the Normal distribution.'),
        ('GlorotUniform',               'Glorot with weights sampled from the Uniform distribution.'),
        #('He',                          'He weight initialization.'),
        ('HeNormal',                    'He initializer with weights sampled from the Normal distribution.'),
        ('HeUniform',                   'He initializer with weights sampled from the Uniform distribution.'),
        ('Orthogonal',                  'Intialize weights as Orthogonal matrix.'),
        ('Sparse',                      'Initialize weights as sparse matrix.'),
    )
    name                = models.CharField(max_length=200, default='')
    init                = models.CharField(max_length=200, choices=INITIALIZERS, default="GlorotUniform")
    init_val            = models.FloatField(default=0)
    init_std            = models.FloatField(default=0.01)
    init_mean           = models.FloatField(default=0)
    init_range          = models.FloatField(default=-0.01)
    init_range_high     = models.FloatField(default=0.01)
    init_gain           = models.FloatField(default=1.0)
    init_gain_relu      = models.BooleanField(default=False)
    init_c01b           = models.BooleanField(default=False)
    init_sparsity       = models.FloatField(default=0.1)
    def __str__(self):
        return self.name
    
class Layer(models.Model):
    LAYER_TYPES = (
    	('InputLayer',            	            'A network input layer'),
    	('DenseLayer',            	            'A fully connected layer.'),
    	#('NINLayer',                           'Network-in-network layer.'),
    	#('Conv1DLayer',                        '1D convolutional layer'),
    	('Conv2DLayer',            	            '2D convolutional layer'),
    	#('MaxPool1DLayer',        	            '1D max-pooling layer'),
    	('MaxPool2DLayer',        	            '2D max-pooling layer'),
    	#('Pool1DLayer',                        '1D pooling layer'),
    	#('Pool2DLayer',                        '2D pooling layer'),
    	#('Upscale1DLayer',        	            '1D upscaling layer'),
    	#('Upscale2DLayer',        	            '2D upscaling layer'),
    	#('GlobalPoolLayer',                    'Global pooling layer'),
    	#('FeaturePoolLayer',                   'Feature pooling layer'),
    	#('FeatureWTALayer',                    'Winner Take All layer'),
    	##('CustomRecurrentLayer',              'A layer which implements a recurrent connection.'),
    	#('RecurrentLayer',        	            'Dense recurrent neural network (RNN) layer'),
    	#('LSTMLayer',                          'A long short-term memory (LSTM) layer.'),
    	#('GRULayer',                           'Gated Recurrent Unit (GRU) Layer'),
    	##('Gate',                              'Simple class to hold the parameters for a gate connection.'),
    	('DropoutLayer',                        'Dropout layer'),
    	#('GaussianNoiseLayer',    	            'Gaussian noise layer.'),
    	#('ReshapeLayer',                       'A layer reshaping its input tensor to another tensor of the same total number of elements.'),
    	#('FlattenLayer',                       'A layer that flattens its input.'),
    	#('DimshuffleLayer',                    'Dimshuffle - A layer that rearranges the dimension of its input tensor, maintaining the same same total number of elements.'),
    	#('PadLayer',                           'Pad all dimensions except the first batch_ndim with width zeros on both sides, or with another value specified in val.'),
    	#('SliceLayer',            	            'Slices the input at a specific axis and at specific indices.'),
    	#('ConcatLayer',                        'Concatenates multiple inputs along the specified axis.'),
    	#('ElemwiseMergeLayer',    	            'This layer performs an elementwise merge of its input layers.'),
    	#('ElemwiseSumLayer',                   'This layer performs an elementwise sum of its input layers.'),
    	('LocalResponseNormalization2DLayer',   'Cross-channel Local Response Normalization for 2D feature maps.'),
        #('BatchNormLayer',	                    'Batch Normalization'),
    	#('EmbeddingLayer',        	            'A layer for word embeddings.'),
    	('NonlinearityLayer',                  'A layer that just applies a nonlinearity.'),
    	#('BiasLayer',                          'A layer that just adds a (trainable) bias term.'),
    	##('ExpressionLayer',                   'This layer provides boilerplate for a custom layer that applies a simple transformation to the input.'),
    	#('InverseLayer',                       'The InverseLayer performs inverse operations for a single layer of a neural network by applying the partial derivative of the layer to be inverted with respect to its input: transposed layer for a DenseLayer, deconvolutional layer for Conv2DLayer, Conv1DLayer; or an unpooling layer for MaxPool2DLayer.'),
    	#('TransformerLayer',                   'Spatial transformer layer'),
    	#('ParametricRectifierLayer',           'A layer that applies parametric rectify nonlinearity to its input'),
    )
    NONLINEARITIES = (
        ('None',                        'Linear'),
        ('sigmoid',                     'Sigmoid activation function'),
        ('softmax',                     'Softmax activation function'),
        ('tanh',                        'Tanh activation function'),
        #('ScaledTanH',                 'Scaled tanh'),
        ('rectify',                     'Rectify activation function'),
        #('LeakyRectify',               'Leaky rectifier'),
        ('leaky_rectify',               'Instance of LeakyRectify with leakiness = 0.01'),
        ('very_leaky_rectify',          'Instance of LeakyRectify with leakiness = 1/3'),
        ('softplus',                    'Softplus activation function'),
        ('linear',                      'Linear activation function'),
        ('identity',                    'Linear activation function'),
    )
    network             = models.ForeignKey(Network, blank=True, null=True, on_delete=models.SET_NULL)
    level               = models.IntegerField(default=0)
    base_type           = models.CharField(max_length=200, choices=LAYER_TYPES, default="InputLayer")
    w                   = models.ForeignKey(Weight, related_name='w', blank=True, null=True, on_delete=models.SET_NULL)
    b                   = models.ForeignKey(Weight, related_name='b', blank=True, null=True, on_delete=models.SET_NULL)
    shape_channels      = models.IntegerField(default=1)
    shape_x             = models.IntegerField(default=1)
    shape_y             = models.IntegerField(default=1)
    p                   = models.FloatField(default=0)
    rescale             = models.BooleanField(default=False)
    num_units           = models.IntegerField(default=0)
    nonlinearity        = models.CharField(max_length=200, choices=NONLINEARITIES, default="rectify")
    #nonlinearity_alpha = models.FloatField(default=0)
    #nonlinearity_beta  = models.FloatField(default=0)
    num_filters         = models.IntegerField(default=0)
    filter_size_x       = models.IntegerField(default=1)
    filter_size_y       = models.IntegerField(default=1)
    stride_x            = models.IntegerField(default=1)
    stride_y            = models.IntegerField(default=1)
    pad_x               = models.IntegerField(default=0)
    pad_y               = models.IntegerField(default=0)
    untie_biases        = models.BooleanField(default=False)
    pool_size_x         = models.IntegerField(default=0)
    pool_size_y         = models.IntegerField(default=0)
    ignore_border       = models.BooleanField(default=True)
    k                   = models.FloatField(default=2)
    alpha               = models.FloatField(default=0.0001)
    beta                = models.FloatField(default=0.75)
    n                   = models.IntegerField(default=5)
    def __str__(self):
        return self.base_type + ' (' + str(self.level) + ')'
    
class Dataset(models.Model):
    DATA_VAR_TYPES = (
        ('uint8',                       'integer BYTE'),
        ('uint16',                      'integer WORD'),
        ('uint32',                      'integer DWORD'),
    )
    DIV_VAR_TYPES = (
        ('float32',                     'float DWORD'),
    )
    name                = models.CharField(max_length=200, default='')
    pickle              = models.BooleanField(default=False)
    image               = models.BooleanField(default=False)
    grayscale           = models.BooleanField(default=False)
    source              = models.CharField(max_length=200, default='')
    filename            = models.CharField(max_length=200, default='')
    zipped              = models.BooleanField(default=False)
    data_type           = models.CharField(max_length=200, choices=DATA_VAR_TYPES, default="uint8")
    offset              = models.IntegerField(default=0)
    split               = models.BooleanField(default=False)
    split_take_first    = models.BooleanField(default=False)
    split_size_first    = models.IntegerField(default=0)
    split_size_second   = models.IntegerField(default=0)
    reshape             = models.BooleanField(default=False)
    shape_channels      = models.IntegerField(default=1)
    shape_x             = models.IntegerField(default=1)
    shape_y             = models.IntegerField(default=1)
    divide              = models.BooleanField(default=False)
    divide_type         = models.CharField(max_length=200, choices=DIV_VAR_TYPES, default="float32")
    divide_num          = models.IntegerField(default=256)
    trim                = models.CharField(max_length=200, default='', blank=True)
    def __str__(self):
        return self.name
        
class Result(models.Model):
    ACTION_TYPES = (
        ('training',        'Training'),
        ('validation',      'Validation'),
        ('prediction',      'Prediction'),
    )
    action              = models.CharField(max_length=200, choices=ACTION_TYPES, default="validation")
    loaded              = models.BooleanField(default=False)
    error               = models.FloatField(default=0)
    accuracy            = models.FloatField(default=0)
    classification      = models.TextField(default="")
    updated_at          = models.DateTimeField()
    def __str__(self):
        return str(self.id) + ' (' + self.action + ')'