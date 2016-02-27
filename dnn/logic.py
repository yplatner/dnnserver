import time
import sys
import os
import time
import numpy as np
import scipy as sp
import theano
import theano.tensor as T
import lasagne
import gzip
import dill as pickle
from django.utils import timezone
from urllib import urlretrieve
from models import User, Update, Network, Weight, Layer, Dataset, Result

from django.http import HttpResponse

# Convert an object to a storable string using dill
def obj_to_str(obj):
    obj_str = repr(pickle.dumps(obj))
    return obj_str

# Convert a stored string back to an object using dill    
def str_to_obj(obj_str):
    obj = pickle.loads(eval(obj_str))
    return obj

# Convert an image (color or grayscale) to a numpy array
def img_to_arr(img, gs):
    arr = sp.misc.imread(img, gs)
    return arr
    
# Helper function iterating over training data in mini-batches of a particular size, optionally in random order. 
# It assumes data is available as numpy arrays.
def iterate(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# Create weight initialization command string
def gen_init(weight):
    if weight.init_gain_relu == True :
        gain = "gain='relu'"
    else :
        gain = 'gain=' + str(weight.init_gain)
    if weight.init == 'Constant':       
        parameters = 'val=' + str(weight.init_val)
    elif weight.init == 'Normal':    
        parameters = 'std=' + str(weight.init_std) + ', ' + 'mean=' + str(weight.init_mean)
    elif weight.init == 'Uniform':      
        parameters = 'range=' + str(weight.init_range) + ', ' + 'std=' + str(weight.init_std) + ', ' + 'mean=' + str(weight.init_mean)
    elif weight.init == 'GlorotNormal':
        parameters = gain + ', ' + 'c01b=' + str(weight.init_c01b)
    elif weight.init == 'GlorotUniform':  
        parameters = gain + ', ' + 'c01b=' + str(weight.init_c01b)
    elif weight.init == 'HeNormal':
        parameters = gain + ', ' + 'c01b=' + str(weight.init_c01b)
    elif weight.init == 'HeUniform':      
        parameters = gain + ', ' + 'c01b=' + str(weight.init_c01b)
    elif weight.init == 'Orthogonal':
        parameters = gain
    elif weight.init == 'Sparse':     
        parameters = 'sparsity=' + str(weight.init_sparsity) + ', ' + 'std=' + str(weight.init_std)
    init_str = 'lasagne.init.' + weight.init + '(' + parameters + ')'
    return init_str
    
# Create loss objective command string
def gen_loss(network):
    if network.loss_objective == 'binary_hinge_loss':       
        parameters = ', True, ' + str(network.loss_delta)
    elif network.loss_objective == 'multiclass_hinge_loss':       
        parameters = ', ' + str(network.loss_delta)
    else:
        parameters = ''
    loss_str = 'lasagne.objectives.' + network.loss_objective + '(prediction, target_var' + parameters + ')'
    return loss_str
    
# Create update strategy command string
def gen_update(network):
    if network.update.strategy == 'sgd':       
        parameters = 'learning_rate=network.update.learning_rate'
    elif network.update.strategy == 'momentum':       
        parameters = 'learning_rate=network.update.learning_rate, momentum=network.update.momentum'
    elif network.update.strategy == 'nesterov_momentum':       
        parameters = 'learning_rate=' + str(network.update.learning_rate) + ', momentum=' + str(network.update.momentum)
    elif network.update.strategy == 'adagrad':       
        parameters = 'learning_rate=network.update.learning_rate, epsilon=network.update.epsilon'
    elif network.update.strategy == 'rmsprop':       
        parameters = 'learning_rate=network.update.learning_rate, rho=network.update.rho, epsilon=network.update.epsilon'
    elif network.update.strategy == 'adadelta':       
        parameters = 'learning_rate=network.update.learning_rate, rho=network.update.rho, epsilon=network.update.epsilon'
    elif network.update.strategy == 'adam':       
        parameters = 'learning_rate=network.update.learning_rate, beta1=network.update.beta1, beta2=network.update.beta2, epsilon=network.update.epsilon'
    update_str = 'lasagne.updates.' + network.update.strategy + '(loss, params, ' + parameters + ')'
    return update_str 

# Create network object from DB values
def build(network, input_var=None):
    layers = network.layer_set.all()
    for i in range(layers.count()):
        layer = layers.get(level=i)
        if layer.base_type == 'InputLayer':       
            result = lasagne.layers.InputLayer(shape=(None, layer.shape_channels, layer.shape_x, layer.shape_y), input_var=input_var)
        elif layer.base_type == 'DenseLayer':
            if layer.nonlinearity == 'None':
                result = lasagne.layers.DenseLayer(result, num_units=layer.num_units, W=eval(gen_init(layer.w)), b=eval(gen_init(layer.b)), nonlinearity=None)
            else:
                result = lasagne.layers.DenseLayer(result, num_units=layer.num_units, W=eval(gen_init(layer.w)), b=eval(gen_init(layer.b)), nonlinearity=eval('lasagne.nonlinearities.' + layer.nonlinearity))
        elif layer.base_type == 'Conv2DLayer':    
            if layer.nonlinearity == 'None':
                result = lasagne.layers.Conv2DLayer(result, num_filters=layer.num_filters, filter_size=(layer.filter_size_x, layer.filter_size_y), stride=(layer.stride_x, layer.stride_y), pad=(layer.pad_x, layer.pad_y), untie_biases=layer.untie_biases, W=eval(gen_init(layer.w)), b=eval(gen_init(layer.b)), nonlinearity=None, convolution=theano.tensor.nnet.conv2d)
            else:
                result = lasagne.layers.Conv2DLayer(result, num_filters=layer.num_filters, filter_size=(layer.filter_size_x, layer.filter_size_y), stride=(layer.stride_x, layer.stride_y), pad=(layer.pad_x, layer.pad_y), untie_biases=layer.untie_biases, W=eval(gen_init(layer.w)), b=eval(gen_init(layer.b)), nonlinearity=eval('lasagne.nonlinearities.' + layer.nonlinearity), convolution=theano.tensor.nnet.conv2d)
        elif layer.base_type == 'MaxPool2DLayer':    
            result = lasagne.layers.MaxPool2DLayer(result, pool_size=(layer.pool_size_x, layer.pool_size_y), stride=(layer.stride_x, layer.stride_y), pad=(layer.pad_x, layer.pad_y), ignore_border=layer.ignore_border)
        elif layer.base_type == 'DropoutLayer':    
            result = lasagne.layers.DropoutLayer(result, p=layer.p)
        elif layer.base_type == 'LocalResponseNormalization2DLayer':
            print("no normalization")
            result = lasagne.layers.LocalResponseNormalization2DLayer(result, alpha=layer.alpha, k=layer.k, beta=layer.beta, n=layer.n)
        elif layer.base_type == 'NonlinearityLayer':    
            if layer.nonlinearity != 'None':
                result = lasagne.layers.NonlinearityLayer(result, nonlinearity=eval('lasagne.nonlinearities.' + layer.nonlinearity))
    if network.generated:
        print("\tloading weights from file")
        with np.load('network' + str(network.id) + '.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(result, param_values)
    else:
        print("\tsaving weights to file")
        np.savez('network' + str(network.id) + '.npz', *lasagne.layers.get_all_param_values(result))
        network.generated = True
        network.save()
    return result

# Loads data from a remote binary file and adapts it
def load(dataset):
    #create_cifar10()
    if not os.path.exists(dataset.filename):
        urlretrieve(dataset.source + dataset.filename, dataset.filename)
        print("retrieved file " + dataset.filename)
    else:
        print("opening existing file " + dataset.filename)
    if dataset.zipped:
        f = gzip.open(dataset.filename, 'rb')
    else:
        f = open(dataset.filename, 'rb')
    if (dataset.image):
        value = img_to_arr(f, dataset.grayscale)
        value = np.swapaxes(value,0,2)
        value = np.expand_dims(value, axis=0)
    if dataset.pickle:
        value = pickle.load(f)
    else:
        value = np.frombuffer(f.read(), eval('np.' + dataset.data_type), offset=dataset.offset)
    if dataset.split:
        take_first = dataset.split_take_first
        size_first = dataset.split_size_first
        size_second = dataset.split_size_second
        value = np.split(value, size_first + size_second)
        #print("Array shape before split:")
        #print(np.shape(value))
        def split_func(data):
            if take_first:
                #return data[0:(size_first-1)] ## FIX THIS
                return data[0];
            else:
                return data[size_first:(size_first+size_second)]
        value = np.apply_along_axis(split_func, axis=0, arr=value)
        #print(np.shape(value))
        value = np.hstack(value)
        #print(np.shape(value))
    if dataset.reshape:
        value = eval('value.reshape(-1, ' + str(dataset.shape_channels) + ', ' + str(dataset.shape_x) + ', ' + str(dataset.shape_y) + ')')
    if dataset.divide:
        value = value / eval('np.' + str(dataset.divide_type) + '(' + str(dataset.divide_num) + ')')
    value = eval('value' + dataset.trim)
    f.close()
    return value
    
# train with input(s) and target(s), and return loss value
# inputs is T.tensor4, targets is T.ivector
def train(network_id, result_id, inputs_dataset_id, targets_dataset_id, batch_size, epochs):
    print("starting training...")
    print(time.ctime())
    network = Network.objects.get(pk=network_id)
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    generated_network = build(network, input_var)
    print("\tnetwork was generated")
    prediction = lasagne.layers.get_output(generated_network)
    loss = eval(gen_loss(network))
    loss = loss.mean()
    #if (network.regularization):
    #    penalty = (lasagne.regularization.regularize_layer_params(generated_network, eval('lasagne.regularization.' + network.penalty)) * network.coefficient);
    #    loss = loss + penalty;
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(generated_network, trainable=True)
    updates = eval(gen_update(network))
    train_fn = theano.function([input_var, target_var], [loss, acc], updates=updates)
    print("\ttraining function was created")
    inputs = load(Dataset.objects.get(pk=inputs_dataset_id))
    print(np.shape(inputs))
    print(inputs)
    print("\tloaded inputs")
    targets = load(Dataset.objects.get(pk=targets_dataset_id))
    print(np.shape(targets))
    print(targets)
    print("\tdata was loaded")
    train_err = 0
    train_acc = 0
    train_batches = 0
    for num in range(0,epochs):
        print("### EPOCH " + str(num) + " ###")
        for batch in iterate(inputs, targets, batch_size, shuffle=True):
            batch_inputs, batch_targets = batch
            err, acc = train_fn(batch_inputs, batch_targets)
            print("\tbatch " + str(train_batches) + ': training error is ' + str(err) + ', accuracy is ' + str(acc))
            print("\t" + time.ctime())
            train_err += err
            train_acc += acc
            train_batches += 1
    res_err = train_err / train_batches
    res_acc = (train_acc / train_batches) * 100
    print("\ttraining done - error:\t\t{:.6f}".format(res_err))
    print(time.ctime())
    result = Result.objects.get(pk=result_id)
    result.error = res_err
    result.accuracy = res_acc
    result.updated_at = timezone.now()
    result.loaded = True
    result.save()
    print("\tresult saved")
    np.savez('network' + str(network_id) + '.npz', *lasagne.layers.get_all_param_values(generated_network))
    network.generated = True
    network.save()
    print("\tsaved training weights to file")
    return True
    
# validate with input(s) and target(s), and return loss and accuracy values
# inputs is T.tensor4, targets is T.ivector
def validate(network_id, result_id, inputs_dataset_id, targets_dataset_id, batch_size):
    print("starting validation...")
    network = Network.objects.get(pk=network_id)
    inputs = load(Dataset.objects.get(pk=inputs_dataset_id))
    targets = load(Dataset.objects.get(pk=targets_dataset_id))
    print("\tdata was loaded")
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    generated_network = build(network, input_var)
    print("\tnetwork was generated")
    prediction = lasagne.layers.get_output(generated_network, deterministic=True)
    loss = eval(gen_loss(network))
    loss = loss.mean()
    #if (network.regularization):
    #    penalty = (regularize_layer_params(generated_network, eval('lasagne.regularization.' + network.regularization_penalty)) * network.regularization_coefficient);
    #    loss = loss + penalty;
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
    val_fn = theano.function([input_var, target_var], [loss, acc])
    print("\tvalidation function was created")
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate(inputs, targets, batch_size, shuffle=False):
        batch_inputs, batch_targets = batch
        err, acc = val_fn(batch_inputs, batch_targets)
        print("\tbatch " + str(val_batches) + ': validation error is ' + str(err) + ', accuracy is ' + str(acc))
        val_err += err
        val_acc += acc
        val_batches += 1
    res_err = val_err / val_batches
    res_acc = (val_acc / val_batches) * 100
    print("\tvalidation done -")
    print("\t\terror:\t\t{:.6f}".format(res_err))
    print("\t\taccuracy:\t\t{:.2f} %".format(res_acc))
    result = Result.objects.get(pk=result_id)
    result.error = res_err
    result.accuracy = res_acc
    result.updated_at = timezone.now()
    result.loaded = True
    result.save()
    print("\tresult saved")
    return True

# get prediction for input
# inputs is T.tensor4, res is T.ivector
def predict(network_id, result_id, inputs_dataset_id):
    print("starting prediction...")
    network = Network.objects.get(pk=network_id)
    inputs = load(Dataset.objects.get(pk=inputs_dataset_id))
    print("\tdata was loaded")
    input_var = T.tensor4('inputs')
    generated_network = build(network, input_var)
    print("\tnetwork was generated")
    prediction = lasagne.layers.get_output(generated_network)
    classification = T.argmax(prediction, axis=1)
    #predict_fn = theano.function([input_var], prediction)
    predict_fn = theano.function([input_var], classification)
    print("\tprediction function was created")
    res_pred = predict_fn(inputs)
    pred_str = ', '.join(str(e) for e in res_pred)
    print("\tclassification:")
    print(pred_str)
    result = Result.objects.get(pk=result_id)
    result.classification = pred_str
    result.updated_at = timezone.now()
    result.loaded = True
    result.save()
    print("\tresult saved")
    return True
    
# reset network weights
def reset(network):
    network.generated = False
    network.save()
    return network

# Create datasets for CIFAR10
def create_cifar10():
    # Load training data
    X, y = [], []
    for num in range(1,5):
        f = open('data_batch_' + str(num), 'rb')
        batch = pickle.load(f)
        X.append(batch['data'])
        y.append(batch['labels'])
    X = np.concatenate(X).reshape(-1, 3, 32, 32).astype(np.float32)
    y = np.concatenate(y).astype(np.int32)
    # Load test data
    f = open('test_batch', 'rb')
    batch = pickle.load(f)
    X_test = batch['data'].reshape(-1, 3, 32, 32).astype(np.float32)
    y_test = np.array(batch['labels'], dtype=np.int32)
    # Split arrays
    ii = np.random.permutation(len(X))
    X_train = X[ii[1000:]]
    y_train = y[ii[1000:]]
    X_val = X[ii[:1000]]
    y_val = y[ii[:1000]]
    # Offset data
    offset = np.mean(X_train, 0)
    scale = np.std(X_train, 0).clip(min=1)
    X_train = (X_train - offset) / scale
    X_val = (X_val - offset) / scale
    X_test = (X_test - offset) / scale
    # Save data
    pickle.dump(X_train, open('cifar10_X_train', "wb"))
    pickle.dump(y_train, open('cifar10_y_train', "wb"))
    pickle.dump(X_val, open('cifar10_X_val', "wb"))
    pickle.dump(y_val, open('cifar10_y_val', "wb"))
    pickle.dump(X_test, open('cifar10_X_test', "wb"))
    pickle.dump(y_test, open('cifar10_y_test', "wb"))
    return True
