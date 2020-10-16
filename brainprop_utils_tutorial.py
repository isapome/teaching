
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import datasets, layers, losses, metrics, optimizers, callbacks, models, Input
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

def import_from_path(module_name, file_path=None):
    """Import the other python files as modules
    
    Keyword arguments:
    module_name -- the name of the python file (with extension, if file_path is None)
    file_path -- path to the file if not in the current directory (default: None)
    """
    if not file_path:
        if module_name.endswith('.py'):
            file_path = module_name
        else:
            file_path = module_name + '.py'
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return importlib.import_module(module_name)

def get_filename(type, dataset, learning_algorithm):  #function that prevents file override
    """Computes the filename for the outputs of the training 
    (checks whether the file already exists, in that case adds a number to the filename 
    to avoid overriding it)
    
    Keyword arguments:
    type -- string, the type of output (accuracy.png, history.pkl or weights.h5)
    """
    filename = "{}_{}_{}".format(dataset, learning_algorithm, type)
    num = 0
    while os.path.isfile(filename):
        filename="{}_{}_{}_{}".format(dataset, learning_algorithm, num, type)
        num += 1
    return filename



def train_model(learning_algorithm, dataset, hidden_layers, batch_dim, learning_rate, seed):
    """ function that trains a neural network with tf.keras with automatic differentiation.
    
    Keyword arguments:
    learning_algorithm -- either 'EBP' for error backpropagation (with softmax and cross-entropy loss) or 'BrainProp'
    dataset -- either 'MNIST', 'CIFAR10' or 'CIFAR100'
    hidden_layers -- list of layers for the network (accepts 'Dense(n)', 'Conv2D(n_filters, (ksize_x,ksize_y)' and any other layer with full input)
    batch_dim -- minibatch size
    learning_rate -- learning rate used for training
    seed -- integer, seed used for reproducible results
    """

    save_plots = True

    print("Experiment begins, training on {} with {}".format(dataset, learning_algorithm))

    np.random.seed(seed)
    tf.random.set_seed(seed)


    if dataset == 'MNIST':
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        if len(np.shape(train_images)) < 4:
            train_images = tf.expand_dims(train_images, -1).numpy()
            test_images = tf.expand_dims(test_images, -1).numpy()
    elif dataset == 'CIFAR10':
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    elif dataset == 'CIFAR100':
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode='fine')
    else:
        raise Exception("Unknown dataset. Choose either \'MNIST\', \'CIFAR10\' or \'CIFAR100\'.")

    if tf.reduce_max(train_images) >1:
        train_images = train_images / 255.0
    if tf.reduce_max(test_images) >1:
        test_images = test_images / 255.0


    image_shape = np.shape(train_images)[1:]
    n_classes = tf.cast(tf.reduce_max(train_labels)+1, dtype=tf.int32)
    n_batches = len(train_images)//batch_dim

    train_labels = tf.keras.utils.to_categorical(train_labels, n_classes, dtype='float32')
    test_labels = tf.keras.utils.to_categorical(test_labels, n_classes, dtype='float32')

    #preparing architecture and optimizer depending on the selected learning algorithm
    if learning_algorithm == 'EBP':
        output_activation_function = 'softmax'
        loss = 'categorical_crossentropy'
        metric = 'accuracy'
        output_layer = layers.Dense
    elif learning_algorithm == 'BrainProp':
        output_activation_function = 'linear'
        metric = 'accuracy'
        brainprop = import_from_path('brainprop', file_path="brainprop.py")
        loss = brainprop.BrainPropLoss(batch_size=batch_dim, n_classes=n_classes, replicas=1)
        output_layer = brainprop.BrainPropLayer
#         if os.path.exists('brainprop.py') != True:
#           ! wget https://github.com/isapome/BrainProp/raw/master/brainprop.py
#         from brainprop import BrainPropLayer, BrainPropLoss
#         loss = BrainPropLoss(batch_size=batch_dim, n_classes=n_classes, replicas=1)
#         output_layer = BrainPropLayer
    else:
        raise Exception("Unknown learning algorithm. Choose between \'EBP\' and \'BrainProp\' ")

    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.)


    bias = False
    initializer = tf.random_normal_initializer(mean=0., stddev=0.01)
    regularizer = None
    pad = 'same'


    model = models.Sequential()
    model.add(Input(shape=image_shape)) #input_shape=image_shape
    
    flatten_layer = 0 #there needs to be a flatten layer between 4dim inputs and dense layers. 

    for hidden_layer in hidden_layers:

        if hidden_layer.__class__.__name__ == 'Dense' and flatten_layer <1:  
            model.add(layers.Flatten())
            flatten_layer += 1
        
        if hidden_layer.__class__.__name__ == 'Conv2D' and flatten_layer >0:  
            raise Exception("Please do not add convolutional layers after dense layers.")

        config = hidden_layer.get_config()
        layer = layers.deserialize({'class_name': hidden_layer.__class__.__name__, 'config': config})
        layer.use_bias=bias
        layer.kernel_initializer=initializer
        layer.kernel_regularizer=regularizer
        if hidden_layer.__class__.__name__ == 'Conv2D':
          layer.padding=pad
        model.add(layer)

    last_layer = output_layer(n_classes, activation=output_activation_function, use_bias=bias, kernel_regularizer=regularizer, kernel_initializer=initializer)
    model.add(last_layer)
    model.summary()

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    epochs = 500 #just as upper bound. Early stopping will act much earlier than this.

    lr_schedule = None
    terminate_on_NaN = callbacks.TerminateOnNaN()
    earlystopping = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10, verbose=1, mode='max', baseline=None, restore_best_weights=False)
    callbacks_list = list(filter(None, [lr_schedule, terminate_on_NaN, earlystopping]))

    tic_training = datetime.datetime.now()
    history = model.fit(train_images, train_labels, batch_size=batch_dim, epochs=epochs, validation_data=(test_images, test_labels), shuffle=True, verbose=2, callbacks=callbacks_list)

    toc_training = datetime.datetime.now()
    elapsed = (toc_training - tic_training).seconds//60
    print("Training, elapsed: {} minute{}.".format(elapsed, 's' if elapsed>1 else ''))


    if save_plots == True: #save a plot of the accuracy as a function of the epochs
        filename_plot = get_filename('accuracy.png', dataset, learning_algorithm)

        n_epochs = len(history.history['accuracy'])

        plt.figure()
        plt.title("{} - {}".format(learning_algorithm, dataset) , fontsize=16)
        plt.plot(history.history['accuracy'], label='accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label = 'validation accuracy', linewidth=2)
        maximum_val_accuracy = np.max(history.history['val_accuracy'])
        argmax_val_accuracy = np.argmax(history.history['val_accuracy'])
        plt.plot([argmax_val_accuracy,argmax_val_accuracy], [-0.4,maximum_val_accuracy], '--', color='green', linewidth=1)
        plt.plot(argmax_val_accuracy,maximum_val_accuracy,'ks', markersize = 7, label='maximum = {:.5}'.format(maximum_val_accuracy))
        plt.xticks(list(plt.xticks()[0]) + [argmax_val_accuracy])
        plt.gca().get_xticklabels()[-1].set_color("white")
        plt.gca().get_xticklabels()[-1].set_fontweight('bold')
        plt.gca().get_xticklabels()[-1].set_bbox(dict(facecolor='green', edgecolor='white', alpha=0.8))
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlim([-0.4, (n_epochs-.5)])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower right', fontsize=12)
        print("Saving the accuracy plot as \'{}\'".format(filename_plot))
        plt.savefig(filename_plot, dpi=300, bbox_inches='tight')
