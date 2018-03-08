import json
import numpy as np
import os
import glob
import argparse
import matplotlib.pyplot as plt

from nets.inception_v3 import InceptionV3
from nets.inception_resnet_v2 import InceptionResNetV2
from nets.nasnet import *
from nets.densenet import *
from nets.resnet152 import *
from nets.resnet101 import *
from nets.resnet50 import *
from nets.inception_v4 import InceptionV4
from nets.vgg19 import *
#from nets.vgg16 import *
# from nets.NASnet import *
from nets.xception import Xception
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.optimizers import *
from nets.densenet import *
from nets.inception_v4 import InceptionV4, preprocess_input
from keras.applications import *
from keras.preprocessing.image import *


train_data_dir = '/home/fenglf/data/dog/stanford/Images/data'
validation_data_dir = '/home/fenglf/data/dog/stanford/Images/test_data'

pretrained_model_root_dir = '/home/fenglf/PycharmProjects/keras-finetuning-master/model/pretrained/'
output_model_root_dir = '/home/fenglf/PycharmProjects/keras-finetuning-master/model/outputk'

#base_model_name = VGG19
base_model_name = InceptionV3

np.random.seed(2018)

top_epochs = 0
fit_epochs = 5

batch_size = 32

nb_fc_hidden_layer = 1024
nb_classes = 120

#NB_LAYERS_TO_FREEZE = 652
NB_LAYERS_TO_FREEZE = 406

# input size
#img_width, img_height = 224, 224
img_width, img_height = 299, 299
# img_width, img_height = 331, 331

learning_rate_finetune = 0.0001
momentum_finetune = 0.9


input_included_model = ['VGG16',
                        'VGG19',
                        'ResNet50',
                        'InceptionV3',
                        'Xception',
                        'ResNet152',
                        'ResNet101',
                        'NASNetLarge',
                        'InceptionResNetV2']

# plot switch: whether to visualize the training loss and acc
plot_switch = True

# the path of pretrained model
pretrained_model_dir = os.path.join(pretrained_model_root_dir, base_model_name.__name__, base_model_name.__name__ + '_notop.h5')

# the path of fine-tuned checkpoint path
# top_layers_checkpoint_path refer to the best new added fc layer fine-tuned checkpoint path
top_layers_checkpoint_path = os.path.join(output_model_root_dir, base_model_name.__name__, 'top_layer_weights', base_model_name.__name__ + '.top.best.hdf5')
# fine_tuned_checkpoint_path refer to the best free layers of base_model fine-tuned checkpoint path
fine_tuned_checkpoint_path = os.path.join(output_model_root_dir, base_model_name.__name__, 'fine_tuned_weights', base_model_name.__name__ + '.fine_tuned.best.hdf5')
# final_weights_path refer to the best fine-tuned final weights
final_weights_path = os.path.join(output_model_root_dir, base_model_name.__name__, 'final_weights', base_model_name.__name__ + '.final_weights.hdf5')
final_weights_json_path = os.path.join(output_model_root_dir, base_model_name.__name__, 'final_weights', base_model_name.__name__ + '.final_weights.json')
final_weights_label_path = os.path.join(output_model_root_dir, base_model_name.__name__, 'final_weights', base_model_name.__name__ + '.final_weights-labels.json')


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
      base_model: keras model excluding top
      nb_classes: # of classes
    Returns:
      new keras model with last layer
    """
    # add a global spatial average pooling layer
    x = base_model.output

    if base_model_name.__name__ in input_included_model:
        #if x.shape[1] < 4:
        ax = GlobalAveragePooling2D()(x)
        #else:
            #ax = AveragePooling2D(pool_size=(2, 2))(x)
            #ax = Flatten(name='flatten')(ax)
    else:
        ax = x

    #x = Dropout(0.5)(x)
    # let's add a fully-connected layer, random init
    x = Dense(nb_fc_hidden_layer)(ax)

    #print ax.shape

    # add BN layer and Dropout flf
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dropout(0.5)(x)

    # and a logistic layer -- we have 120 classes
    predictions = Dense(nb_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    return model


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    # i.e. freeze all convolutional layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'], )


def setup_to_finetune(model):
    """Freeze the bottom NB_LAYERS_TO_FREEZE and retrain the remaining top layers.
    note: NB_LAYERS_TO_FREEZE corresponds to the top 2 inception blocks in the base_model arch
    Args:
    model: keras model
    """
    for layer in model.layers[:NB_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_LAYERS_TO_FREEZE:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate

    model.compile(optimizer=SGD(lr=learning_rate_finetune, momentum=momentum_finetune), loss='categorical_crossentropy', metrics=['accuracy'])


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


def train(lambda_func):
    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255)

    test_datagen = ImageDataGenerator(
        rescale=1. / 255)

    train_datagen = ImageDataGenerator(
        preprocessing_function=lambda_func)

    test_datagen = ImageDataGenerator(
        preprocessing_function=lambda_func)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    class_dict = train_generator.class_indices
    print class_dict

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')


    # define the input_shape
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)


    # setup model
    if base_model_name.__name__ == 'DenseNet201':
        # densenet_weights_path = densenet_weights_root + 'densenet169_weights_tf.h5'
        base_model = base_model_name(reduction=0.5, input_shape=(img_width, img_height, 3), weights=pretrained_model_dir)
    else:
        #base_model = base_model_name(weights=pretrained_model_dir, input_shape=input_shape, include_top=False)
        base_model = base_model_name(weights=pretrained_model_dir, include_top=False)
    model = add_new_last_layer(base_model, nb_classes)

    # base_model.layers.pop()
    # for i, layer in enumerate(base_model.layers):
    #     print (i, layer.name)

    if os.path.exists(top_layers_checkpoint_path):
        model.load_weights(top_layers_checkpoint_path)
        print ("Checkpoint " + top_layers_checkpoint_path + " loaded.")

    # first: train only the top layers (which were randomly initialized)
    setup_to_transfer_learn(model, base_model)

    # Save the model after every epoch.
    mc_top = ModelCheckpoint(
        top_layers_checkpoint_path,
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)
    # Save the TensorBoard logs.
    tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    # train the model on the new data for a few epochs
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames) // batch_size,
        epochs=top_epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames) // batch_size,
        callbacks=[mc_top, tb])

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from base model. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print (i, layer.name)

    if os.path.exists(fine_tuned_checkpoint_path):
        model.load_weights(fine_tuned_checkpoint_path)
        print ("Checkpoint '" + fine_tuned_checkpoint_path + "' loaded.")

    # we chose to train the some top blocks, i.e. we will freeze
    # the first NB_LAYERS_TO_FREEZE layers and unfreeze the rest:
    setup_to_finetune(model)

    # Save the model after every epoch.
    mc_fit = ModelCheckpoint(
        fine_tuned_checkpoint_path,
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    history_ft = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames) // batch_size,
        epochs=fit_epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames) // batch_size,
        callbacks=[mc_fit, tb])

    # save final weights
    model.save_weights(final_weights_path)

    # serialize model to JSON
    model_json = model.to_json()
    with open(final_weights_json_path, "w") as json_file:
        json_file.write(model_json)
    with open(final_weights_label_path, "w") as json_file:
        json.dump(class_dict, json_file)

    if plot_switch:
        plot_training(history_ft)


if __name__ == '__main__':
    train(inception_v3.preprocess_input)
