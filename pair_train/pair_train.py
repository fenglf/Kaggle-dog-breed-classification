import os

import keras
import numpy as np
import tensorflow as tf
from keras import Input
from keras import backend as K
from keras.applications import Xception
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.applications import *


train_data_dir = '/home/fenglf/data/dog/stanford/Images/data'
validation_data_dir = '/home/fenglf/data/dog/stanford/Images/test_data'

pretrained_model_root_dir = '/home/fenglf/PycharmProjects/keras-finetuning-master/model/pretrained/'
first_trained_weights = './inv3_xc_first.h5'
fine_tuned_weights = './inv3_xc_ft.h5'
best_saved_weights = '/home/fenglf/PycharmProjects/keras-finetuning-master/xcep_incep14-0.9964-0.9976_ft_best.h5'


img_width, img_height = 299, 299
batch_size = 16
nb_fc_hidden_layer = 1024
nb_classes = 120


def get_datagen(preprocess_func):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func)

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func)


    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator


def get_base_model(MODEL, input_tensor, pretrained_model_dir):
    # define input
    base_model = MODEL(input_tensor=input_tensor, weights=pretrained_model_dir, include_top=False)
    print ("Primary Checkpoint '" + pretrained_model_dir + "' loaded.")
    return base_model


def get_model_out(MODEL, image_size):
    pretrained_model_dir = os.path.join(pretrained_model_root_dir, MODEL.__name__,
                                        MODEL.__name__ + '_notop.h5')
    print(MODEL.__name__)

    # define input
    input_tensor = Input((image_size[0], image_size[1], 3))

    # setup model
    model = get_base_model(MODEL, input_tensor, pretrained_model_dir)

    # model = GlobalAveragePooling2D()(model.output)

    return model


def add_new_last_layer(feature, nb_classes, name):
    """Add last layer to the convnet
    Args:
      base_model: keras model excluding top
      nb_classes: # of classes
    Returns:
      new keras model with last layer
    """
    # add a global spatial average pooling layer

    # x = Dropout(0.5)(x)
    x = Dense(nb_fc_hidden_layer)(feature)

    # print ax.shape

    # add BN layer and Dropout flf
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    # and a logistic layer -- we have 120 classes
    predictions = Dense(nb_classes, activation='softmax', name=name)(x)

    return predictions


def pair_generator(cur_generator, batch_size, train=True):
    cur_cnt = 0

    while True:
        if train and cur_cnt % 4 == 1:
            # provide same image
            x1, y1 = train_generator.next()
            if y1.shape[0] != batch_size:
                x1, y1 = train_generator.next()
            # print(y1)
            # print(np.sort(np.argmax(y1, 1), 0))
            y1_labels = np.argmax(y1, 1)
            has_move = list()
            last_not_move = list()
            idx2 = [-1 for i in range(batch_size)]

            for i, label in enumerate(y1_labels):
                if i in has_move:
                    continue
                for j in range(i+1, batch_size):
                    if y1_labels[i] == y1_labels[j]:
                        idx2[i] = j
                        idx2[j] = i
                        has_move.append(i)
                        has_move.append(j)
                        break
                if idx2[i] == -1:
                    # same element not found and hasn't been moved
                    if len(last_not_move) == 0:
                        last_not_move.append(i)
                        idx2[i] = i
                    else:
                        idx2[i] = last_not_move[-1]
                        idx2[last_not_move[-1]] = i
                        del last_not_move[-1]
            x2 = list()
            y2 = list()
            for i2 in range(batch_size):
                x2.append(x1[idx2[i2]])
                y2.append(y1[idx2[i2]])
            # print(y2)
            x2 = np.asarray(x2)
            y2 = np.asarray(y2)
            # print(x2.shape)
            # print(y2.shape)
        else:
            x1, y1 = cur_generator.next()

            if y1.shape[0] != batch_size:
                x1, y1 = cur_generator.next()
            x2, y2 = cur_generator.next()
            if y2.shape[0] != batch_size:
                x2, y2 = cur_generator.next()
        same = (np.argmax(y1, 1) == np.argmax(y2, 1)).astype(int)
        one_hot_same = np.zeros([batch_size, 2])
        one_hot_same[np.arange(batch_size), same] = 1

        # print cur_cnt
        # print same
        # print one_hot_same
        # print(np.argmax(y1, 1))
        # print(np.argmax(y2, 1))
        # print(same)
        cur_cnt += 1
        # print cur_generator.filenames
        yield [x1, x2], [y1, y2, one_hot_same]


def eucl_dist(inputs):
    x, y = inputs
    return (x - y)**2


def first_train(train_generator, validation_generator):
    if os.path.exists(first_trained_weights):
        model = load_model(first_trained_weights)
    else:
        # create the base pre-trained model
        # input_tensor = Input(shape=(299, 299, 3))
        # base_model = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
        # plot_model(base_model, to_file='xception_model.png')
        # base_model.layers.pop()
        # base_model.outputs = [base_model.layers[-1].output]
        # base_model.layers[-1].outbound_nodes = []
        # base_model.output_layers = [base_model.layers[-1]]

        base_model1 = get_model_out(Xception, (299, 299))
        base_model2 = get_model_out(InceptionV3, (299, 299))

        # for i, layer in enumerate(base_model.layers):
        #     print (i, layer.name)

        # feature = base_model

        img1 = Input(shape=(299, 299, 3), name='img_1')
        img2 = Input(shape=(299, 299, 3), name='img_2')

        # feature3 = feature(img1)

        feature1 = GlobalAveragePooling2D()(base_model1(img1))
        feature2 = GlobalAveragePooling2D()(base_model2(img2))
        # feature2 = GlobalAveragePooling2D()(base_model1(img2))
        # let's add a fully-connected layer

        category_predict1 = add_new_last_layer(feature1, nb_classes, name='ctg_out_1')
        category_predict2 = add_new_last_layer(feature2, nb_classes, name='ctg_out_2')


        # category_predict1 = Dense(100, activation='softmax', name='ctg_out_1')(
        #     Dropout(0.5)(feature1)
        # )
        # category_predict2 = Dense(100, activation='softmax', name='ctg_out_2')(
        #     Dropout(0.5)(feature2)
        # )

        # concatenated = keras.layers.concatenate([feature1, feature2])
        dis = Lambda(eucl_dist, name='square')([feature1, feature2])
        # concatenated = Dropout(0.5)(concatenated)
        # let's add a fully-connected layer
        # x = Dense(1024, activation='relu')(concatenated)
        x = Dense(256)(dis)
        # add BN layer and Dropout
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        judge = Dense(2, activation='softmax', name='bin_out')(x)

        # judge = Dense(1, activation='sigmoid', name='bin_out')(x)
        model = Model(inputs=[img1, img2], outputs=[category_predict1, category_predict2, judge])

        for i, layer in enumerate(model.layers):
            print (i, layer.name)

        # model.save('dog_xception.h5')
        plot_model(model, to_file='model_combined_inv3_xception.png', show_shapes=True)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional layers
        for layer in base_model1.layers:
            layer.trainable = False
        for layer in base_model2.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='nadam',
                      loss={'ctg_out_1': 'categorical_crossentropy',
                            'ctg_out_2': 'categorical_crossentropy',
                            'bin_out': 'categorical_crossentropy'},
                            # 'bin_out': 'binary_crossentropy'},
                      loss_weights={
                          'ctg_out_1': 1.,
                          'ctg_out_2': 1.,
                          'bin_out': 0
                      },
                      metrics=['accuracy'])
        # model = make_parallel(model, 3)
        # train the model on the new data for a few epochs

        save_model = ModelCheckpoint('xcep_incep{epoch:01d}-{ctg_out_1_acc:.4f}-{ctg_out_2_acc:.4f}_top_best.h5',
                                     monitor='loss',
                                     save_best_only=True,
                                     mode='auto',
                                     period=1)

        model.fit_generator(pair_generator(train_generator, batch_size=batch_size),
                            # steps_per_epoch=train_generator.samples // batch_size,  # must divided exactly,
                            steps_per_epoch=train_generator.samples / batch_size+1,
                            epochs=5,
                            validation_data=pair_generator(validation_generator, train=False, batch_size=batch_size),
                            validation_steps=validation_generator.samples/batch_size+1,
                            callbacks=[early_stopping, auto_lr, save_model])
        model.save('dog_inceptionv3_xception.h5')


def fine_tune(train_generator, validation_generator, model=None):
    if os.path.exists(best_saved_weights):
        model = load_model(fine_tuned_weights)
        print 'load best-saved weights: {}'.format(best_saved_weights)
        print 'continue fine tuning...'
    elif os.path.exists(fine_tuned_weights):
        model = load_model(fine_tuned_weights)
        print 'load fine-tuned weights: {}'.format(fine_tuned_weights)
        print 'continue fine tuning...'

    for i, layer in enumerate(model.layers):
        print (i, layer.name)

    xception_model = model.layers[2]
    for layer in xception_model.layers[:126]:
        layer.trainable = False
    for layer in xception_model.layers[126:]:
        layer.trainable = True

    inception_model = model.layers[3]
    for layer1 in inception_model.layers[:295]:
        layer1.trainable = False
    for layer1 in inception_model.layers[295:]:
        layer1.trainable = True


        # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  loss={'ctg_out_1': 'categorical_crossentropy',
                        'ctg_out_2': 'categorical_crossentropy',
                        'bin_out': 'categorical_crossentropy'},
                  loss_weights={
                      'ctg_out_1': 1.,
                      'ctg_out_2': 1.,
                      'bin_out': 0.5
                  },
                  metrics=['accuracy'])
    # model = make_parallel(model, 3)
    # train the model on the new data for a few epochs

    save_model = ModelCheckpoint('xcep_incep{epoch:01d}-{ctg_out_1_acc:.4f}-{ctg_out_2_acc:.4f}_ft_best.h5',
                                 monitor='loss',
                                 save_best_only=True,
                                 mode='auto',
                                 period=1)

    model.fit_generator(pair_generator(train_generator, batch_size=batch_size),
                        steps_per_epoch=train_generator.samples / batch_size + 1,
                        epochs=20,
                        validation_data=pair_generator(validation_generator, train=False, batch_size=batch_size),
                        validation_steps=validation_generator.samples / batch_size + 1,
                        callbacks=[early_stopping, auto_lr, save_model])

    model.save(fine_tuned_weights)


if __name__ == '__main__':
    train_generator, validation_generator = get_datagen(xception.preprocess_input)

    '''
       def lr_decay(epoch):
           lrs = [0.0001, 0.0001, 0.0001,0.0001,0.00001, 0.000001, 0.000001, 0.00001, 0.000001,
                  0.000001, 0.000001, 0.000001,
                  0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]

           return lrs[epoch]
       '''
    early_stopping = EarlyStopping(monitor='loss', patience=8)
    # my_lr = LearningRateScheduler(lr_decay)
    auto_lr = ReduceLROnPlateau(monitor='loss',
                                factor=0.1,
                                patience=3,
                                verbose=0,
                                mode='auto',
                                epsilon=0.0001,
                                cooldown=0,
                                min_lr=0)

    first_train(train_generator, validation_generator)

    # model = load_model(first_trained_weights)
    # first_trained_model = load_model('/home/fenglf/PycharmProjects/keras-finetuning-master/xcep_incep19-0.9850-0.9902.h5')
    print 'first_trained_model loaded.'
    print 'start fine tune...'

    # fine_tune(train_generator, validation_generator, first_trained_model)
    fine_tune(train_generator, validation_generator)
