# coding:utf-8

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

np.random.seed(2018)

trained_model_root_dir = './model/output/mul/'

nb_fc_hidden_layer = 1024
nb_classes = 120
top_epoch = 50
ft_epoch = 100

learning_rate_finetune = 0.0001
momentum_finetune = 0.9

h5path = '/home/fenglf/PycharmProjects/keras-finetuning-master/gap_feature/'

trained_model_path = os.path.join(trained_model_root_dir, '13_qdropori_nobn_1024.h5')
trained_model_best_path = os.path.join(trained_model_root_dir, '13_qdropori_nobn_1024_best.h5')
# trained_model_json_path = os.path.join(trained_model_root_dir, 'model_13_qdropori_1024.json')

plot_switch = False

bottleneck_feature_train_list = ["gap_InceptionResNetV2_train.h5",
                                 "gap_DenseNet121_train.h5",
                                 "gap_DenseNet169_train.h5",
                                 "gap_DenseNet161_train.h5",
                                 "gap_DenseNet201_train.h5",
                                 "gap_Xception_train.h5",
                                 "gap_ResNet50_train.h5",
                                 "gap_ResNet101_train.h5",
                                 "gap_ResNet152_train.h5",
                                 "gap_InceptionV3_train.h5",
                                 "gap_InceptionV4_train.h5",
                                 "gap_VGG16_train.h5",
                                 "gap_VGG19_train.h5"]


def plot_training(history):
    acc = history.history['acc']
    # val_acc = history.history['val_acc']
    loss = history.history['loss']
    # val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    # plt.plot(epochs, val_acc, 'r')
    plt.title('Training accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    plt.title('Training loss')
    plt.show()


def add_new_last_layer(input_tensor, nb_classes):
    """Add last layer to the convnet
    Args:
      base_model: keras model excluding top
      nb_classes: # of classes
    Returns:
      new keras model with last layer
    """
    # x = Dropout(0.5)(input_tensor)
    # let's add a fully-connected layer, random init
    x = Dense(nb_fc_hidden_layer)(input_tensor)

    # add BN layer and Dropout flf
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #x = Dense(1024)(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    # x = Dropout(0.3)(x)
    #x = Dense(256)(x)
    #x = Activation('relu')(x)
    # and a logistic layer -- we have 120 classes
    predictions = Dense(nb_classes, activation='softmax')(x)

    model = Model(input_tensor, predictions)
    return model


def gen_train_samples():
    X_train = []
    y_train = []
    X_test = []
    # for filename in ["gap_InceptionResNetV2.h5", "gap_InceptionV3.h5"]:
    for filename in bottleneck_feature_train_list:
        with h5py.File(h5path + filename, 'r') as h:
            input_t = np.array(h['train'])
            print input_t.shape
            X_train.append(np.array(h['train']))
            y_train = np.array(h['label'])
            # print len(X_train)
    # 拼接特征
    X_train = np.concatenate(X_train, axis=1)

    # 需要打乱，否则验证集会出错
    X_train, y_train = shuffle(X_train, y_train)
    # print y_train
    # 对于多分类，需要将标签转化为one-hot-encoding
    y_train = np_utils.to_categorical(y_train, nb_classes)
    return X_train, y_train


def get_model(X_train):
    # setup and compile model
    if os.path.exists(trained_model_best_path):
        # load json and create model
        # json_file = open(trained_model_json_path, 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # model = model_from_json(loaded_model_json)
        model = load_model(trained_model_best_path)
        # load weights into new model
        model.load_weights(trained_model_best_path)
        print("Loaded model from {}".format(trained_model_best_path))

        model.compile(optimizer=SGD(lr=learning_rate_finetune, momentum=momentum_finetune),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        nb_epoch = ft_epoch

        #model.compile(optimizer='adadelta',
        #             loss='categorical_crossentropy',
        #             metrics=['accuracy'])
    else:
        input_tensor = Input(X_train.shape[1:])
        model = add_new_last_layer(input_tensor, nb_classes)

        model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=momentum_finetune),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        #model.compile(optimizer='adadelta',
        #              loss='categorical_crossentropy',
        #              metrics=['accuracy'])
        nb_epoch = top_epoch
        # model.compile(optimizer=SGD(lr=learning_rate_finetune, momentum=momentum_finetune), loss='categorical_crossentropy',
        #               metrics=['accuracy'])

        # from IPython.display import SVG
        # from keras.utils.vis_utils import model_to_dot, plot_model
        # SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

    return model, nb_epoch


def gap_train():
    # 训练数据
    X_train, y_train = gen_train_samples()

    # 构建模型
    model, nb_epoch = get_model(X_train)

    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='acc', patience=8)

    # Save the model after every epoch.
    mc_fit = ModelCheckpoint(
        trained_model_best_path,
        monitor='acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)

    # train model
    history_ft = model.fit(X_train,
                           y_train,
                           batch_size=32,
                           epochs=nb_epoch,
                           verbose=2,
                           callbacks=[mc_fit, early_stopping])
    #history_ft = model.fit(X_train, y_train, batch_size=32, epochs=nb_epoch, verbose=2,
    #                       callbacks=[mc_fit])
    model.save(trained_model_path)
    # plot_model(model, to_file='gap_model.png')
    #
    # model_json = model.to_json()
    # with open(trained_model_json_path, "w") as json_file:
    #     json_file.write(model_json)

    if plot_switch:
        plot_training(history_ft)


if __name__ == '__main__':
    gap_train()