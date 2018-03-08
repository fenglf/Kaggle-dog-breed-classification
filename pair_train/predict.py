# coding:utf-8
from keras.applications import *
from nets.inception_v3 import InceptionV3
from nets.inception_resnet_v2 import InceptionResNetV2
from nets.nasnet import NASNet
from nets.xception import Xception
from nets.inception_v4 import InceptionV4
from keras.preprocessing import image
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
from pair_train import pair_generator
# from train import add_new_last_layer

import numpy as np
import os

test_data_dir = '/home/fenglf/data/dog/kaggle/test'

pretrained_model_root_dir = '/home/fenglf/PycharmProjects/keras-finetuning-master/model/pretrained/'
output_model_root_dir = '/home/fenglf/PycharmProjects/keras-finetuning-master/model/output'

csv_sample_path = './predict_csv/sample_submission.csv'
csv_out_path1 = './predict_csv/inv3_xc_pair_pair1.csv'
csv_out_path2 = './predict_csv/inv3_xc_pair_pair2.csv'
csv_out_path = './predict_csv/inv3_pair_9960_9980.csv'

base_model_name = InceptionV4
lambda_func = inception_v3.preprocess_input

batch_size = 16

final_weights_path = os.path.join(output_model_root_dir, base_model_name.__name__, 'final_weights', base_model_name.__name__ + '.final_weights.hdf5')
ft_best_weights_path = os.path.join(output_model_root_dir, base_model_name.__name__, 'fine_tuned_weights', base_model_name.__name__ + '.fine_tuned.best.hdf5')
final_weights_json_path = os.path.join(output_model_root_dir, base_model_name.__name__, 'final_weights', base_model_name.__name__ + '.final_weights.json')

np.random.seed(2018)

pair_model_best = '/home/fenglf/PycharmProjects/keras-finetuning-master/xcep_incep2-0.9960-0.9980_ft_best.h5'


def gen_test_gen(image_size, preprocess_func):
    print "test_generator creating..."
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func)
    # 注意：使用此方法时，test_data_dir必须有子文件夹
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(image_size[0], image_size[1]),
        batch_size=batch_size,
        shuffle=False,
        # class_mode=None)
        class_mode='categorical')
    return test_generator


def predict(single_model, test_generator):
    if os.path.exists(single_model):
        # # load json and create model
        # json_file = open(final_weights_json_path, 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # model = model_from_json(loaded_model_json)
        # # load weights into new model
        # model.load_weights(ft_best_weights_path)
        model = load_model(single_model)
        print ("Checkpoint " + single_model + " loaded.")
    # print test_generator.filenames
    predictions = model.predict_generator(
        test_generator,
        steps=(test_generator.samples / batch_size) + 1,
        verbose=1)
    return predictions
    # print test


def pair_predict(pair_model_path, test_generator):

    pair_model = load_model(pair_model_path)
    print ("Checkpoint " + pair_model_path + " loaded.")
    print 'predicting...'

    for i, layer in enumerate(pair_model.layers):
        print (i, layer.name)

    # model_xc = Model(input=pair_model.input[0], output=pair_model.get_layer('ctg_out_1').output)
    model_ic = Model(input=pair_model.layers[1].input, output=pair_model.get_layer('ctg_out_2').output)
    # model2 = Model(input=pair_model.inputs, output=pair_model.get_layer('ctg_out_2').output)
    # dif = Model(input=pair_model.inputs, output=pair_model.get_layer('bin_out').output)

    for i, layer in enumerate(model_ic.layers):
        print (i, layer.name)

    # predictions1 = model_ic.predict_generator(
    #     test_generator,
    #     steps=(test_generator.samples / batch_size) + 1,
    #     verbose=1)

    predictions1 = model_ic.predict_generator(
        test_generator,
        steps=(test_generator.samples / batch_size) + 1,
        verbose=1)

    # diffs = dif.predict_generator(
    #     test_generator,
    #     steps=(test_generator.samples / batch_size) + 1,
    #     verbose=1)

    # return predictions1, predictions2, diffs
    return predictions1

    # predictions1 = pair_model.predict_generator(
    #     pair_generator(test_generator, batch_size, train=False),
    #     steps=(test_generator.samples / batch_size) + 1,
    #     verbose=1)
    # return predictions1


def write_csv(test, csv_out_path):
    n = 0

    with open(csv_sample_path, 'r') as f:
        id = f.readline()
        # print id
    with open(csv_out_path, 'a') as f:
        f.writelines(id)
        for i, file_dir in enumerate(test_generator.filenames):
            file_name = file_dir.split('/')[-1]
            file_name, file_ext = file_name.split('.')
            # print file_name, file_ext
            pred_test = test[i]
            if file_ext == 'png' or file_ext == 'jpg':
                f.write(file_name)
                for str in pred_test:
                    f.write(',')
                    # f.write(str)
                    f.write(str.astype("str"))
                f.write('\n')

                n += 1

                print("count_image:", n)


if __name__ == '__main__':
    test_generator = gen_test_gen((299, 299), xception.preprocess_input)

    # test = predict(pair_model_best, test_generator)
    # test1, test2, dif = pair_predict(pair_model_best, test_generator)
    test = pair_predict(pair_model_best, test_generator)

    write_csv(test, csv_out_path)


    # # predict single image
    # im = Image.open("00d9537c197b7c4c4cdbd5d03c34b58a.jpg")
    # predict_im = predict(model, im, (img_height, img_width))
    # print predict_im
    #
    # predict images


