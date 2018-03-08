# coding:utf-8

import h5py
import numpy as np
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.optimizers import *
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

np.random.seed(2018)

csv_sample_path = './predict_csv/sample_submission.csv'
csv_out_path = './predict_csv/13_hdropori_1024_10000.csv'

# test_data_dir = '/home/fenglf/data/dog/kaggle/cropped_test_jpg'
test_data_dir = '/home/fenglf/data/dog/kaggle/test'
h5path = '/home/fenglf/PycharmProjects/keras-finetuning-master/gap_feature/'
model_best_path = './model/output/mul/model_13_hdropori_1024_best.h5'
# model_json_path = './model/output/mul/model_13_ndropcop_2048.json'


nb_classes = 120
batch_size = 32

# img_height, img_width = 299, 299
eval_switch = True

bottleneck_feature_test_list = ["gap_InceptionResNetV2_test.h5",
                                "gap_DenseNet121_test.h5",
                                "gap_DenseNet169_test.h5",
                                "gap_DenseNet161_test.h5",
                                "gap_DenseNet201_test.h5",
                                "gap_Xception_test.h5",
                                "gap_ResNet50_test.h5",
                                "gap_ResNet101_test.h5",
                                "gap_ResNet152_test.h5",
                                "gap_InceptionV3_test.h5",
                                "gap_InceptionV4_test.h5",
                                "gap_VGG16_test.h5",
                                "gap_VGG19_test.h5"]


def predict():
    print("start...")
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    print "test_generator creating..."
    # 注意：使用此方法时，test_data_dir必须有子文件夹
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
        class_mode=None)

    # define the input_shape
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # setup model
    # load json and create model
    json_file = open('model_json_3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model_3.h5")
    print("Loaded model from disk")

    # if os.path.exists(final_weights_path):
    #     model.load_weights(final_weights_path)
    #     print ("Checkpoint " + final_weights_path + " loaded.")

    print 'predicting...'

    # print test_generator.filenames

    test = model.predict_generator(
        test_generator,
        steps=(test_generator.samples // batch_size) + 1,  # 25
        verbose=1)


    print test
    # y_pred = model.predict(X_test, verbose=1)
    # top1_class = y_pred.max(axis=1)
    # print top1_class


def gap_predict():
    X_test = []

    for filename in bottleneck_feature_test_list:
        with h5py.File(h5path + filename, 'r') as h:
            X_test.append(np.array(h['test']))


    # setup model
    # load json and create model
    # json_file = open(model_json_path, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # # load weights into new model
    # model.load_weights(model_best_path)
    model = load_model(model_best_path)
    print("Loaded model from {}".format(model_best_path))

    # 拼接特征
    X_test = np.concatenate(X_test, axis=1)

    res0 = model.predict(X_test)
    return res0


def write_csv(y_pred):
    n = 0

    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(test_data_dir, shuffle=False, batch_size=batch_size, class_mode=None)
    if os.path.exists(csv_out_path):
        os.remove(csv_out_path)

    with open(csv_sample_path, 'r') as f:
        id = f.readline()
    # print id
    with open(csv_out_path, 'a') as f:
        f.writelines(id)
        for i, file_dir in enumerate(test_generator.filenames):
            file_name = file_dir.split('/')[-1]
            file_name, file_ext = file_name.split('.')
            # print file_name, file_ext
            pred_test = y_pred[i]
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
    y_pred = gap_predict()
    write_csv(y_pred)