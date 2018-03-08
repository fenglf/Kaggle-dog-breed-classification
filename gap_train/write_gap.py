from keras.models import *
from nets.densenet import *
from nets.inception_v4 import InceptionV4
from nets.imagenet_utils import preprocess_input
from keras.applications import *
from keras.preprocessing.image import *

import math
import h5py


#train_data_dir = '/home/fenglf/data/dog/stanford/Images/data'
train_data_dir = '/home/fenglf/data/dog/stanford/anno_crop_image/data'
test_data_dir = '/home/fenglf/data/dog/kaggle/cropped_test_jpg'
#test_data_dir = '/home/fenglf/data/dog/kaggle/test'

pretrained_model_root_dir = '/home/fenglf/PycharmProjects/keras-finetuning-master/model/pretrained/'
output_model_root_dir = '/home/fenglf/PycharmProjects/keras-finetuning-master/model/output'

h5path = '/home/fenglf/PycharmProjects/keras-finetuning-master/gap_feature_crop/'

batch_size = 32

input_included_model = ['VGG16',
                        'VGG19',
                        'ResNet50',
                        'InceptionV3',
                        'Xception',
                        'ResNet152',
                        'ResNet101',
                        'InceptionResNetV2']


# VGG ResNet DenseNet preprocess
def v_preprocess_input(x):
    x[:, :, 0] = (x[:, :, 0] - 124) * 0.0167
    x[:, :, 1] = (x[:, :, 1] - 117) * 0.0167
    x[:, :, 2] = (x[:, :, 2] - 104) * 0.0167
    return x
def vr_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x
def d_preprocess_input(x):
    x = x[:, :, ::-1]
    x[:, :, 0] = (x[:, :, 0] - 103.94) * 0.017
    x[:, :, 1] = (x[:, :, 1] - 116.78) * 0.017
    x[:, :, 2] = (x[:, :, 2] - 123.68) * 0.017
    return x

def den_preprocess_input(x):
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    x[..., 0] /= std[0]
    x[..., 1] /= std[1]
    x[..., 2] /= std[2]


def get_model(MODEL, image_size, input_tensor=None, pretrained_model_dir=None, final_weights_path=None):
    if final_weights_path:
        if MODEL.__name__ in input_included_model:
            base_model = MODEL(input_tensor=input_tensor, weights=pretrained_model_dir, include_top=False)
            base_model.load_weights(final_weights_path, by_name=True)
            # base_model.load_weights(final_weights_path, by_name=False)
            # for i, layer in enumerate(base_model.layers):
            #     print (i, layer.name)
            out = GlobalAveragePooling2D()(base_model.output)
            # print(out.shape)
            model = Model(input_tensor, out)

        elif MODEL.__name__ == 'InceptionV4':
            base_model = MODEL(weights=pretrained_model_dir, include_top=False)
        elif MODEL.__name__ == 'DenseNet121':
            # densenet_weights_path = densenet_weights_root + 'densenet121_weights_tf.h5'
            model = MODEL(reduction=0.5, input_shape=(image_size[0], image_size[1], 3), weights=pretrained_model_dir)
        elif MODEL.__name__ == 'DenseNet161':
            # densenet_weights_path = densenet_weights_root + 'densenet161_weights_tf.h5'
            model = MODEL(reduction=0.5, input_shape=(image_size[0], image_size[1], 3), weights=pretrained_model_dir)
        elif MODEL.__name__ == 'DenseNet169':
            # densenet_weights_path = densenet_weights_root + 'densenet169_weights_tf.h5'
            model = MODEL(reduction=0.5, input_shape=(image_size[0], image_size[1], 3), weights=pretrained_model_dir)
        elif MODEL.__name__ == 'DenseNet201':
            # densenet_weights_path = densenet_weights_root + 'densenet169_weights_tf.h5'
            model = MODEL(reduction=0.5, input_shape=(image_size[0], image_size[1], 3), weights=pretrained_model_dir)
        print ("Fine-tuned Checkpoint '" + final_weights_path + "' loaded.")

    else:
        if MODEL.__name__ in input_included_model:

            base_model = MODEL(input_tensor=input_tensor, weights=pretrained_model_dir, include_top=False)
            print ("Primary Checkpoint '" + pretrained_model_dir + "' loaded.")
            out = GlobalAveragePooling2D()(base_model.output)
            # print(out.shape)
            model = Model(input_tensor, out)
        elif MODEL.__name__ == 'InceptionV4':
            model = MODEL(weights=pretrained_model_dir, include_top=False)
        elif MODEL.__name__ == 'DenseNet121':
            # densenet_weights_path = densenet_weights_root + 'densenet121_weights_tf.h5'
            model = MODEL(reduction=0.5, input_shape=(image_size[0], image_size[1], 3), weights=pretrained_model_dir)
        elif MODEL.__name__ == 'DenseNet161':
            # densenet_weights_path = densenet_weights_root + 'densenet161_weights_tf.h5'
            model = MODEL(reduction=0.5, input_shape=(image_size[0], image_size[1], 3), weights=pretrained_model_dir)
        elif MODEL.__name__ == 'DenseNet169':
            # densenet_weights_path = densenet_weights_root + 'densenet169_weights_tf.h5'
            model = MODEL(reduction=0.5, input_shape=(image_size[0], image_size[1], 3), weights=pretrained_model_dir)
        elif MODEL.__name__ == 'DenseNet201':
            # densenet_weights_path = densenet_weights_root + 'densenet169_weights_tf.h5'
            model = MODEL(reduction=0.5, input_shape=(image_size[0], image_size[1], 3), weights=pretrained_model_dir)
    return model


def write_gap(MODEL, image_size, lambda_func=None):
    # define weights path
    pretrained_model_dir = os.path.join(pretrained_model_root_dir, MODEL.__name__,
                                        MODEL.__name__ + '_notop.h5')
    final_weights_path = os.path.join(output_model_root_dir, MODEL.__name__, 'final_weights',
                                      MODEL.__name__ + '.final_weights.hdf5')
    print(MODEL.__name__)

    # define input
    input_tensor = Input((image_size[0], image_size[1], 3))
    #x = input_tensor

    # setup model
    #model = get_model(MODEL, image_size, input_tensor, pretrained_model_dir, final_weights_path)
    model = get_model(MODEL, image_size, input_tensor, pretrained_model_dir)

    # prepare data
    gen = ImageDataGenerator(
        preprocessing_function=lambda_func
    )

    # width = image_size[0]
    # height = image_size[1]
    # input_tensor = Input((height, width, 3))
    # x = input_tensor
    # if lambda_func:
    #     x = Lambda(lambda_func)(x)
    #
    # base_model = MODEL(input_tensor=x, weights=pretrained_model_dir, include_top=False)
    # model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    #
    # gen = ImageDataGenerator()

    train_generator = gen.flow_from_directory(train_data_dir, image_size, shuffle=False,
                                              batch_size=batch_size)
    test_generator = gen.flow_from_directory(test_data_dir, image_size, shuffle=False,
                                             batch_size=batch_size, class_mode=None)
    # gen bottle-neck feature
    train = model.predict_generator(
        train_generator,
        #steps=train_generator.samples // batch_size,  # must divided exactly
        steps=train_generator.samples / batch_size + 1,
        # steps=2,
        verbose=1)

    with h5py.File(h5path + "gap_%s_train.h5" % MODEL.func_name) as h:
        print 'creating_bottleneck_feature...'
        h.create_dataset("train", data=train)
        # h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)

    test = model.predict_generator(
        test_generator,
        steps=test_generator.samples / batch_size + 1,  # if not exatly divided
        verbose=1)
    with h5py.File(h5path + "gap_%s_test.h5" % MODEL.func_name) as h:
        h.create_dataset("test", data=test)


if __name__ == '__main__':
    #write_gap(ResNet50, (224, 224))
    #write_gap(InceptionResNetV2, (299, 299), inception_resnet_v2.preprocess_input)
    #write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
    write_gap(InceptionV4, (299, 299), preprocess_input)
    #write_gap(Xception, (299, 299), xception.preprocess_input)
    # write_gap(ResNet50, (224, 224), vr_preprocess_input)
    # write_gap(ResNet101, (224, 224), vr_preprocess_input)
    # write_gap(ResNet152, (224, 224), vr_preprocess_input)
    # write_gap(DenseNet121, (224, 224), den_preprocess_input)
    # write_gap(DenseNet161, (224, 224), den_preprocess_input)
    # write_gap(DenseNet169, (224, 224), den_preprocess_input)
    # write_gap(VGG16, (224, 224), vr_preprocess_input)
    # write_gap(VGG19, (224, 224), vr_preprocess_input)
    # write_gap(DenseNet201, (224, 224), den_preprocess_input)
