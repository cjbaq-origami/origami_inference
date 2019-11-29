import tensorflow as tf
import numpy as np
from keras.models import load_model
import time, os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import pickle
from keras.models import Model
import sys
from keras import backend as K
from keras.utils import print_summary
import argparse

# with a Sequential model
#get_3rd_layer_output = K.function([model.layers[0].input],
#                                  [model.layers[3].output])
#layer_output = get_3rd_layer_output([x])[0]
max_mean = np.float('-inf')
min_mean = np.float('inf')
def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg


  if data_format == 'channels_first':
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def compute_accuracy(accuracy_list):
    print("length of acc list is: ", len(accuracy_list))
    print("accuracy is:", sum(accuracy_list)/len(accuracy_list))

def sparsify(feat):
    zero_rate = 25
    feat_size = feat.shape
    null_matrix =  np.random.randint(0, 100, size=feat_size) > zero_rate
    feat = np.multiply(feat, null_matrix)
    return feat

def thresholding(feat):
    # As implemented in KDD 2018 paper
    feat = np.float16(feat)
    B = 2950.6548
    inf_norm = np.max(np.abs(feat))
    #print("norms: ", inf_norm, B, inf_norm/B)
    feat = feat/max(1.0, inf_norm/B)
    return np.float16(feat)

def randomize(feat):
    #B = 2950.6548
    #epsilon = 3.0
    scale = 100.0
    feat = feat + np.random.laplace(0, scale)
    return np.float16(feat)

def resnet50_asitis(path):
    model = VGG16(weights='imagenet')
    direc_path = path 
    i = 0
    accuracy_list=[]
    for direc in os.listdir(direc_path):
        #if i > 100: break
        for img_name in os.listdir(direc_path + direc):
            # pre-process each image and normalize before inference
            inp_img = image.load_img(direc_path + direc + "/" + img_name, target_size=(224, 224))
            #inp_img = image.load_img(direc_path + direc + "/" + img_name, target_size=(299,299))
            x = image.img_to_array(inp_img)
            x = np.expand_dims(x, axis=0)
            inputInit = preprocess_input(x.copy())
            result2 = model.predict(inputInit)
            predicted2 = decode_predictions(result2, top=3)[0]
            accuracy_list.append(predicted2[0][0] == direc)
            if i and i % 10 == 0:
                compute_accuracy(accuracy_list)
            i += 1

def main(path):
    strides = 2
    batches = 1
    #firstConv = load_model('first_conv.h5')
    model_full = VGG16(weights='imagenet')
    #print_summary(model_full, line_length=None, positions=None, print_fn=None)
    #sys.exit(0)
    #layer_name_conv1 = 'block1_conv1'
    get_1st_layer_output = K.function([model_full.layers[0].input],
                                  [model_full.layers[6].output]) 
    
    get_final_layer_output = K.function([model_full.layers[7].input],
                                  [model_full.layers[-1].output]) 
    #firstConv_model = Model(inputs=model_full.input , outputs=model_full.get_layer(layer_name_conv1).output) 
    #model_part2 = load_model('ResNet50_NoConv1.h5')
    #layer_name_conv2 = 'block1_conv2'
    #layer_name_final = 'predictions'
    #model_part2 = Model(inputs=model_full.get_layer(layer_name_conv2).input, outputs=model_full.get_layer(layer_name_final).output) 
    #layer_name = 'bn2a_branch1'
    #intermediate_layer_model = Model(inputs=model_part2.input,
    #                                 outputs=model_part2.get_layer(layer_name).output)
    #base = RN(weights='imagenet')
    direc_path = path 
    i = 0
    accuracy_list=[]
    for direc in os.listdir(direc_path):
        #if i > 100: break
        for img_name in os.listdir(direc_path + direc):
            # pre-process each image and normalize before inference
            inp_img = image.load_img(direc_path + direc + "/" + img_name, target_size=(224, 224))
            #inp_img = image.load_img(direc_path + direc + "/" + img_name, target_size=(299,299))
            x = image.img_to_array(inp_img)
            x = np.expand_dims(x, axis=0)
            #x = randomize(x)
            inputInit = preprocess_input(x.copy())
            #inputInit = randomize(inputInit)
	    # Model part 1
            start = time.time()
            firstFeat = get_1st_layer_output([inputInit])[0]  
            firstFeat = np.float16(firstFeat) 
            newDataSet = (x, firstFeat)
            if i and i % 10 == 0:
            #    #print(intermediateFeat.shape)
                print(firstFeat.shape)
            #    print('intermediate conv takes', time.time() - start)
            
            with open('perturbedData/%d.pickle' % i, 'wb') as handle:
                pickle.dump(newDataSet, handle)
	    # Model part 3
            start = time.time()
            result2 = get_final_layer_output([firstFeat])[0]
            predicted2 = decode_predictions(result2, top=3)[0]
            accuracy_list.append(predicted2[0][0] == direc)
            if i and i % 10 == 0:
                compute_accuracy(accuracy_list)
                print('remaining takes', time.time() - start)

            i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="the path to dataset") 
    args = parser.parse_args()
    path = args.path
    main(path)
    #resnet50_asitis(path)
