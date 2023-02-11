#!/usr/bin/env python
# coding: utf-8

# # Pothole Detection using Computer Vision
# 

# ### Environment Setup

# #### Install tensorflow v1.14

# In[1]:


# check tensorflow version
import tensorflow as tf
print(tf.__version__)


# In[2]:


# Colab default version is now 2.6.0, so uninstall it and install v1.14

get_ipython().run_line_magic('tensorflow_version', '2.x')
get_ipython().system('pip uninstall -y tensorflow tensorboard tensorflow-estimator tensorboard-plugin-wit')
get_ipython().system('pip install tensorflow-gpu==1.14.0 tensorboard==1.14.0 tensorflow-estimator==1.14.0')


# In[2]:


# tensorflow version

import tensorflow as tf
print(tf.__version__)


# #### Install necessary libraries

# In[3]:


# install necessary libraries

get_ipython().system('apt-get install -qq protobuf-compiler python-pil python-lxml python-tk')
get_ipython().system('pip install -q pillow lxml jupyter matplotlib cython pandas contextlib2')
get_ipython().system('pip install -q pycocotools tf_slim')


# #### Mount the drive

# In[4]:


#mount google drive

from google.colab import drive
drive.mount('/content/gdrive')


# #### Create a project folder inside the drive

# In[ ]:


# create a folder 'pothole' inside My Drive and navigate to it
get_ipython().run_line_magic('cd', '/content/gdrive/My Drive/pothole')


# #### Clone tensorflow 1 object detection api

# In[ ]:


# clone tensorflow 1 object detection api
get_ipython().system('git clone --quiet -b r1.13.0 https://github.com/tensorflow/models.git')


# #### Compile protos

# In[ ]:


#navigate to /content/gdrive/My Drive/pothole/models/research folder 
get_ipython().run_line_magic('cd', '/content/gdrive/My Drive/pothole/models/research')
 
# compile protos
get_ipython().system('protoc object_detection/protos/*.proto --python_out=.')

import os
os.environ['PYTHONPATH'] += ':/content/gdrive/My Drive/pothole/models/research/:/content/gdrive/My Drive/pothole/models/research/slim/'

get_ipython().system('pip install .')


# #### Test installation

# In[9]:


# Test installation
get_ipython().system('python object_detection/builders/model_builder_test.py')


# #### COCO API Installation

# In[ ]:


#navigate to /content/gdrive/My Drive/pothole/ folder
get_ipython().run_line_magic('cd', '/content/gdrive/My Drive/pothole/')

# clone cocoapi
get_ipython().system('git clone --quiet https://github.com/cocodataset/cocoapi.git')


# #### Build makefile

# In[12]:


# navigate to the folder PythonAPI
get_ipython().run_line_magic('cd', '/content/gdrive/My Drive/pothole/cocoapi/PythonAPI')

# build makefile
get_ipython().system('make')


# In[ ]:


# copy the folder pycocotools to the location '/content/gdrive/MyDrive/pothole/models/research/'
get_ipython().system('cp -r pycocotools /content/gdrive/MyDrive/pothole/models/research/')


# ### Download Pretrained Model
# * Create 'training' directory
# * Navigate in to 'training' folder and create the folder 'pre-trained-models' 

# In[ ]:


get_ipython().run_line_magic('cd', '/content/gdrive/My Drive/pothole/training/pre-trained-models')


# #### Download the pretrained ssd_mobilenet from [TF1 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

# In[ ]:


get_ipython().system('wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz')


# #### Extract contents of ssd_mobilenet

# In[ ]:


get_ipython().system('tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz')


# ### Data Preprocessing

# * Create 'images' folder and navigate into it
# 

# In[ ]:


# navigate to images folder
get_ipython().run_line_magic('cd', '/content/gdrive/My Drive/pothole/training/images')


# * Compress train and test images and create train.zip and test.zip files
# * Upload train.zip and test.zip to 'images' folder
# * Extract train.zip and test.zip files

# In[ ]:


# unzip train.zip
get_ipython().system('unzip train.zip')

# unzip test.zip
get_ipython().system('unzip test.zip')


# #### Convert .xml files to csv

# In[ ]:


# Create train data:
get_ipython().system('python scripts/xml_to_csv.py -i images/train -o annotations/train_labels.csv')

# Create test data:
get_ipython().system('python scripts/xml_to_csv.py -i images/test -o annotations/test_labels.csv')


# #### create label map file

# In[ ]:


# create lable_map.pbtxt as shown 
# place it in annotations folder
get_ipython().system('cat annotations/label_map.pbtxt')


# #### Create train.record and test.record files

# In[ ]:


# Create train data:
get_ipython().system('python scripts/generate_tfrecord_v1.py      --csv_input=annotations/train_labels.csv      --output_path=annotations/train.record      --img_path=images/train      --label_map annotations/label_map.pbtxt')

# Create test data:
get_ipython().system('python scripts/generate_tfrecord_v1.py      --csv_input=annotations/test_labels.csv      --output_path=annotations/test.record      --img_path=images/test      --label_map annotations/label_map.pbtxt')


# ### Training

# #### Tensorboard

# In[ ]:


# track training usnig tensorboard

get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', "--logdir='/content/gdrive/MyDrive/pothole/training/my_ssd_mobilenet_v2_coco_2018_03_29'")


# #### edit pipiline.config file 
# * Note: please remove the line 'batch_norm_trainable: true' from config file

# In[ ]:


# navigate to pre-trained-models/ssd_mobilenet_v2_coco_2018_03_29
get_ipython().run_line_magic('cd', '/content/gdrive/My Drive/pothole/training/pre-trained-models/ssd_mobilenet_v2_coco_2018_03_29')

# copy the pipiline.config file 
get_ipython().system('cp -r pipeline.config /content/gdrive/MyDrive/pothole/training/models/my_ssd_mobilenet_v2_coco_2018_03_29/')


# In[ ]:


# edit the pipline.config file as shown
get_ipython().system('cat /content/gdrive/MyDrive/pothole/training/models/my_ssd_mobilenet_v2_coco_2018_03_29/pipeline2.config')


# In[ ]:


# training

get_ipython().system('python /content/gdrive/MyDrive/pothole/models/research/object_detection/model_main.py      --pipeline_config_path=/content/gdrive/MyDrive/pothole/training/models/my_ssd_mobilenet_v2_coco_2018_03_29/pipeline.config      --model_dir=/content/gdrive/MyDrive/pothole/training/my_ssd_mobilenet_v2_coco_2018_03_29      --alsologtostderr')


# In[ ]:


# continue training

get_ipython().system('python /content/gdrive/MyDrive/pothole/models/research/object_detection/model_main.py      --pipeline_config_path=/content/gdrive/MyDrive/pothole/training/models/my_ssd_mobilenet_v2_coco_2018_03_29/pipeline.config      --model_dir=/content/gdrive/MyDrive/pothole/training/my_ssd_mobilenet_v2_coco_2018_03_29      --alsologtostderr')


# ### Training Result

# #### Loss vs Steps

# <img src="3_ssd_loss.png" alt="3_ssd_loss">

# * As loss is not reducing, training is stopped after 44469 steps

# #### mAP@0.5 IoU vs Steps

# <img src="3_ssd_mAP.png" alt="3_ssd_mAP">

# <img src="3_ssd_trainStep.png" alt="3_ssd_trainStep">  

# As shown above obtained loss of 5.45, also achieved mAP (mean Average Precision) of 0.155 at IoU (Intersection over Union) threshold of 0.5.

# ### Prediction

# #### Export the trained model

# In[19]:


import os
import re
import numpy as np

# train logs
model_dir = '/content/gdrive/MyDrive/pothole/training/my_ssd_mobilenet_v2_coco_2018_03_29'

# get latest model check point
lst = os.listdir(model_dir)
lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
steps=np.array([int(re.findall('\d+', l)[0]) for l in lst])
last_model = lst[steps.argmax()].replace('.meta', '')

last_model_path = os.path.join(model_dir, last_model)
print(last_model_path)


# In[21]:


# export the model
get_ipython().system('python /content/gdrive/MyDrive/pothole/models/research/object_detection/export_inference_graph.py      --input_type=image_tensor      --pipeline_config_path=/content/gdrive/MyDrive/pothole/training/models/my_ssd_mobilenet_v2_coco_2018_03_29/pipeline.config      --output_directory=/content/gdrive/MyDrive/pothole/training/final_model      --trained_checkpoint_prefix={last_model_path}')


# #### Set images for the inference

# In[23]:


IMAGE_DIR = '/content/gdrive/MyDrive/pothole/training/test_images'
IMAGE_PATHS = []

for file in os.listdir(IMAGE_DIR):
    if file.endswith(".JPG"):
        IMAGE_PATHS.append(os.path.join(IMAGE_DIR, file))

IMAGE_PATHS


# In[24]:


# label_map path
PATH_TO_LABELS = '/content/gdrive/MyDrive/pothole/training/annotations/label_map.pbtxt'


# In[25]:


# model check point path
output_dir = '/content/gdrive/MyDrive/pothole/training/final_model'
PATH_TO_CKPT = os.path.join(os.path.abspath(output_dir), "frozen_inference_graph.pb")


# In[45]:


# test image bounding boxes
import pandas as pd
img_bbox_df = pd.read_csv('/content/gdrive/MyDrive/pothole/training/test_images/test_bboxes_tf.csv')
img_bbox_df.head()


# #### Inference

# In[61]:


# Reference: https://towardsdatascience.com/object-detection-by-tensorflow-1-x-5a8cb72c1c4b
import cv2
from google.colab.patches import cv2_imshow
get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/pothole/models/research/object_detection')

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops


# This is needed to display the images.
get_ipython().run_line_magic('matplotlib', 'inline')


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

num_classes = 3

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (25, 30)
color = np.array([98])

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


for image_path in IMAGE_PATHS:
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        # track_ids=color,
        use_normalized_coordinates=True,
        line_thickness=10)
    
    plt.figure(figsize=IMAGE_SIZE, dpi=200)
    image_id = image_path.split('/')[-1]
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    coords = img_bbox_df[img_bbox_df['image_id'] == image_id.split('.')[0]]
    coords = coords[['x', 'y', 'x_max', 'y_max']].values
    plt.subplot(1, 2, 1)
    plt.title('Input')
    plt.xticks([])
    plt.yticks([])
    # drawing BBoxes
    for c in coords:
        cv2.rectangle(img, (c[0], c[1]), 
                      (c[2], c[3]), (255, 0, 0), 3)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title('Detection')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image_np)
  


# ### Summary
# 

# * Obtained minimum loss of 5.45
# * Achieved map of 0.155 @ 0.5 IoU
# * From the above plots we can see that model is able to detect potholes quite well. 
# * It also missed potholes in case of multiple potholes in a single image. Also the bounding boxes drawn do not fit the potholes properly. 
# * In case of MobileNet there is always trade-off between performance and model complexity.
