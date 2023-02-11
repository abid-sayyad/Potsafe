#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

import os
import pandas as pd
from sklearn import model_selection
import numpy as np
import shutil
import cv2
import random
import matplotlib.pyplot as plt
from IPython.display import Image


# In[8]:


# data directory
data_dir = 'processed_data/train/positive' 


# In[12]:


get_ipython().system('ls')


# In[15]:


def get_file_path_df(folder):
  """
  Function to get name and path of .jpg files in the specified folder
  takes folder input and returns pandas dataframe
  """
  image_path = [] 
  image_name = [] 

  for root, dirs, files in os.walk(folder):
    for f in files:
      if f.endswith(".JPG"): # checking for JPG extension 
        image_name.append(f.split('.')[0])
        image_path.append(os.path.join(root, f)) # appending to image_path list
  # df = dataframe
  df_cols = {'image_name':image_name, 'path': image_path} # creating a dictionary
  df = pd.DataFrame(df_cols) # creating a dataframe
  return df


# In[16]:


# getting absolute path of all images and storing in a dataframe called data_df
data_df = get_file_path_df(data_dir)

# displying random 5 rows
data_df.sample(5)


# In[17]:


data_df.shape


# * Only positive images are considered for training YOLO
# * Data contains 2658 images with potholes
# * 80% of the data is used for training and 20% used for validation

# In[9]:


# splitting the data in to train and validation 

train_df, valid_df = model_selection.train_test_split(data_df, test_size=0.2, random_state=21, shuffle=True)


# In[11]:


# Number of images in training set
print(train_df.shape)


# In[13]:


# Number of images in validation set
valid_df.shape


# In[14]:


# loading the csv file containing bounding box coordinates

bbox_df = pd.read_csv("augmented_BBox_df.csv")
bbox_df.head()


# In[50]:


def process_bbox(img_df, bbox_df, data_type, img_w, img_h):
    """
    Function to convert bounding box coordinates into YOLO format
    and also to arrange the images and bounding boxes in specified folders
    
    Parameters:
        1. img_df: data frame containing image name and path
        2. bbox_df: data frame containing bounding box coordinates
        3. data_type: type of data train / valid
        4. img_w: image width
        5. img_h: image height
    """
    # drictories for images and labels
    dirs = ['images/train',
            'images/valid',
            'labels/train',
            'labels/valid',
           ]
    
    # if folder does not exist create them
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # loop over each row of image data frame
    for _, row in img_df.iterrows():
        yolo_bbox = []
        
        image_name = row['image_name']
        bbox_coords = bbox_df[bbox_df['image_id'] == image_name]
        bbox_array = bbox_coords.loc[:, ['x', 'y', 'x_max', 'y_max']].values
        
        for bbox in bbox_array:
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]

        
            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
        
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h
            
            yolo_bbox.append([0, # object class 
                              x_center, # bbox x-center
                              y_center, # bbox y-center
                              w, # bbox width
                              h]) # bbox height
            
        yolo_bbox = np.array(yolo_bbox)
        label_path = f"labels/{data_type}/{image_name}.txt"
        
        # saving txt file containing class label and bbox coordinates
        np.savetxt(label_path, yolo_bbox)
        
        img_source = row['path']
        img_desti = f"images/{data_type}/{image_name}.JPG"
        # moving images to the specific folder
        shutil.move(img_source, img_desti)
    print("Done")


# In[51]:


image_width = 3680
image_height = 1964

process_bbox(train_df, bbox_df, 'train', image_width, image_height)


# In[52]:


process_bbox(valid_df, bbox_df, 'valid', image_width, image_height)


# In[1]:


# clone the yolov5 github repository

get_ipython().system('git clone https://github.com/ultralytics/yolov5')


# In[ ]:


# install all the requirements

get_ipython().system('pip install -r requirements.txt')


# ### Training

# In[73]:


get_ipython().system('python train.py --img 640 --batch 12 --epochs 100 --data pothole.yaml --weights yolov5s.pt --cache')


# ### Training Result

# #### Loss vs Epoch

# <img src="4_yolo_loss.png" alt="4_yolo_loss">

# <img src="4_yolo_trainStep.png" alt="4_yolo_trainStep">

# Model is trained for 100 epochs. After 100th epoch for validation data 
# * box_loss/localization loss (errors between the predicted bounding box and ground truth) is 0.05 
# * obj_loss/confidence loss (the objectness of the box) is 0.01 
# * cls_loss/classification loss is 0 (as there is only one class).

# <img src="4_yolo_mAP.png" alt="4_yolo_mAP">

# * Achieved mAP (mean Average Precision) of 0.792 at IoU (Intersection over Union) threshold of 0.5

# ### Testing

# In[82]:


get_ipython().system('python detect.py --source ../processed_data/test --weights runs/train/exp4/weights/best.pt --conf 0.5')


# In[18]:


def visualize_detection(img_dir, img_bbox_df, res_dir):
  """
  Function to visualize images
  Parameters:
    1. img_dir: image directory
    2. img_bbox_df: image bbox
    3. res_dir: result directory
  """
  
  # smpls = random.sample(os.listdir(img_dir), 5)  
  smpls = os.listdir(img_dir)[:5]

  for img_id in smpls:
    img_path = f'{img_dir}/{img_id}'
    for res_img_id in os.listdir(res_dir):
        if img_id == res_img_id:
            res_path = f'{res_dir}/{res_img_id}'
            
            plt.figure(figsize = (25, 30))
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            coords = img_bbox_df[img_bbox_df['image_id'] == img_id.split('.')[0]]
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
            res = cv2.imread(res_path, cv2.IMREAD_UNCHANGED)  
            plt.subplot(1, 2, 2)
            plt.title('Detection')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        plt.show()


# ### Inference

# In[19]:


img_bbox_df = pd.read_csv('test_bboxes.csv')


# In[21]:


image_dir = 'processed_data/test_img'
result_dir = 'yolov5/runs/detect/exp9'

visualize_detection(image_dir, img_bbox_df, result_dir)


# In[17]:


image_dir = 'processed_data/test'
result_dir = 'yolov5/runs/detect/exp9'

visualize_detection(image_dir, img_bbox_df, result_dir)

