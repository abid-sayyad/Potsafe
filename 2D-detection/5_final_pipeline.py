#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

import os
import cv2
import matplotlib.pyplot as plt
from detect import run


# In[2]:


def detect(test_input, type='image'):
    """ 
    Function to detect potholes given an image or a video input
    Parameters:
        test_input: image/video input
    Output:
        image/video with detection
    """
    # weights
    weights_path = 'runs/train/exp4/weights/best.pt'
    # to save the detection
    output_path = 'detect_out'
    # img dimension
    img_width, img_height = 640, 640
    img_size = [img_width, img_height]
    # confidence threshold
    conf_threshold = 0.5
    # iou threshold
    iou_threshold = 0.5
    # bounding box thickness
    bbox_line_thick = 5

    # detection on image/video
    run(weights=weights_path, 
        source=test_input, 
        imgsz=img_size,
        conf_thres=conf_threshold, 
        iou_thres = iou_threshold,
        project=output_path,
        name='',
        exist_ok=True,
        line_thickness=bbox_line_thick
    )
    # if the input is image show the detection
    if type == 'image':
        # plot the detection
        plt.figure(figsize = (25, 30))
        # output path
        res_path = os.path.join(output_path, test_input)
        # read the image
        res = cv2.imread(res_path, cv2.IMREAD_UNCHANGED)
        plt.title('Detection')
        plt.xticks([])
        plt.yticks([])
        # show the image
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        plt.show()


# ## Inference

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
img = 'G0028847.JPG'
detect(img)


# In[4]:


img = 'G0028334.JPG'
detect(img)


# In[9]:


img = 'G0016516.JPG'
detect(img)


# In[10]:


img = 'G0027694.JPG'
detect(img)

