import torch
import torchvision
from torchvision import transforms

import os
import numpy as np
import cv2
from PIL import Image

def run(image_name):
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights = "DEFAULT") # Load the pre-trained DeepLabV3 model with ResNet-50 backbone
    
    model.eval() # model in evaluatopn mode

    # Define the image transformation pipeline:
    # 1. Convert to tensor (0-255 → 0-1)
    # 2. Normalize based on ImageNet statistics (what the model expects)
    transform = transforms.Compose([
        transforms.ToTensor() , # change the image from 0 to 255
        transforms.Normalize(mean = [0.485, 0.456, 0.486], std= [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_name).convert("RGB")  # Open the image and convert it to RGB (to avoid 4-channel errors)

    with torch.no_grad(): #we don't require a gradient for back propagation
        pred = model(transform(img)[None, ...])


     # Get the prediction output:
    # 1. Remove batch dimension
    # 2. Take argmax across the class dimension to get the class for each pixel
    output = pred["out"].squeeze().argmax(0)

    # List of class names as per Pascal VOC dataset (excluding background class '0')
    names = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
             "cow","dining_table","dog","horse","bike","person","plant","sheep","sofa","train","tv"]
    
    all_objects = [] # keep the names of all the objects that can be found in the output
    all_segments = [] #all the images in white background that can be found all the objects

    for i in range(output.unique().shape[0] - 1): # '-1' to remove the background
        num = output.unique()[i+1] # '0' is always background
        all_objects.append(names[num -1])

        temp = torch.zeros_like(output)         # Create a blank mask the same shape as the output
        temp[output == num] = 255 # Set pixels of the mask where the class ID matches the current object (Object region is 255, rest is 0)

        mask = temp.numpy().astype("uint8")         # Convert the torch mask to a NumPy uint8 array for OpenCV
        real = cv2.imread(image_name)         # Read the original image using OpenCV (as BGR)

        real[mask != 255] = (255, 255, 255)         # Make all non-object pixels white (255, 255, 255)
        all_segments.append(real.copy())         # Append the segmented image to the list

    return all_objects, all_segments