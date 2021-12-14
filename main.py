from PIL import Image
from AI import model
import matplotlib.pyplot as plt
import numpy as np
import os
from peekingduck.pipeline.nodes.model import yolo
from peekingduck.pipeline.nodes.draw import bbox
from typing import Tuple
from model import *
yolo_node = yolo.Node()
draw_node = bbox.Node()
model = getmodel()

classes = ["real", "fake"]

def crop(image, desired_size : Tuple[int, int], youtput):
    global trainX, left
    percentage = youtput['bbox_scores']
    bboxes = youtput['bboxes']
    width, height = image.size
    highestindex = 0
    highest = 0
    count = 0
    for i in percentage:
        if i > highest:
            highest = i
            highestindex = count
        count += 1
    if not len(bboxes):
        return
    elif highest < 0.5: 
        return
    bbox = bboxes[0]
    x1 = bbox[0] * height
    x2 = bbox[2] * height
    y1 = bbox[1] * width 
    y2 = bbox[3] * width
    middleX = (x1 + x2) // 2
    middleY = (y1 + y2) // 2
    halfdimH = desired_size[0]/2
    halfdimW = desired_size[1]/2
    # x1 = width 
    # y1 = height
    left = middleX - halfdimW
    right = middleX + halfdimW
    upper = middleY - halfdimH
    lower = middleY + halfdimH
    if middleX < halfdimW: 
        left = 0
        remainding = halfdimW - middleX
        right = middleX + halfdimW + remainding
    if (halfdimW + middleX) > width:
        right = width
        left = width - desired_size[1]
    if middleY < halfdimH:
        upper = 0
        lower = 0 + desired_size[0]
    if (halfdimH + middleY) > height:
        lower = height
        upper = height - desired_size[0]
    newimage = image.crop((left, upper, right, lower)) # left, upper, right, lower
    newimage = np.array(newimage)
    return newimage

def main():
    inputdir = input("Please enter the file path of your image\n")
    try:
        image = Image.open(inputdir)
    except:
        print("Not a valid filepath.")
        return
    width, height = image.size
    if width > 480 and height > 480:
        print("Your image has to be larger than 480 by 480!")
        return
    youtput = yolo_node.run(inputs = {"img" : image})
    croppedimage = crop(image, (480, 480), youtput)
    prediction = model.predict(croppedimage.reshape(1, 480, 480, 3))
    biggest = 0
    for i in prediction: 
        if i > biggest:
            biggest = i
    prediction = prediction.argmax()
    labels = np.array([f"{classes[prediction]} {biggest}"])
    draw_node.run(inputs = {"img" : image, "bboxes" : youtput["bboxes"][0], "bbox_labels" : labels})
    image = np.array(image)
    plt.imshow(image)
    plt.show()

while True:
    main()
