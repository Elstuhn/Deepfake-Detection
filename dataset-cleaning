import os
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np 
from peekingduck.pipeline.nodes.model import yolo
from peekingduck.pipeline.nodes.draw import bbox
classes = ["real", "fake"]
labels = [1 for i in range(56589)] + [0 for i in range(57827)] 
biggest = (0, 0, 3)
total = 56589 + 57827

trainX = []
testX = []
yolo_node = yolo.Node()
draw_node = bbox.Node()
def detect(image): 
    image = np.array(image)
    youtput = yolo_node.run(inputs = {"img" : image})
    return youtput 

def crop(image, desired_size : Tuple[int, int], youtput):
    global trainX, left
    try:
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
        noutput = yolo_node.run(inputs = {"img" : newimage})
    #draw_node.run(inputs = {"img" : newimage, "bboxes" : noutput["bboxes"], "bbox_labels" : noutput["bbox_labels"]})
        if not len(noutput["bboxes"]):
            return
        image = np.array(image)
        trainX.append(newimage)
        
    except:
        return
    

filesgonethrough = 0
count = 0
for files in os.listdir("images"):
    #if count in [0, 1, 2, 3]:
    #    count += 1
    #    continue
    for images in os.listdir(f"images/{files}"):
        images = Image.open(f"images/{files}/{images}")
        crop(images, (480, 480), detect(images))
        filesgonethrough += 1
    print(f"{filesgonethrough}/{total}")
    
        
for files in os.listdir("imagesoriginal"):
    for images in os.listdir(f"imagesoriginal/{files}"):
        images = Image.open(f"imagesoriginal/{files}/{images}")
        crop(images, (480, 480), detect(images))
        filesgonethrough += 1
    print(f"{filesgonethrough}/{total}")

trainX = np.array(trainX)
testX = trainX[45147:68030] #22883
del trainX[45147:68030]
labelsY = [1 for i in range(11442)] + [0 for i in range(11441)]

with open('dataset/trainX', 'wb') as writefile:
    pickle.dump(trainX, writefile)

with open('dataset/testX', 'wb') as writefile:
    pickle.dump(testX, writefile)

with open('dataset/trainY', 'wb') as writefile:
    pickle.dump(labels, writefile)
    
with open('dataset/testY', 'wb') as writefile:
    pickle.dump(labelsY, writefile)

plt.imshow(trainX[0])
plt.show()
trainX = trainX / 255.0
plt.imshow(trainX[0])
plt.show()
 
