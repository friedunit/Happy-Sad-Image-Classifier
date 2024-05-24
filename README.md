# Happy-Sad-Image-Classifier

## Overview

#### For this final project, I'll work on an image classification model for determining happy and sad images. Similar to the cancer classification, images can be classified with 0 and 1 for happy and sad. I will use the Keras Sequential model with convolutional 2D layers. A model like this could be used for sentiment analysis or determining emotions in real-time. This will be a fairly simplified model with only binary output but could be expanded for other emotions such as angry, scared, surprised, etc.

#### Most of the images I retrieved from Google searches and some were from a Kaggle dataset for emotions. 

## Observing the Data

![image](https://github.com/friedunit/Happy-Sad-Image-Classifier/assets/10797098/21276246-f21c-4e59-95c2-cd88dae4bf3d)

##
![image](https://github.com/friedunit/Happy-Sad-Image-Classifier/assets/10797098/f3192ad5-40b1-4b83-b3cd-5bea973c032e)

### After preprocessing the data to rescale, the images are classified as 1 for Sad and 0 for Happy and each image is 256 x 256, lets look at 12 rescaled and classified images

0 = Happy
1 = Sad

![image](https://github.com/friedunit/Happy-Sad-Image-Classifier/assets/10797098/274464bf-3622-4cc8-8da3-b99ecadf6409)

## Model Architecture

#### For this first Sequential model, I'll use 3 Convolutional 2D layers each with (3x3) filter size and stride of 1. The input shape is (256, 256, 3). 'relu' activation converts negative values from output to 0 and positive stays the same. MaxPooling2D layer takes max value after relu activation and returns that value to condense the info.

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">448</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)   │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">126</span>, <span style="color: #00af00; text-decoration-color: #00af00">126</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">4,640</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">63</span>, <span style="color: #00af00; text-decoration-color: #00af00">63</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">61</span>, <span style="color: #00af00; text-decoration-color: #00af00">61</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">4,624</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14400</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │     <span style="color: #00af00; text-decoration-color: #00af00">3,686,656</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">257</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>

## Plot Performance

![image](https://github.com/friedunit/Happy-Sad-Image-Classifier/assets/10797098/f8cc658d-ef45-4156-884a-903f9813faad)

![image](https://github.com/friedunit/Happy-Sad-Image-Classifier/assets/10797098/053436b3-a549-474e-a032-7055454aadf9)

Precision: 1.0, Recall: 1.0, Acc: 1.0

## Initial Results from Training

#### As we see above, the model performed very well with 100% accuracy. This could be from some possible duplicates in the data and only 256 images isn't much for a true model. 

## Testing the model

Predicted class is sad
![image](https://github.com/friedunit/Happy-Sad-Image-Classifier/assets/10797098/b1548251-9512-4c7e-abd5-4f61780ae997)

Predicted is happy
![image](https://github.com/friedunit/Happy-Sad-Image-Classifier/assets/10797098/f169a5bf-ae4f-424d-b164-3ee3b9870aa8)

## Predicted values vs True labels on Test batch

![image](https://github.com/friedunit/Happy-Sad-Image-Classifier/assets/10797098/8a02da4d-6662-4684-b1f2-af8e7db44e8e)

## Conclusion

#### With the initial model being trained on 192 images with 32 used for validation data and ran on only 20 epochs, the model performed extremely well. The validation accuracy hit 1.0 after only 10 epochs. With the test images above, the model also classified them all correctly. I'm very happy with the results and a model like this could be used for emotion detection in images or video frames. Expanding to other facial expressions and emotions would make it even more useful.
