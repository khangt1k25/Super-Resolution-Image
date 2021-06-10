## Super resolution image

SRResnet & SRGAN implementations.

## Dataset

I used VOC2012 with 17k pair of images (LR, HR) to train models. (LR, HR) with size (22x22) upto (88x88) with scale factor = 4. 

Then test phase with 3 benchmark datasets:  BDS100, Set14, Set5. 



## Result

Here is the result after 50 epochs with default setting. 

Note: PSNR/SSIM 

|   |BDS100   | SET14  | SET5  | 
|---|---|---|---|
| SRResnet  |  26.68/0.82 |  30.87/0.94 | 27.46/0.86  |
| SRGAN  | 26.07/0.81  | 29.36/0.92  | 26.49/0.85  |




## Training

* Git clone this repo
* Install requirements
* Create checkpoint folder inside 
* Prepocessing data
* Train and Inference


You can change the setting directly in train_SRResnet.py or train_SRGAN.py.

Run with default setting.


```
python train_SRResnet.py
python train_SRGAN.py
```

To inference 

```
python inference.py
```

## Prepocessing

I compressed dataset in .pkl file for training on google colab. Check notebook for more details.
Pkl saving format: List of np array.  

Download pkl file with demo [1000/300]:

