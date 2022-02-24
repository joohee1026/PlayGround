## Summary Results

No.|model|Public Score|Private Score|
----|----|----|----|
1|Xception ensemble (3 model)|0.90291|0.89948|
2|EffNetB4 ensemble (6 model)|0.94691|0.93914|
3|No.2 + No.3 ensemble (9 model)|0.94787|0.94136|
4|EffNetB5 + Catboost|0.91582|0.90785|
5|EffNetB5 + ABN|0.92395|0.91130|
6|EffNetB4, B5 (19 model)|0.94787|**0.94161** |  

#
-----

## Common Details
- Input size : 300x300x3
- If the f-score did not improve for 10 epochs, stop learning.
- If the f-score did not improve for 2 epochs, cut in half learning rate.
- Use pre-trained weight in ImageNet
- Global Average Pooling the output of the last conv layer, and encode using fully-connected layer in 512 dims.
- Data augmentation setting (ImageDataGenerator in keras)
~~~
ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 90,
            width_shift_range = 20,
            height_shift_range = 20,
            brightness_range = [0.5, 1.5],
            fill_mode = 'nearest',
            zoom_range = 0.1,
            shear_range = 0.1,
            horizontal_flip = True,
            vertical_flip = True
            )
~~~

### EfficientNetB4, B5
- Originally input size of EffNetB4, B5 is 380x380, 456x456 respectively. --> use 300x300 size due to GPU limitations. 
- If don't use pre-trained weights, the loss does not converge. --> may be due to use 4 batch

#
### use CatBoost
- Applying GAP to last Conv layer output of EffNetB5 model, obtain a 2048 dimensional feature vector.
- Use Catboost to this vector, doesn't get a good results --> worth a try to hyperparameter tuning or other ML models.

#
### Attention Branch Network (ABN)
<img src="https://github.com/SSinyu/Deeprison/blob/master/Kaggle/3rd_kakr/ABN.PNG" height="300">

- Structure that better image classification performance by using an attention map obtained from the learning process.
- Shallow ABN based on ResNet doesn't get a good results (single model)
