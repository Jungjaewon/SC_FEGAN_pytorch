# SC_FEGAN_pytorch
This repository is an unofficial implementation of [SC-FEGAN](https://github.com/run-youngjoo/SC-FEGAN) by pytorch which published in ICCV 2019. 

## Requirements
* python3.7+
* pytorch 1.7.0+
* others.

## Usage
training a model
```bash
python3 main.py --config config.yml
```

testing a model
```bash
Not implmented yet
```

## Results

### Train Images
![train_image](images/030-images.jpg)

### Test Images
![test_image1](images/961.jpg)
![test_image2](images/972.jpg)
![test_image3](images/3039.jpg)
![test_image4](images/3040.jpg)

## License
Attribution-NonCommercial-ShareAlike 4.0 International

## Reference
1. [SE-FEGAN](https://github.com/run-youngjoo/SC-FEGAN)
2. [DeepFillv2](https://github.com/zhaoyuzhi/deepfillv2)
3. [Spectral Normalization](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py)
4. [Dataset](https://github.com/switchablenorms/CelebAMask-HQ)

## To do
- [ ] upgrade training code
- [ ] upgrade color domain image and training

## Color Data by super pixel
<img src='images/face_color_250.jpg' width='256'>