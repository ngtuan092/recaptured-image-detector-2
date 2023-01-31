# Moire Pattern Detector

When taking a picture of LCD screen, you may see a moire pattern. This is because the screen is made of a grid of pixels, and the camera is also made of a grid of pixels. When the camera is close to the screen, the two grids will interfere with each other and create a moire pattern.

This program detects the moire pattern in a picture.

## Usage
### Requirements
* Python 3
* Python Imaging Library (PIL)
* NumPy
* Torch
* Torchvision

### Training
```
python train.py
```
### Testing
```
python test.py
```
### Predicting
```
python predict.py
```