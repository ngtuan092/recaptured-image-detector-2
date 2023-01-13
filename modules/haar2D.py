import pywt
import torch

# 1D Haar Wavelet Transform


def haar1D(x, length):
    if length == 1:
        return x
    else:

        length = int(length / 2)
        x[0:length] = (x[0:length] + x[length:2*length]) / 2
        x[length:2*length] = (x[0:length] - x[length:2*length]) / 2
        return x


def splitFreqBand(img, rows, cols):
    halfRows = int(rows / 2)
    halfCols = int(cols / 2)
    LL = img[:, 0:halfRows, 0:halfCols].unsqueeze(0)
    LH = img[:, 0:halfRows, halfCols:cols].unsqueeze(0)
    HL = img[:, halfRows:rows, 0:halfCols].unsqueeze(0)
    HH = img[:, halfRows:rows, halfCols:cols].unsqueeze(0)

    return LL, LH, HL, HH


class WaveletTransform(object):
    def __call__(self, img):
        rows, cols = img.shape[1], img.shape[2]
        # 1D Haar Wavelet Transform
        for i in range(rows):
            img[0, i, :] = haar1D(img[0, i, :], cols)

        for i in range(cols):
            img[0, :, i] = haar1D(img[0, :, i], rows)

        # Split Frequency Bands
        LL, LH, HL, HH = splitFreqBand(img, rows, cols)
        return LL, LH, HL, HH


class WaveletTransform2(object):
    def __call__(self, img):
        ll, (lh, hl, hh) = pywt.dwt2(img, 'haar')
        return ll, lh, hl, hh


if __name__ == '__main__':
    print('Testing Haar Wavelet Transform')
    img = torch.randn(1, 4, 4)
    LL, LH, HL, HH = WaveletTransform()(img)
    print(LL.shape, LH.shape, HL.shape, HH.shape)
    ll, (lh, hl, hh) = pywt.dwt2(img.numpy(), 'haar')
    print(ll.shape, lh.shape, hl.shape, hh.shape)
    ll = torch.from_numpy(ll)
    print(ll)
    print(LL)
    print('Testing Done')
    from PIL import Image
    import numpy as np
    Image.open('data/processed/0/0.png')