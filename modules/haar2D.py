import pywt
import torch
import numpy as np
# 1D Haar Wavelet Transform


def haar1D(x, length):
    if length == 1:
        return x
    else:
        halfLength = int(length / 2)
        temp = torch.empty_like(x)
        for i in range(halfLength):
            temp[i] = (x[2 * i] + x[2 * i + 1]) / 2
            temp[i + halfLength] = (x[2 * i] - x[2 * i + 1]) / 2
        return temp


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


def splitFreqBands(img, levRows, levCols):
    halfRow = int(levRows/2)
    halfCol = int(levCols/2)
    LL = img[0:halfRow, 0:halfCol]
    LH = img[0:halfRow, halfCol:levCols]
    HL = img[halfRow:levRows, 0:halfCol]
    HH = img[halfRow:levRows, halfCol:levCols]

    return LL, LH, HL, HH


def haarDWT1D(data, length):
    avg0 = 0.5
    avg1 = 0.5
    dif0 = 0.5
    dif1 = -0.5
    temp = np.empty_like(data)
    temp = temp.astype(float)
    h = int(length/2)
    for i in range(h):
        k = i*2
        temp[i] = data[k] * avg0 + data[k + 1] * avg1
        temp[i + h] = data[k] * dif0 + data[k + 1] * dif1

    data[:] = temp

# computes the homography coefficients for PIL.Image.transform using point correspondences


def fwdHaarDWT2D(img):
    img = np.array(img)
    levRows = img.shape[0]
    levCols = img.shape[1]
    img = img.astype(float)
    for i in range(levRows):
        row = img[i, :]
        haarDWT1D(row, levCols)
        img[i, :] = row
    for j in range(levCols):
        col = img[:, j]
        haarDWT1D(col, levRows)
        img[:, j] = col

    return splitFreqBands(img, levRows, levCols)


if __name__ == '__main__':
    print('Testing Haar Wavelet Transform')
    img = torch.randn(1, 4, 4)
    img1 = img.clone()
    img2 = img.clone()
    LL, LH, HL, HH = WaveletTransform()(img)
    print(LL.shape, LH.shape, HL.shape, HH.shape)
    ll, (lh, hl, hh) = pywt.dwt2(img1.numpy(), 'haar')
    print(ll.shape, lh.shape, hl.shape, hh.shape)
    ll = torch.from_numpy(ll)
    print('Testing Done')
    Ll, Lh, Hl, Hh = fwdHaarDWT2D(img2.numpy()[0])
    print(Ll.shape, Lh.shape, Hl.shape, Hh.shape)
    print(ll)
    print(LL)
    print(Ll)