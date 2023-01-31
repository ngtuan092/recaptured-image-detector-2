from torchvision import transforms
from torchvision.datasets import ImageFolder

from modules.haar2D import WaveletTransform2

MoireDataset = ImageFolder(root='./data_ver1', transform=transforms.Compose([
    transforms.Resize((1024, 768)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Grayscale(num_output_channels=1),
    WaveletTransform2(),
]))

if __name__ == '__main__':
    print(MoireDataset[0][0][0].shape)
