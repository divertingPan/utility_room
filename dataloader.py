# -*- coding: utf-8 -*-
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy
import matplotlib.pyplot as plt


def input_x(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((360, 360), Image.ANTIALIAS)
    compose = [transforms.RandomAffine(20, scale=(1, 2), shear=15, resample=Image.BILINEAR),
               transforms.RandomHorizontalFlip(p=0.5),
               transforms.RandomVerticalFlip(p=0.5),
               transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
               ]
    compose = transforms.Compose([transforms.RandomOrder(compose),
                                  transforms.RandomCrop(360, pad_if_needed=False),
                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                  ])
    image = compose(image)
    return image


def dataloader(attr, data_path):
    df = pd.read_csv(data_path)
    image_path = [attr + '/' + x for x in df['image_path']]
    label = [x for x in df['label']]
    return image_path, label


if __name__ == '__main__':
    
    train_attr = '训练集'
    train_path = '训练集/train.csv'
    val_attr = '测试集'
    val_path = '测试集/test.csv'
    test_attr = '评估数据集'
    test_path = '评估数据集/upload.csv'
    
    train_image_path, train_label = dataloader(train_attr, train_path)
    val_image_path, val_label = dataloader(val_attr, val_path)
    test_image_path, test_label = dataloader(test_attr, test_path)
    
    for i in range(5):
        image = input_x(train_image_path[i])
    
        image = numpy.array(image)
        image = numpy.transpose(image, (1, 0, 2))
        image = numpy.transpose(image, (0, 2, 1))
        
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        print(train_label[i])
    