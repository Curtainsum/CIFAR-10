import pickle
import  numpy as np
import os
# 目录
CIFAR_DIR = "C:/Users/summer/PycharmProjects/cifar/CIFAR-10_data/cifar-10-batches-py"

print('1----->>',os.listdir(CIFAR_DIR))
'''
输出:
    ['batches.meta', 'test_batch', 'data_batch_5', 'data_batch_3', 'data_batch_2', 'data_batch_4', 'readme.html', 'data_batch_1']
这是目录下的所有文件
'''

with open(os.path.join(CIFAR_DIR, 'data_batch_1'), 'rb') as f:
    data = pickle.load(f, encoding='bytes') # python3 需要添加 encoding='bytes'
    print('2----->>',type(data)) # 输出 <class 'dict'>
    print('3----->>',data.keys()) # 输出 dict_keys([b'filenames', b'data', b'labels', b'batch_label'])

    print('4----->>',type(data[b'data'])) # 输出 <class 'numpy.ndarray'>
    print('5----->>',data[b'data'].shape) # 输出 (10000, 3072) 说明有 10000 个样本, 3072个特征

    print(data[b'data'][1])
    #print(data[b'labels'])
    '''
        [[ 59  43  50 ... 140  84  72]
         [154 126 105 ... 139 142 144]
         [255 253 253 ...  83  83  84]]
         二维矩阵,每一行是一个特征
        [6, 9] 标签值
        6代表第六个分类,9代表第九个分类
    '''
# 一个名字对应矩阵的一行

# 现在将一个行向量转化为一张图片,看看样子
image_arr = data[b'data'][22] # 拿出 第 101 个样本
image_arr = image_arr.reshape((3, 32, 32)) # 将一个一维向量改变形状,目的是得到这样一个元组:(高,宽,通道数)
image_arr = image_arr.transpose((1, 2, 0)) # 将上一行的元组变为我们需要的样子,第一个元素放到最后
import matplotlib.pyplot as plt
plt.imshow(image_arr) # 将图片输出
plt.show()
