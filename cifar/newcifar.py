# -*- coding: utf-8 -*-
# 基于cnn的特定类别动物和载具识别（图像识别）
import os
import numpy as np
import pickle as p


def batch(filename): # 按批次读取
    with open(filename, 'rb') as f:
        batch_dict = p.load(f, encoding='bytes')
        fimage = batch_dict[b'data']
        flabel = batch_dict[b'labels']
        fimage = fimage.reshape(10000, 3, 32, 32)
        fimage = fimage.transpose(0, 2, 3, 1)
        flabel = np.array(flabel)
        return fimage, flabel


def data(data_dir):  # 读取全部数据
    fimage_train = []
    flabel_train = []
    for i in range(5):
        f = os.path.join(data_dir, 'data_batch_%d' % (i + 1))
        print('loading', f)
        image_batch, label_batch = batch(f)
        fimage_train.append(image_batch)
        flabel_train.append(label_batch)
        Xtrain = np.concatenate(fimage_train)  # 数组拼接，传入参数为一个或多个数组的元组或列表
        Ytrain = np.concatenate(flabel_train)
        del image_batch, label_batch  # 删除变量，但不删除数据
    Xtest, Ytest = batch(os.path.join(data_dir, 'test_batch'))
    print('finished loadding CIFAR-10 data')
    return Xtrain, Ytrain, Xtest, Ytest


data_dir = 'CIFAR-10_data/cifar-10-batches-py'
Xtrain, Ytrain, Xtest, Ytest = data(data_dir)
print('training data shape:', Xtrain.shape)  # 显示数据信息
print('training labels shape:', Ytrain.shape)
print('test data shape:', Xtest.shape)
print('test labels shape:', Ytest.shape)

import matplotlib.pyplot as plt

plt.imshow(Xtrain[6])
print(Ytrain[6])
label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer(鹿)", 5: "dog", 6: "frog(青蛙)", 7: "horse",
              8: "ship", 9: "truck(货车)"}


def plot_all(fimage, flabel, prediction, id, num):  # 显示图像、标签函数
    idp = 0
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    for i in range(0, num):
        ax = plt.subplot(3, 6, 1 + i)
        ax.imshow(fimage[id], cmap='binary')
        title = str(i) + ',' + label_dict[flabel[id]]
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[idp]]
        ax.set_title(title, fontsize=10)
        id += 1
        idp += 1
    plt.show()


plot_all(Xtest, Ytest, [], 777, 18)

# 进行数字标准化（以[0，1]小数表示）
Xtrain_normalize = Xtrain.astype('float32') / 255.0
Xtest_normalize = Xtest.astype('float32') / 255.0

#print(Xtest_normalize[1])

from sklearn.preprocessing import OneHotEncoder  # 独热编码（One-Hot）

encoder = OneHotEncoder(sparse=False)
OH = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
encoder.fit(OH)
Ytrain_reshape = Ytrain.reshape(-1, 1)
Ytrain_onehot = encoder.transform(Ytrain_reshape)
Ytest_reshape = Ytest.reshape(-1, 1)
Ytest_onehot = encoder.transform(Ytest_reshape)

import tensorflow as tf

tf.reset_default_graph()


def weight(shape):  # 定义权值
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')


def bias(shape):  # 定义偏置
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')


def conv2d(x, W):  # 定义卷积操作，步长为1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):  # 定义池化，步长为2
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope('input_layer'):  # 输入层
    x = tf.placeholder('float', shape=[None, 32, 32, 3], name='x')

with tf.name_scope('conv_1'):  # 卷积层1
    W1 = weight([3, 3, 3, 32])
    b1 = bias([32])
    conv_1 = conv2d(x, W1) + b1
    conv_1 = tf.nn.relu(conv_1)

with tf.name_scope('pool_1'):  # 池化层1
    pool_1 = max_pool_2x2(conv_1)

with tf.name_scope('conv_2'):  # 卷积层2
    W2 = weight([3, 3, 32, 64])
    b2 = bias([64])
    conv_2 = conv2d(pool_1, W2) + b2
    conv_2 = tf.nn.relu(conv_2)

with tf.name_scope('pool_2'):  # 池化层2
    pool_2 = max_pool_2x2(conv_2)

with tf.name_scope('fc'):  # 全连接层
    W3 = weight([4096, 256])
    b3 = bias([256])
    flat = tf.reshape(pool_2, [-1, 4096])
    h = tf.nn.relu(tf.matmul(flat, W3) + b3)
    h_dropout = tf.nn.dropout(h, keep_prob=0.8)

with tf.name_scope('output_layer'):  # 输出层
    W4 = weight([256, 10])
    b4 = bias([10])
    pred = tf.nn.softmax(tf.matmul(h_dropout, W4) + b4)

with tf.name_scope('optimizer'):
    y = tf.placeholder('float', shape=[None, 10], name='label')
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)

with tf.name_scope("evaluation"):  # 准确率
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

from time import time

train_epochs = 8  # 轮数
batch_size = 50  # 一批大小
total_batch = int(len(Xtrain) / batch_size)  # 批次
epoch = tf.Variable(0, name='epoch', trainable=False)
epoch_list = []
accuracy_list = []
loss_list = []
startTime = time()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

ckpt_dir = "CIFAR10-log/"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
saver = tf.train.Saver(max_to_keep=1)
ckpt = tf.train.latest_checkpoint(ckpt_dir)  # 检查点文件
if ckpt != None:
    saver.restore(sess, ckpt)  # 加载所有参数
else:
    print("Training from scratch.")
start = sess.run(epoch)
print("Training start from {} epoch.".format(start + 1))  # 从第xx轮开始


def get_train_batch(number, batch_size):  # 获得训练数据
    return Xtrain_normalize[number * batch_size:(number + 1) * batch_size], Ytrain_onehot[number * batch_size:(number + 1) * batch_size]


for ep in range(start, train_epochs):
    for i in range(total_batch):
        batch_x, batch_y = get_train_batch(i, batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if i % 100 == 0:
            print("step {}".format(i), " finished")
    loss, acc = sess.run([loss_function, accuracy], feed_dict={x: batch_x, y: batch_y})
    epoch_list.append(ep + 1)
    loss_list.append(loss)
    accuracy_list.append(acc)
    print("Train epoch: ", '%02d' % (sess.run(epoch) + 1), "Loss=", "{:.6f}".format(loss), "Accuracy=", acc)
    saver.save(sess, ckpt_dir + "CIFAR10_cnn_model.cpkt", global_step=ep + 1)  # 保存
    sess.run(epoch.assign(ep + 1))
duration = time() - startTime
print("Train finished takes: ", duration)  # 训练时长

import matplotlib.pyplot as plt
# 对训练准确率进行图表绘制
fig = plt.gcf()
fig.set_size_inches(4, 2)
plt.plot(epoch_list, loss_list, label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
plt.plot(epoch_list, accuracy_list, label="accuracy")
fig = plt.gcf()
fig.set_size_inches(4, 2)
plt.ylim(0.1, 1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

# 计算测试集上准确率
test_total_batch = int(len(Xtest_normalize) / batch_size)
test_acc_sum = 0.0
for i in range(test_total_batch):
    test_image_batch = Xtest_normalize[i * batch_size:(i + 1) * batch_size]
    test_label_batch = Ytest_onehot[i * batch_size:(i + 1) * batch_size]
    test_batch_acc = sess.run(accuracy, feed_dict={x: test_image_batch, y: test_label_batch})
    test_acc_sum += test_batch_acc
test_acc = float(test_acc_sum / test_total_batch)
print("Test accuracy:","{:.6f}".format(test_acc))

#预测
test_pred = sess.run(pred, feed_dict={x: Xtest_normalize[7:12]})
prediction_result = sess.run(tf.argmax(test_pred, 1))
plot_all(Xtest, Ytest, prediction_result, 7, 5)

def prediction_inall(img):
    test = sess.run(pred, feed_dict={x:img})
    prediction_rst = sess.run(tf.argmax(test,1))
    return prediction_rst # 得到对应的种类编号（0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer(鹿)", 5: "dog", 6: "frog(青蛙)", 7: "horse",8: "ship", 9: "truck(货车)"）
