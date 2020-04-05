"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle

root_dir = r"/Users/lulei05/Documents/study/deep_learning/common_data/第4章 配套资源/4-4  将excel文件制作成内存对象数据集/"

def image_demo():
    def load_sample(sample_dir):
        '''递归读取文件。只支持一级。返回文件名、数值标签、数值对应的标签名'''
        print ('loading sample  dataset..')
        lfilenames = []
        labelsnames = []
        for (dirpath, dirnames, filenames) in os.walk(sample_dir):#递归遍历文件夹
            for filename in filenames:                            #遍历所有文件名
                #print(dirnames)
                filename_path = os.sep.join([dirpath, filename])
                lfilenames.append(filename_path)               #添加文件名
                labelsnames.append( dirpath.split('/')[-1] )#添加文件名对应的标签

        lab= list(sorted(set(labelsnames)))  #生成标签名称列表
        labdict=dict( zip( lab  ,list(range(len(lab)))  )) #生成字典

        labels = [labdict[i] for i in labelsnames]
        return shuffle(np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)


    data_dir = root_dir + 'mnist_digits_images/'  #定义文件路径

    (image,label),labelsnames = load_sample(data_dir)   #载入文件名称与标签
    print(len(image),image[:2],len(label),label[:2])#输出load_sample返回的数据结果
    print(labelsnames[ label[:2] ],labelsnames)#输出load_sample返回的标签字符串


    def get_batches(image,label,input_w,input_h,channels,batch_size):

        queue = tf.train.slice_input_producer([image,label])  #使用tf.train.slice_input_producer实现一个输入的队列
        label = queue[1]                                        #从输入队列里读取标签

        image_c = tf.read_file(queue[0])                        #从输入队列里读取image路径

        image = tf.image.decode_bmp(image_c,channels)           #按照路径读取图片

        image = tf.image.resize_image_with_crop_or_pad(image,input_w,input_h) #修改图片大小


        image = tf.image.per_image_standardization(image) #图像标准化处理，(x - mean) / adjusted_stddev

        image_batch,label_batch = tf.train.batch([image,label],#调用tf.train.batch函数生成批次数据
                   batch_size = batch_size,
                   num_threads = 64)

        images_batch = tf.cast(image_batch,tf.float32)   #将数据类型转换为float32

        labels_batch = tf.reshape(label_batch,[batch_size])#修改标签的形状shape
        return images_batch,labels_batch


    batch_size = 16
    image_batches,label_batches = get_batches(image,label,28,28,1,batch_size)


    def showresult(subplot,title,thisimg):          #显示单个图片
        p =plt.subplot(subplot)
        p.axis('off')
        #p.imshow(np.asarray(thisimg[0], dtype='uint8'))
        p.imshow(np.reshape(thisimg, (28, 28)))
        p.set_title(title)

    def showimg(index,label,img,ntop):   #显示
        plt.figure(figsize=(20,10))     #定义显示图片的宽、高
        plt.axis('off')
        ntop = min(ntop,9)
        print(index)
        for i in range (ntop):
            showresult(100+10*ntop+1+i,label[i],img[i])
        plt.show()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)  #初始化

        coord = tf.train.Coordinator()          #开启列队
        threads = tf.train.start_queue_runners(sess = sess,coord = coord)
        try:
            for step in np.arange(10):
                if coord.should_stop():
                    break
                images,label = sess.run([image_batches,label_batches]) #注入数据

                showimg(step,label,images,batch_size)       #显示图片
                print(label)                                 #打印数据

        except tf.errors.OutOfRangeError:
            print("Done!!!")
        finally:
            coord.request_stop()

        coord.join(threads)                             #关闭列队


def csv_demo():
    def read_data(file_queue):  # csv文件处理函数
        reader = tf.TextLineReader(skip_header_lines=1)  # tf.TextLineReader 可以每次读取一行
        key, value = reader.read(file_queue)

        defaults = [[0], [0.], [0.], [0.], [0.], [0]]  # 为每个字段设置初始值
        cvscolumn = tf.decode_csv(value, defaults)  # tf.decode_csv对每一行进行解析

        featurecolumn = [i for i in cvscolumn[1:-1]]  # 分离出列中的样本数据列
        labelcolumn = cvscolumn[-1]  # 分离出列中的标签数据列

        return tf.stack(featurecolumn), labelcolumn  # 返回结果

    def create_pipeline(filename, batch_size, num_epochs=None):  # 创建队列数据集函数
        # 创建一个输入队列
        file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

        feature, label = read_data(file_queue)  # 载入数据和标签

        min_after_dequeue = 1000  # 设置队列中的最少数据条数（取完数据后，保证还是有1000条）
        capacity = min_after_dequeue + batch_size  # 队列的长度

        feature_batch, label_batch = tf.train.shuffle_batch(  # 生成乱序的批次数据
            [feature, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue
        )

        return feature_batch, label_batch  # 返回指定批次数据

    # 读取训练集
    x_train_batch, y_train_batch = create_pipeline(root_dir + 'iris_training.csv', 32, num_epochs=1)
    # 读取测试集
    x_test, y_test = create_pipeline(root_dir + 'iris_test.csv', 32)

    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()  # 初始化
        local_init_op = tf.local_variables_initializer()  # 初始化本地变量，没有回报错
        sess.run(init_op)
        sess.run(local_init_op)

        coord = tf.train.Coordinator()  # 创建协调器
        threads = tf.train.start_queue_runners(coord=coord)  # 开启线程列队

        try:
            while True:
                if coord.should_stop():
                    break
                example, label = sess.run([x_train_batch, y_train_batch])  # 注入训练数据
                print("训练数据：", example)  # 打印数据
                print("训练标签：", label)  # 打印标签
        except tf.errors.OutOfRangeError:  # 定义取完数据的异常处理
            print('Done reading')
            example, label = sess.run([x_test, y_test])  # 注入测试数据
            print("测试数据：", example)  # 打印数据
            print("测试标签：", label)  # 打印标签
        except KeyboardInterrupt:  # 定义按ctrl+c键时，对应的异常处理
            print("程序终止...")
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


if __name__ == "__main__":
    csv_demo()