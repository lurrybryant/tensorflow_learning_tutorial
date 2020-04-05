# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""



import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import *
import numpy as np
from tfrecord import *
from matplotlib import pyplot as plt

Dataset =tf.data.Dataset


def interleave_demo():

    def parse_fn(x):
        print(x)
        return x

    dataset = (Dataset.list_files('testset\*.txt', shuffle=False)
                   .interleave(lambda x:
                       tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
                       cycle_length=2, block_length=2))



    def getone(dataset):
        iterator = dataset.make_one_shot_iterator()			#生成一个迭代器
        one_element = iterator.get_next()					#从iterator里取出一个元素
        return one_element

    one_element1 = getone(dataset)				#从dataset里取出一个元素


    def showone(one_element,datasetname):
        print('{0:-^50}'.format(datasetname))
        for ii in range(20):
            datav = sess.run(one_element)#通过静态图注入的方式，传入数据
            print(datav)



    with tf.Session() as sess:	# 建立会话（session）
        showone(one_element1,"dataset1")



###############  range(*args)


'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.range(5)
iterator = dataset.make_one_shot_iterator()			#从到到尾读一次
one_element = iterator.get_next()					#从iterator里取出一个元素
with tf.Session() as sess:	# 建立会话（session）

    for i in range(5):		#通过for循环打印所有的数据
        print(sess.run(one_element))				#调用sess.run读出Tensor值
'''



###############  zip(datasets)



'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset2 = tf.data.Dataset.from_tensor_slices(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]))
dataset = Dataset.zip((dataset1,dataset2))
iterator = dataset.make_one_shot_iterator()			#从到到尾读一次
one_element = iterator.get_next()					#从iterator里取出一个元素
with tf.Session() as sess:	# 建立会话（session）

    for i in range(5):		#通过for循环打印所有的数据
        print(sess.run(one_element))				#调用sess.run读出Tensor值
'''



###############  concatenate(dataset)



'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset2 = tf.data.Dataset.from_tensor_slices(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]))
dataset = dataset1.concatenate(dataset2)
iterator = dataset.make_one_shot_iterator()			#从到到尾读一次
one_element = iterator.get_next()					#从iterator里取出一个元素
with tf.Session() as sess:	# 建立会话（session）

    for i in range(10):		#通过for循环打印所有的数据
        print(sess.run(one_element))				#调用sess.run读出Tensor值
'''



###############  repeat(count=None)



'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset1.repeat(2)
iterator = dataset.make_one_shot_iterator()			#从到到尾读一次
one_element = iterator.get_next()					#从iterator里取出一个元素
with tf.Session() as sess:	# 建立会话（session）

    for i in range(10):		#通过for循环打印所有的数据
        print(sess.run(one_element))				#调用sess.run读出Tensor值
'''


###############  shuffle(buffer_size,seed=None,reshuffle_each_iteration=None)


'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset1.shuffle(1000)
iterator = dataset.make_one_shot_iterator()			#从到到尾读一次
one_element = iterator.get_next()					#从iterator里取出一个元素
with tf.Session() as sess:	# 建立会话（session）

    for i in range(5):		#通过for循环打印所有的数据
        print(sess.run(one_element))				#调用sess.run读出Tensor值

'''




###############  batch(count=None)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.batch(batch_size=2)
iterator = dataset.make_one_shot_iterator()			#从到到尾读一次
one_element = iterator.get_next()					#从iterator里取出一个元素
with tf.Session() as sess:	# 建立会话（session）
	while True:
	    for i in range(2):		#通过for循环打印所有的数据
	        print(sess.run(one_element))				#调用sess.run读出Tensor值
'''

###############  padded_batch

'''
data1 = tf.data.Dataset.from_tensor_slices([[1, 2],[1,3]])
data1 = data1.padded_batch(2,padded_shapes=[4])
iterator = data1.make_initializable_iterator()
next_element = iterator.get_next()
init_op = iterator.initializer

with tf.Session() as sess2:
    print(sess2.run(init_op))
    print("batched data 1:",sess2.run(next_element))
'''

###############  flat_map(map_func)




'''
import numpy as np

##在内存中生成数据
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = tf.data.Dataset.from_tensor_slices(np.array([[1,2,3],[4,5,6]]))

dataset = dataset.flat_map(lambda x: Dataset.from_tensors(x)) 			
iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(10):							#通过for循环打印所有的数据
        print(sess.run(one_element))			#调用sess.run读出Tensor值
'''



######interleave(map_func,cycle_length,block_length=1)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.interleave(lambda x: Dataset.from_tensors(x).repeat(3),
             cycle_length=2, block_length=2)			
iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(100):							#通过for循环打印所有的数据
        print(sess.run(one_element),end=' ')			#调用sess.run读出Tensor值
'''

######filter(predicate)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.filter(lambda x: tf.less(x, 3))			
iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(100):							#通过for循环打印所有的数据
        print(sess.run(one_element),end=' ')			#调用sess.run读出Tensor值

#过滤掉全为0的元素
dataset = tf.data.Dataset.from_tensor_slices([ [0, 0],[ 3.0, 4.0] ])
dataset = dataset.filter(lambda x: tf.greater(tf.reduce_sum(x), 0))		  #过滤掉全为0的元素	
iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(100):							#通过for循环打印所有的数据
        print(sess.run(one_element),end=' ')			#调用sess.run读出Tensor值

#过滤掉中文字符串(1)加入一个判断列
dataset = tf.data.Dataset.from_tensor_slices([ "hello","niha好" ])

def _parse_data(line):
    def checkone(line):
        for ch in line:
            #print(line,ch)
            if ch<23 or ch>127:
                return False
        return True
    isokstr = tf.py_func( checkone, [line], tf.bool)
    #tf.cast(isokstr,tf.bool)[0]

    return line,isokstr#tf.cast(isokstr,tf.bool)[0]
dataset = dataset.map(_parse_data)

dataset = dataset.filter(lambda x,y: y)		  #过滤掉全为0的元素	
iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(100):							#通过for循环打印所有的数据
        print(sess.run(one_element),end=' ')			#调用sess.run读出Tensor值

#过滤掉中文字符串(2)简单实现
dataset = tf.data.Dataset.from_tensor_slices([ "hello","niha好" ])

def myfilter(x):
    def checkone(line):
        for ch in line:
            #print(line,ch)
            if ch<23 or ch>127:
                return False
        return True
    isokstr = tf.py_func( checkone, [x], tf.bool)
    return isokstr
dataset = dataset.filter(myfilter)		  #过滤掉全为0的元素	
#dataset = dataset.filter(lambda x,y: y)		  #过滤掉全为0的元素	
iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(100):							#通过for循环打印所有的数据
        print(sess.run(one_element),end=' ')			#调用sess.run读出Tensor值

'''
######apply(transformation_func)
'''
data1 = np.arange(50).astype(np.int64)
dataset = tf.data.Dataset.from_tensor_slices(data1)
#将数据集中偶数行与奇数行分开，以window_size为窗口大小，一次取window_size个偶数行和window_size个奇数行。在window_size中，以batch为批次进行分割。
dataset = dataset.apply((tf.contrib.data.group_by_window(key_func=lambda x: x%2, reduce_func=lambda _, els: els.batch(10), window_size=20)  ))

iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(100):							#通过for循环打印所有的数据
        print(sess.run(one_element),end=' ')			#调用sess.run读出Tensor值
'''

def test_demo01():
    # 在内存中生成模拟数据
    def GenerateData(datasize=100):
        train_X = np.linspace(-1, 1, datasize)  # train_X为-1到1之间连续的100个浮点数
        train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3  # y=2x，但是加入了噪声
        return train_X, train_Y  # 以生成器的方式返回

    train_data = GenerateData()

    # 将内存数据转化成数据集
    dataset = tf.data.Dataset.from_tensor_slices(train_data)  # 元祖
    dataset2 = tf.data.Dataset.from_tensor_slices({  # 字典
        "x": train_data[0],
        "y": train_data[1]
    })  #

    batchsize = 10  # 定义批次样本个数
    dataset3 = dataset.repeat().batch(batchsize)  # 批次划分数据集

    dataset4 = dataset2.map(lambda data: (data['x'], tf.cast(data['y'], tf.int32)))  # 自定义处理数据集元素
    dataset5 = dataset.shuffle(100)  # 乱序数据集

    def getone(dataset):
        iterator = dataset.make_one_shot_iterator()  # 生成一个迭代器
        one_element = iterator.get_next()  # 从iterator里取出一个元素
        return one_element

    one_element1 = getone(dataset)  # 从dataset里取出一个元素
    one_element2 = getone(dataset2)  # 从dataset2里取出一个元素
    one_element3 = getone(dataset3)  # 从dataset3里取出一个批次的元素
    one_element4 = getone(dataset4)  # 从dataset4里取出一个批次的元素
    one_element5 = getone(dataset5)  # 从dataset5里取出一个批次的元素

    def showone(one_element, datasetname):
        print('{0:-^50}'.format(datasetname))
        for ii in range(5):
            datav = sess.run(one_element)  # 通过静态图注入的方式，传入数据
            print(datasetname, "-", ii, "| x,y:", datav)

    def showbatch(onebatch_element, datasetname):
        print('{0:-^50}'.format(datasetname))
        for ii in range(5):
            datav = sess.run(onebatch_element)  # 通过静态图注入的方式，传入数据
            print(datasetname, "-", ii, "| x.shape:", np.shape(datav[0]), "| x[:3]:", datav[0][:3])
            print(datasetname, "-", ii, "| y.shape:", np.shape(datav[1]), "| y[:3]:", datav[1][:3])

    with tf.Session() as sess:  # 建立会话（session）
        showone(one_element1, "dataset1")
        showone(one_element2, "dataset2")
        showbatch(one_element3, "dataset3")
        showone(one_element4, "dataset4")
        showone(one_element5, "dataset5")


def img_demo():
    def _distorted_image(image, size, ch=1, shuffleflag=False, cropflag=False,
                         brightnessflag=False, contrastflag=False):  # 定义函数，实现变化图片
        distorted_image = tf.image.random_flip_left_right(image)

        if cropflag == True:  # 随机裁剪
            s = tf.random_uniform((1, 2), int(size[0] * 0.8), size[0], tf.int32)
            distorted_image = tf.random_crop(distorted_image, [s[0][0], s[0][0], ch])

        distorted_image = tf.image.random_flip_up_down(distorted_image)  # 上下随机翻转
        if brightnessflag == True:  # 随机变化亮度
            distorted_image = tf.image.random_brightness(distorted_image, max_delta=10)
        if contrastflag == True:  # 随机变化对比度
            distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        if shuffleflag == True:
            distorted_image = tf.random_shuffle(distorted_image)  # 沿着第0维乱序
        return distorted_image

    def _norm_image(image, size, ch=1, flattenflag=False):  # 定义函数，实现归一化，并且拍平
        image_decoded = image / 255.0
        if flattenflag == True:
            image_decoded = tf.reshape(image_decoded, [size[0] * size[1] * ch])
        return image_decoded

    from skimage import transform
    def _random_rotated30(image, label):  # 定义函数实现图片随机旋转操作

        def _rotated(image):  # 封装好的skimage模块，来进行图片旋转30度
            shift_y, shift_x = np.array(image.shape.as_list()[:2], np.float32) / 2.
            tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(30))
            tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
            tf_shift_inv, image.size = transform.SimilarityTransform(
                translation=[shift_x, shift_y]), image.shape  # 兼容transform函数
            image_rotated = transform.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
            return image_rotated

        def _rotatedwrap():
            image_rotated = tf.py_function(_rotated, [image], [tf.float64])  # 调用第三方函数
            return tf.cast(image_rotated, tf.float32)[0]

        a = tf.random_uniform([1], 0, 2, tf.int32)  # 实现随机功能
        image_decoded = tf.cond(tf.equal(tf.constant(0), a[0]), lambda: image, _rotatedwrap)

        return image_decoded, label

    def dataset(directory, size, batchsize, random_rotated=False):  # 定义函数，创建数据集
        """ parse  dataset."""
        (filenames, labels), _ = load_sample(directory, shuffleflag=False)  # 载入文件名称与标签
        # print(filenames)

        def _parseone(filename, label):  # 解析一个图片文件
            """ Reading and handle  image"""
            image_string = tf.read_file(filename)  # 读取整个文件

            image_decoded = tf.image.decode_image(image_string)
            image_decoded.set_shape([None, None, None])  # 必须有这句，不然下面会转化失败
            image_decoded = _distorted_image(image_decoded, size)  # 对图片做扭曲变化
            image_decoded = tf.image.resize(image_decoded, size)  # 变化尺寸
            image_decoded = _norm_image(image_decoded, size)  # 归一化
            image_decoded = tf.cast(image_decoded, dtype=tf.float32)
            label = tf.cast(tf.reshape(label, []), dtype=tf.int32)  # 将label 转为张量
            return image_decoded, label

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))  # 生成Dataset对象
        dataset = dataset.map(_parseone)  # 有图片内容的数据集

        if random_rotated == True:
            dataset = dataset.map(_random_rotated30)

        dataset = dataset.batch(batchsize)  # 批次划分数据集

        return dataset

    # 如果显示有错，可以尝试使用np.reshape(thisimg, (size[0],size[1],3))或
    # np.asarray(thisimg[0], dtype='uint8')改变类型与形状
    def showresult(subplot, title, thisimg):  # 显示单个图片
        p = plt.subplot(subplot)
        p.axis('off')
        p.imshow(thisimg)
        p.set_title(title)

    def showimg(index, label, img, ntop):  # 显示
        plt.figure(figsize=(20, 10))  # 定义显示图片的宽、高
        plt.axis('off')
        ntop = min(ntop, 9)
        print(index)
        for i in range(ntop):
            showresult(100 + 10 * ntop + 1 + i, label[i], img[i])
        plt.show()

    def getone(dataset):
        iterator = dataset.make_one_shot_iterator()  # 生成一个迭代器
        one_element = iterator.get_next()  # 从iterator里取出一个元素
        return one_element

    sample_dir = directory
    size = [96, 96]
    batchsize = 10
    tdataset = dataset(sample_dir, size, batchsize)
    tdataset2 = dataset(sample_dir, size, batchsize, True)
    print(tdataset.output_types)  # 打印数据集的输出信息
    print(tdataset.output_shapes)

    one_element1 = getone(tdataset)  # 从tdataset里取出一个元素
    one_element2 = getone(tdataset2)  # 从tdataset2里取出一个元素

    with tf.Session() as sess:  # 建立会话（session）
        sess.run(tf.global_variables_initializer())  # 初始化

        try:
            for step in np.arange(1):
                value = sess.run(one_element1)
                value2 = sess.run(one_element2)

                showimg(step, value[1], np.asarray(value[0] * 255, np.uint8), 10)  # 显示图片
                showimg(step, value2[1], np.asarray(value2[0] * 255, np.uint8), 10)  # 显示图片


        except tf.errors.OutOfRangeError:  # 捕获异常
            print("Done!!!")


if __name__ == "__main__":
    # interleave_demo()
    # test_demo01()
    img_demo()