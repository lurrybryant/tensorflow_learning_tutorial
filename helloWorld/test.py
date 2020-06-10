# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def test0():
    x = tf.constant([[1, 4],
                    [2,3],
                     [2,3]])
    # m = x[:,0]
    # t = tf.reshape(m, (-1,1))
    y = tf.constant([2, 2, 2])
    # y = tf.reshape(y, (-1,1))
    h = tf.stack([x[:,0], y])
    z = tf.transpose(h)

    # 初始化变量
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


    # 启动图 (graph)
    sess = tf.Session()
    sess.run(init)

    print(sess.run([h, z]))


def test1():
    from sklearn.metrics import auc
    la, pr = [0,0,0,1,1,1,1], [0.2,0.3,0.8,0.3,0.6,0.7,0.8]
    label = tf.constant(la)
    print(auc(la, pr))
    predict = tf.constant(pr)
    auc1 = tf.metrics.auc(label, predict)

    # 初始化变量
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # 启动图 (graph)
    sess = tf.Session()
    sess.run(init)
    print(sess.run(auc1))

def test02():
    one_element = {"ft": np.array([2,3,4,5])}
        # tf.constant([2,3,4,5])

    # numeric_feature_column = tf.feature_column.numeric_column(key="ft", default_value=0.0)
    # bucketized_feature_column = tf.feature_column.bucketized_column(source_column=numeric_feature_column,
    #                                                                 boundaries=[1960])
    # x = tf.feature_column.input_layer(one_element, bucketized_feature_column)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        z = sess.run(x)
        print(z)


def test03():
    colors = {'colors': ['green', 'red', 'blue', 'yellow', 'pink', 'blue', 'red', 'indigo']}
    colors2 = {'colors': ['red', 'blue', 'yellow', 'pink', 'blue', 'red', 'indigo', 'for']}

    column = tf.feature_column.categorical_column_with_hash_bucket(
        key='colors',
        hash_bucket_size=5,
    )

    indicator = tf.feature_column.indicator_column(column)
    tensor_x = tf.feature_column.input_layer(colors, [indicator])
    tensor_y = tf.feature_column.input_layer(colors2, [indicator])

    x = tf.string_to_hash_bucket_fast(tf.constant(["xf", "sf"]), num_buckets=20)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([x]))


def test04():
    dataset = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0, 4.0, 5.0])#, [4.0,6,7,8]])
    # dataset = dataset.range(5)
    iterator = dataset.make_one_shot_iterator()  # 从到到尾读一次
    one_element = iterator.get_next()  # 从iterator里取出一个元素
    with tf.Session() as sess:  # 建立会话（session）

        for i in range(2):  # 通过for循环打印所有的数据
            print(sess.run(one_element))  # 调用sess.run读出Tensor值

def test05():
    xy = [[(1, 2), (3,4), (4,5)], [(5,6)]]
    def gen():
        for i in range(0, 2):
            yield xy[i]

    # Create dataset from generator
    # The output shape is variable: (None,)
    dataset = tf.data.Dataset.from_generator(gen)

    # The issue here is that we want to batch the data
    dataset = dataset.apply(tf.data.experimental.unbatch())
    # dataset = dataset.unbatch()
    # dataset = dataset.batch(2)

    # Create iterator from dataset
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()  # shape (None,)

    sess = tf.Session()
    for i in range(10):
        print(sess.run(x))


def test06():
    x = []
    x.append([2,3])
    print(x)

def test07():
    def generate_sample(index_words):

        context_window_size = 4
        split_pair = []
        s = ""
        """ Form training pairs according to the skip-gram model. """
        for index, center in enumerate(index_words):
            print("index: %d" % index)
            context = context_window_size
            # context = random.randint(1, context_window_size)
            # get a random target before the center word [center, target]
            for target in index_words[max(0, index - context): index]:
                split_pair.append([center, target])
                # split_pair.append(center + b"\t" + target)
                # s += center + b"\t"+ target + ","
            # get a random target after the center word
            for target in index_words[index + 1: index + context + 1]:
                split_pair.append([center, target])
                # split_pair.append(center + b"\t" + target)
                # s += center + b"\t" + target + ","
        return split_pair
        # return np.array([3,4,5,6])
        # return np.array([[2,3],[4,5],[5,6]])
        # return np.array(split_pair)

    # index_words = b"jfjafj,jfdjfosj,fdfdf,fdfsf,joijfjfodjfoj,fd,ffdf"
    index_words=[b'ffd',b'ffds',b'y',b'b',b's']
    # index_words = [1,2,3,4,5]

    # m = generate_sample(index_words.split(","))
    # print(m)

    x = tf.constant(index_words)
    # output = tf.string_split([x], delimiter=',').values
    output1 = tf.py_function(generate_sample, [x,], tf.string)
    # print(output1)
    sess = tf.Session()
    z = sess.run(output1)
    print(z)
    print(len(z))


def test08():
    def generate_sample(index_words):
        split_pair = []
        """ Form training pairs according to the skip-gram model. """
        for index, center in enumerate(index_words):
            print("index: %d" % index)
            context = 4
            for target in index_words[max(0, index - context): index]:
                split_pair.append([center, target])
            for target in index_words[index + 1: index + context + 1]:
                split_pair.append([center, target])

        return np.array(split_pair)


    index_words0 = b"jfjafj,jfdjfosj,fdfdf,fdfsf,joijfjfodjfoj,fd,ffdf"
    m = tf.string_split([index_words0], delimiter=',').values
    index_words = tf.string_to_hash_bucket_fast(m, num_buckets=20)
    # index_words = [1, 2, 3, 4, 5, 9]
    # x = tf.constant(index_words)
    output = tf.py_function(generate_sample, [index_words], tf.int32)


    sess = tf.Session()
    z = sess.run(output)
    print(sess.run(index_words))
    print(z)

def test09():
    # split_pair = [[2,3],[3,4.1]]
    # print(29527 /20/3 * 400000000)
    # import datetime
    # print(datetime.datetime.now().strftime('%Y-%m-%d'))
    # print(np.array(split_pair, dtype=np.int64))
    # with open("x.txt", "w") as f:
    #     for i in range(5000):
    #         f.write("232141323\t")
    # for i, (x, y) in enumerate(zip([1,2,3], [4,5,6])):
    #     print(i, x, y)

#     s = [614795 ,
# 616366 ,
# 614655]
#     a = 40000 / () / 9 / 60 * ()
#     print(a)
    # print(sum(s))
    a = [
        100, 150, 200, 250, 300, 320, 350, 380, 400, 420, 450, 480, 500, 520, 550, 580, 600, 620, 650, 680, 700, 720,
        750, 800, 820, 850, 880, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1400
    ]
    print(len(a))


if __name__ == "__main__":
    test09()
