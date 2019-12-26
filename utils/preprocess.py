#coding=utf-8
#Version:python3.5.2
import tensorflow as tf
import numpy as np
import alex.config as cfg
import os
#加载均值文件
# numpy_file = np.load('mean.npy')
'''定义data类'''
class Data(object):
    def __init__(self,phase, rebuild=False):
        #分类图片存放地址
        self.data_path = os.path.join(cfg.DATA_PATH, 'tfrecord')
        self.path = os.path.join(self.data_path,'train.tfrecords')
        #tfrecord 缓存文件保存地址
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.classes = cfg.CLASSES
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        #数据增强标志
        self.flipped = cfg.FLIPPED
        self.phase = phase  #加载训练还是验证文件
        self.rebuild = rebuild#与是否加载缓存有关的标志
    def get(self,image, label):
        # 当一次出列操作完成后,队列中元素的最小数量,往往用于定义元素的混合级别.
        min_after_dequeue = 100
        # 定义了随机取样的缓冲区大小,此参数越大表示更大级别的混合但是会导致启动更加缓慢,并且会占用更多的内存
        capacity = min_after_dequeue + 3 * self.batch_size
        image_batch0, label_batch0 = tf.train.shuffle_batch(
            [image, label], batch_size=self.batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue,
            num_threads=16)
        one_hot_labels0 = tf.one_hot(indices=tf.cast(label_batch0, tf.int32), depth=len(self.classes))
        return image_batch0, one_hot_labels0

    def read_and_decode(self,filename):
        # 根据文件名生成一个队列
        filename_queue = tf.train.string_input_producer([filename],
                                                        # num_epochs=100,
                                                        shuffle=True)
        reader = tf.TFRecordReader()#用来读取数据
        # 返回文件名和文件
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,#做一个解析
                                           features={
                                               'image': tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([], tf.int64),
                                           })
        # 获取图片数据
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, [224, 224, 3])
        self.image = tf.cast(image, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
        #TODO 做减去均值处理
        # tensor_mean = tf.convert_to_tensor(numpy_file)
        # tensor_mean=tf.expand_dims(tensor_mean, -1)
        # tensor_mean = tf.image.resize_image_with_crop_or_pad(tensor_mean, 224, 224)
        # image=tf.subtract(image , tensor_mean, name=None)
        # image = tf.image.per_image_standardization(image)
        # if self.flipped:
        #     #按水平 (从左向右) 随机翻转图像.以1比2的概率,输出image沿着第二维翻转的内容,即,width.否则按原样输出图像.
        #     image = tf.image.random_flip_left_right(image)
        #     image = tf.image.random_flip_up_down(image)##将图片随机进行垂直翻转
        # 获取label
        self.label = tf.cast(features['label'], tf.int32)
        return self.image, self.label

    #TODO 使用tf.data api进行数据加载
    def parse_record(self,raw_record):
        keys_to_features = {
            'image': tf.FixedLenFeature((), tf.string),
            'label': tf.FixedLenFeature((), tf.int64),
        }
        parsed = tf.parse_single_example(raw_record, keys_to_features)
        image = tf.decode_raw(parsed['image'], tf.uint8)
        image = tf.to_float(image)
        image = tf.reshape(image, [224, 224, 3])
        label = tf.cast(parsed['label'], tf.int32)
        one_hot_labels0 = tf.one_hot(indices=tf.cast(label, tf.int32), depth=len(self.classes))

        return image, one_hot_labels0
    def dataset_batch(self):
        dataset = tf.data.TFRecordDataset(self.path)
        dataset = dataset.map(self.parse_record, num_parallel_calls=2)
        dataset = dataset.map(lambda image, label: self.prepare(image, label, is_training=True), num_parallel_calls=8)
        buffer_size = 1000
        num_epochs = 10000
        batch_size = 32
        dataset = dataset.shuffle(buffer_size).repeat(num_epochs).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        image, caption = iterator.get_next()
        return image, caption
    def batch_data(self):#必须写在一起
        filename_queue = tf.train.string_input_producer([self.path])  # 生成一个queue队列

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image': tf.FixedLenFeature([], tf.string),
                                           })  # 将image数据和label取出来

        img = tf.decode_raw(features['image'], tf.uint8)
        # image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
        img = tf.reshape(img, [224, 224, 3])  # reshape为224*224的3通道图片
        label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
        image = self.prepare(img,label)
        # 当一次出列操作完成后,队列中元素的最小数量,往往用于定义元素的混合级别.
        min_after_dequeue = 100
        batch_size = 32
        capacity = min_after_dequeue + 3 * batch_size
        image_batch0, label_batch0 = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue,
            num_threads=8)
        one_hot_labels0 = tf.one_hot(indices=tf.cast(label_batch0, tf.int32), depth=len(self.classes))
        return image_batch0,one_hot_labels0
    def prepare(self,img,label,is_training=True):
        #data enhancement
        image = (tf.cast(img, tf.float32) * (1. / 255) - 0.5) * 2  # 在流中抛出img张量
        if self.flipped:
            #1 按水平 (从左向右) 随机翻转图像.以1比2的概率,输出image沿着第二维翻转的内容,即,width.否则按原样输出图像.
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)##将图片随机进行垂直翻转
            # 3、随机改变亮度和对比度：对得到的图片进行亮度和对比度的随机改变。
        distorted_image = tf.image.random_brightness(image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)
        # 随机旋转
        random_angles = tf.random.uniform(shape=(1,), minval=-np.pi / 6, maxval=np.pi / 6)
        rotated_images = tf.contrib.image.transform(
            distorted_image,
            tf.contrib.image.angles_to_projective_transforms(
                random_angles, tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
            ))
        return rotated_images,label
def parse_record(raw_record):
    keys_to_features = {
      'image': tf.FixedLenFeature((), tf.string),
      'label': tf.FixedLenFeature((), tf.int64),
    }
    parsed = tf.parse_single_example(raw_record, keys_to_features)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.to_float(image)
    image = tf.reshape(image, [224,224,3])
    label = tf.cast(parsed['label'], tf.int32)
    return image, label
def preprocess_image(image,label,is_training = True):
    # 3、随机改变亮度和对比度：对得到的图片进行亮度和对比度的随机改变。
    distorted_image = tf.image.random_brightness(image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    # 随机旋转
    random_angles = tf.random.uniform(shape=(1,), minval=-np.pi/6, maxval=np.pi/6)
    rotated_images = tf.contrib.image.transform(
        distorted_image,
        tf.contrib.image.angles_to_projective_transforms(
            random_angles, tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
        ))
    return rotated_images,label


if __name__ =='__main__':

    #测试方式一：打包在一个函数里
    # shuju = Data(phase=True)
    # image, label = shuju.batch(r'D:\pycharm\Alexnet\data\tfrecord\train.tfrecords')
    # with tf.Session() as sess:
    #     init_op = tf.initialize_all_variables()
    #     sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     for i in range(2):
    #         img, lab = sess.run([image, label])
    #         print(np.shape(img))
    #         print(np.shape(lab))
    #     coord.request_stop()
    #     coord.join(threads)
    #测试方式二：分开写
    TFRECORD_FILE = r'D:\pycharm\Alexnet\data\tfrecord\train.tfrecords'
    shuju = Data(phase=True)
    image, label = shuju.read_and_decode(r'D:\pycharm\Alexnet\data\tfrecord\train.tfrecords')
    img1, lab1 = shuju.get(image, label)
    #测试方式三  使用tf.data api
    # TFRECORD_FILE = r'D:\pycharm\Alexnet\data\tfrecord\train.tfrecords'
    # dataset = tf.data.TFRecordDataset(TFRECORD_FILE)
    # dataset = dataset.map(parse_record)
    # 对dataset中的每条数据, 应用lambda函数, 输入image, label, 用preprocess_image()函数处理,得到新的dataset
    # dataset = dataset.map(lambda image, label: preprocess_image(image, label, is_training=True))
    # buffer_size = 1000#buffer_size参数等效于tf.train.shuffle_batch的min_after_dequeue参数
    # num_epochs = 10
    # batch_size = 32
    # dataset = dataset.shuffle(buffer_size).repeat(num_epochs).batch(batch_size)
    # iterator = dataset.make_one_shot_iterator()
    # image, caption = iterator.get_next()
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(2):
            img ,lab = sess.run([img1,lab1])
    #         img ,lab = sess.run([image,caption])
    #         # img ,lab = sess.run([image_batch0,label_batch0])
            print(np.shape(img))
    #         print(np.shape(lab))
        coord.request_stop()
        coord.join(threads)

