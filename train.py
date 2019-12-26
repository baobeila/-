#!/usr/bin/env python
# encoding: utf-8
"""
@version: JetBrains PyCharm 2017.3.2 x64
@author: baobeila
@contact: baibei891@gmail.com
@software: PyCharm
@file: train.py
@time: 2019/12/20 17:42
"""
import argparse
import alex.config as cfg
import os
from alex.alex_net import AlexNet
import datetime
import tensorflow as tf
import numpy as np
from utils.timer import Timer
from utils.preprocess import Data


class Solver(object):
    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.skip_layer = cfg.SKIP_LAYER#不加载与训练权重的层，也就是要进行训练的层
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()
        self.ckpt_file = os.path.join(self.output_dir, 'alexnet')
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)
        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate,
            self.global_step,
            self.decay_steps,
            self.decay_rate,
            self.staircase, name='learning_rate')

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess =  tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        #TODO 从断点继续训练
        if self.weights_file is not None:
            # List of trainable variables of the layers we want to train
            # select
            if len(cfg.SKIP_LAYER)==0:#权重全部加载
                self.load_initial_weights(self.sess)
            else:
                var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in cfg.SKIP_LAYER]
            with tf.name_scope("train"):
                # Get gradients of all trainable variables
                gradients = tf.gradients(self.net.total_loss, var_list)  # 计算梯度
                gradients = list(zip(gradients, var_list))
                # Create optimizer and apply gradient descent to the trainable variables
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate)
                self.train_op = self.optimizer.apply_gradients(grads_and_vars=gradients, global_step=self.global_step)
            # Add gradients to summary
            for gradient, var in gradients:
                tf.summary.histogram(var.name + '/gradient', gradient)

            # Add the variables we train to the summary
            for var in var_list:
                tf.summary.histogram(var.name, var)
            print('Restoring weights from: ' + self.weights_file)
            self.load_initial_weights(self.sess)
        else:
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.net.total_loss, global_step = self.global_step)
        # Merge all summaries together,处于所有summary后面
        self.summary_op = tf.summary.merge_all()
        self.writer.add_graph(self.sess.graph)#写入图结构
        self.saver = tf.train.Saver()

    def train(self):
        train_timer = Timer()
        load_timer = Timer()
        # images, labels = self.data.batch_data()
        images, labels = self.data.dataset_batch()
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        for step in range(1, self.max_iter + 1):
            load_timer.tic()
            img_batch, one_hot_labels = self.sess.run([images, labels])
            load_timer.toc()
            feed_dict = {self.net.X: img_batch,
                         self.net.labels: one_hot_labels}
            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:#10 倍，进行一次loss输出
                    train_timer.tic()
                    summary_str, loss, train_acc, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.net.accuracy, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()
#TODO 计算epoch 并进行打印
                    print('{}, Step: {}, Learning rate: {},Loss: {:5.3f},Train Accuracy = {:.4f}'.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6), loss, train_acc),end=' ')

                    print('Speed: {:.3f}s/iter,Load: {:.3f}s/iter, Remain:{}'.format(
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter)#还需多长时间训练好
                    ))
                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                self.writer.add_summary(summary_str, step)
            else:#没到记录日志，只是训练
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()
            if step % self.save_iter == 0:  # 保留检查点，以供测试时用
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(  # 保存会话，将模型文件保存
                    self.sess, self.ckpt_file, global_step=self.global_step)
                print("save done!!!")
        # coord.request_stop()
        # coord.join(threads)
    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                #检测字符串中所有的字母是否都为大写
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)



    def load_initial_weights(self, session):
        """Load weights from file into network.
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.weights_file, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.skip_layer:

                with tf.variable_scope(op_name, reuse=True):

                    # reuse=True tf.variable_scope只能获取已经创建过的变量
                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)  # 不用初始化，指定不进行训练,get从中加载
                            session.run(var.assign(data))  # 赋值在会话里进行

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',default="bvlc_alexnet .npy",type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()
    if args.gpu is not None:#命令行参数可以更新配置文件的参数
        cfg.GPU = args.gpu
    if args.data_dir != cfg.DATA_PATH:
        cfg.DATA_PATH = args.data_dir
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    alexnet = AlexNet()
    dataset = Data('train')
    solver = Solver(alexnet, dataset)
    print('Start training ...')
    solver.train()
    print('Done training.')
if __name__ == '__main__':
    main()