import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import alex.config as cfg
from alex.alex_net import AlexNet
from utils.timer import Timer


class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.threshold = cfg.THRESHOLD
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result,conf):
    #传入的是引用，不需返回？？
        lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
        cv2.putText(
            img, result + ' : %.2f' % conf,
            # (img.shape[1] W- 5,img.shape[0] H- 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1, lineType)


    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result ,conf= self.detect_from_cvmat(inputs)
        return result,conf

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.X: inputs})
        results = np.argmax(net_output, axis=1)
        return cfg.CLASSES[results[0]],np.max(net_output)


    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()
        result, conf= self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))

        self.draw_result(image, result,conf)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="alexnet-5000", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    alexnet = AlexNet(False)
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    detector = Detector(alexnet, weight_file)

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from image file
    imname = 'test/cat.202.jpg'
    detector.image_detector(imname)


if __name__ == '__main__':
    main()
 # python test.py  --data_dir ./data/pascal_voc/output --weight_dir 2019_09_07_19_13  --weights yolo-10
