import tensorflow as tf
import numpy as np


class ImageProcessor:
    class ImageProcessorException(Exception):
        def __init__(self, message: str):
            self.__msg = message

        def __str__(self):
            return self.__msg

    def __init__(self, classes_number: int, train_images_num: int, image_size: [], epoch_every_train_num: int):
        self.__batch_size = 64
        if type(image_size) != list or len(image_size) != 3:
            raise self.ImageProcessorException("Bad image size data. This must be list of 3 integers")
        self.__image_size = image_size
        self.__classes_num = classes_number
        self.__on_epoch = None
        self.__init_nn()
        self.__iteration_idx = 0
        self.__train_images_num = train_images_num
        self.__saver = tf.train.Saver()
        self.__epoch_every_train_num = epoch_every_train_num

    @staticmethod
    def __create_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    @staticmethod
    def __create_biases(size):
        return tf.Variable(tf.constant(0.05, shape=[size]))

    @staticmethod
    def __create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
        weights = ImageProcessor.__create_weights(
            shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        biases = ImageProcessor.__create_biases(num_filters)

        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
        layer += biases

        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        layer = tf.nn.relu(layer)

        return layer

    @staticmethod
    def __create_flatten_layer(layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer = tf.reshape(layer, [-1, num_features])

        return layer

    @staticmethod
    def __create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
        weights = ImageProcessor.__create_weights(shape=[num_inputs, num_outputs])
        biases = ImageProcessor.__create_biases(num_outputs)

        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    def __init_nn(self):
        # set variables
        self.__x = tf.placeholder(tf.float32,
                                  shape=[None, self.__image_size[0], self.__image_size[1], self.__image_size[2]],
                                  name='x')

        self.__y_true = tf.placeholder(tf.float32, shape=[None, self.__classes_num], name='y_true')
        y_true_cls = tf.argmax(self.__y_true, axis=0)

        self.__session = tf.Session()
        self.__session.run(tf.global_variables_initializer())

        # network design
        filter_size_conv1 = 3
        num_filters_conv1 = 32
        filter_size_conv2 = 3
        num_filters_conv2 = 32
        filter_size_conv3 = 3
        num_filters_conv3 = 64
        fc_layer_size = 128

        layer_conv1 = self.__create_convolutional_layer(input=self.__x, num_input_channels=self.__image_size[2],
                                                        conv_filter_size=filter_size_conv1,
                                                        num_filters=num_filters_conv1)

        layer_conv2 = self.__create_convolutional_layer(input=layer_conv1, num_input_channels=num_filters_conv1,
                                                        conv_filter_size=filter_size_conv2,
                                                        num_filters=num_filters_conv2)

        layer_conv3 = self.__create_convolutional_layer(input=layer_conv2, num_input_channels=num_filters_conv2,
                                                        conv_filter_size=filter_size_conv3,
                                                        num_filters=num_filters_conv3)

        layer_flat = self.__create_flatten_layer(layer_conv3)

        layer_fc1 = self.__create_fc_layer(input=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                           num_outputs=fc_layer_size,
                                           use_relu=True)

        layer_fc2 = self.__create_fc_layer(input=layer_fc1, num_inputs=fc_layer_size,
                                           num_outputs=self.__classes_num,
                                           use_relu=False)

        y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

        y_pred_cls = tf.argmax(y_pred, axis=0)
        self.__session.run(tf.global_variables_initializer())

        # prediction variables
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                                labels=self.__y_true)
        self.__cost = tf.reduce_mean(cross_entropy)

        # optimisation
        self.__optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.__cost)

        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        self.__accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.__session.run(tf.global_variables_initializer())

    def __init_label(self, label_id: int):
        label = np.zeros(self.__classes_num)
        label[label_id] = 1.0
        return label

    def train_batch(self, images: [{}]):
        x_batch = [img['object'] for img in images]
        y_true_batch = [self.__init_label(int(img['label_id']) - 1) for img in images]

        feed_dict = {self.__x: x_batch, self.__y_true: y_true_batch}

        self.__session.run(self.__optimizer, feed_dict=feed_dict)

        if self.__on_epoch is not None:
            if self.__iteration_idx % self.__epoch_every_train_num == 0:
                self.__on_epoch()

        self.__iteration_idx += 1

    def set_on_epoch(self, callback: callable):
        self.__on_epoch = callback

    def get_loss_value(self, images: [{}]):
        x_batch = [img['object'] for img in images]
        y_true_batch = [self.__init_label(int(img['label_id']) - 1) for img in images]

        feed_dict = {self.__x: x_batch, self.__y_true: y_true_batch}

        try:
            return self.__session.run(self.__cost, feed_dict=feed_dict)
        except ValueError:
            print('kek')

    def get_accuracy(self, images: [{}]):
        x_batch = [img['object'] for img in images]
        y_true_batch = [self.__init_label(int(img['label_id']) - 1) for img in images]

        feed_dict = {self.__x: x_batch, self.__y_true: y_true_batch}

        return self.__session.run(self.__accuracy, feed_dict=feed_dict)

    def get_cur_epoch(self):
        return self.__iteration_idx // self.__epoch_every_train_num

    def save_state(self, path: str):
        self.__saver.save(self.__session, path)


class Predictor:
    def __init__(self, path, classes_number: int, image_size: []):
        self.__session = tf.Session()
        saver = tf.train.import_meta_graph(path)
        saver.restore(self.__session, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        self.__y_pred = graph.get_tensor_by_name("y_pred:0")
        self.__x = graph.get_tensor_by_name("x:0")
        self.__y_true = graph.get_tensor_by_name("y_true:0")
        self.__y_test_images = np.zeros((1, classes_number))
        self.__image_size = image_size

    def predict(self, image):
        x_batch = image['object'].reshape(1, self.__image_size[0], self.__image_size[1], self.__image_size[2])
        feed_dict_testing = {self.__x: x_batch, self.__y_true: self.__y_test_images}
        return self.__session.run(self.__y_pred, feed_dict=feed_dict_testing)
