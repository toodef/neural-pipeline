import tensorflow as tf


class ImageProcessor:
    class ImageProcessorException(Exception):
        def __init__(self, message: str):
            self.__msg = message

        def __str__(self):
            return self.__msg

    def __init__(self, train_images: [{}], validation_images: [{}], classes_number: int, image_size: []):
        self.__batch_size = 16
        self.__train_images = train_images
        self.__validation_images = validation_images
        if type(image_size) != list or len(image_size) != 3:
            raise self.ImageProcessorException("Bad image size data. This must be list of 3 integers")
        self.__image_size = image_size
        self.__classes_num = classes_number

    @staticmethod
    def __create_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    @staticmethod
    def __create_biases(size):
        return tf.Variable(tf.constant(0.05, shape=[size]))

    @staticmethod
    def __create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
        ## We shall define the weights that will be trained using create_weights function.
        weights = ImageProcessor.__create_weights(
            shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        ## We create biases using the create_biases function. These are also trained.
        biases = ImageProcessor.__create_biases(num_filters)

        ## Creating the convolutional layer
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        layer += biases

        ## We shall be using max-pooling.
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        ## Output of pooling is fed to Relu which is the activation function for us.
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
        # Let's define trainable weights and biases.
        weights = ImageProcessor.__create_weights(shape=[num_inputs, num_outputs])
        biases = ImageProcessor.__create_biases(num_outputs)

        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    def init_nn(self):
        # set variables
        x = tf.placeholder(tf.float32, shape=[None, self.__image_size[0], self.__image_size[1], self.__image_size[2]],
                           name='x')

        y_true = tf.placeholder(tf.float32, shape=[None, self.__classes_num], name='y_true')
        y_true_cls = tf.argmax(y_true, dimension=1)

        # network design
        layer_conv1 = self.__create_convolutional_layer(input=x, num_input_channels=self.__image_size[2],
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
                                           num_outputs=num_classes,
                                           use_relu=False)

        # prediction variables
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                                labels=y_true)
        cost = tf.reduce_mean(cross_entropy)

        # optimisation
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    def train_batch(self):
        pass