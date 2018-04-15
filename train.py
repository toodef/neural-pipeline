import time

from image_conveyor import ImageConveyor, PathLoader
from image_processor import ImageProcessor
import os
import cv2
import numpy as np
from random import shuffle

train_dir = 'train'
validation_dir = 'validation'
image_size = 256
batch_size = 32
epoch_every_images_parts = 4
result_dir = 'result'
result_name_preffix = 'furniture_segmentation'

# classes = [int(dir) for dir in os.listdir(train_dir)]
classes = [1, 2]


def get_pathes(directory):
    res = []
    for cur_class in classes:
        res += [{'path': os.path.join(os.path.join(directory, str(cur_class)), file), 'label_id': cur_class}
                for file in
                os.listdir(os.path.join(directory, str(cur_class)))]
    return res


train_pathes = get_pathes(train_dir)
shuffle(train_pathes)
validation_pathes = get_pathes(validation_dir)

epoch_every_train_num = int(epoch_every_images_parts * len(train_pathes) / batch_size)

img_processor = ImageProcessor(len(classes), len(train_pathes), [image_size, image_size, 3], epoch_every_train_num=epoch_every_train_num)

# last_train_images = []

start_time = None


def after_load(image: {}):
    image['object'] = np.multiply(cv2.resize(image['object'], (image_size, image_size), 0, 0, cv2.INTER_LINEAR), 1 / 255)


def on_epoch():
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    img_processor.save_state(os.path.join(result_dir, result_name_preffix))

    with ImageConveyor(PathLoader().after_load(after_load), validation_pathes, batch_size) as tmp_conveyor:
        loss_values = []
        valid_accuracies = []
        accuracyes = []
        for images in tmp_conveyor:
            if len(images) < batch_size:
                continue
            accuracyes.append(img_processor.get_accuracy(images))
            loss_values.append(img_processor.get_loss_value(images))
            valid_accuracies.append(img_processor.get_accuracy(images))

            for img in images:
                img['object'] = None

    # tmp_train_pathes = train_pathes.copy()[0:batch_size * 50]
    # shuffle(tmp_train_pathes)
    with ImageConveyor(PathLoader().after_load(after_load), train_pathes, batch_size) as tmp_conveyor:
        accuracyes = []
        for images in tmp_conveyor:
            if len(images) < batch_size:
                continue
            accuracyes.append(img_processor.get_accuracy(images))

            for img in images:
                img['object'] = None

    epoch = img_processor.get_cur_epoch()
    valid_acc = np.average(np.array(valid_accuracies))
    loss = np.average(np.array(loss_values))
    accuracy = np.average(np.array(accuracyes)) # img_processor.get_accuracy(last_train_images)
    print(
        "Epoch: {}, Training accuracy: {:>6.1%}, Validation accuracy {:>6.1%}, validation loss: {:.3f},  Time: {:.2f} min".format(
            epoch, accuracy, valid_acc, loss, (time.time() - start_time) / 60))


img_processor.set_on_epoch(on_epoch)


with ImageConveyor(PathLoader().after_load(after_load), train_pathes, batch_size) as conveyor:
    conveyor.set_iterations_num(len(train_pathes) * 100)
    # conveyor.set_processes_num(4)
    start_time = time.time()
    for images in conveyor:
        if len(images) < batch_size:
            continue
        # last_train_images = images
        img_processor.train_batch(images)

        for img in images:
            img['object'] = None
