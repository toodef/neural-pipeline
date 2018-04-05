import time

from image_conveyor import ImageConveyor, PathLoader
from image_processor import ImageProcessor
import os
import cv2
import numpy as np
from random import shuffle

train_dir = 'train'
validation_dir = 'validation'
image_size = 128
images_part = 32
result_path = 'result\\furniture_segmentation'

classes = [int(dir) for dir in os.listdir(train_dir)]


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

img_processor = ImageProcessor(len(classes), len(train_pathes), [image_size, image_size, 3], epoch_every_train_num=1000)

last_train_images = []

start_time = None


def on_epoch():
    img_processor.save_state(result_path)

    accuracy = img_processor.get_accuracy(last_train_images)
    with ImageConveyor(PathLoader(), validation_pathes, images_part) as conveyor:
        loss_values = []
        valid_accuracies = []
        for images in conveyor:
            if len(images) < images_part:
                continue
            for img in images:
                img['object'] = cv2.resize(img['object'], (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            loss_values.append(img_processor.get_loss_value(images))
            valid_accuracies.append(img_processor.get_accuracy(images))

    epoch = img_processor.get_cur_epoch()
    valid_acc = np.mean(np.array(valid_accuracies))
    loss = np.mean(np.array(loss_values))
    print("Epoch: {}, Training accuracy: {:>6.1%}, Validation accuracy {:>6.1%}, validation loss: {:.3f},  Time: {:.2f} sec".format(epoch, accuracy, valid_acc, loss, time.time() - start_time))


img_processor.set_on_epoch(on_epoch)


with ImageConveyor(PathLoader(), train_pathes, images_part) as conveyor:
    conveyor.set_iterations_num(len(train_pathes) * 100)
    start_time = time.time()
    for images in conveyor:
        if len(images) < images_part:
            continue
        for img in images:
            img['object'] = cv2.resize(img['object'], (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
        last_train_images = images
        img_processor.train_batch(images)
