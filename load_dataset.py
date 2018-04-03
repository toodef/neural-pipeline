import os
import sys
import time
import cv2

from image_conveyor import ImageConveyor, UrlLoader
import json


def load(data_file_path: str):
    with open(data_file_path + ".json", 'r') as file:
        data = json.load(file)

    pathes = []
    label_ids = {}
    for a in data['annotations']:
        label_ids[a['image_id']] = a['label_id']

    for img in data['images']:
        pathes.append({'path': img['url'][0], 'label_id': label_ids[img['image_id']], 'image_id': img['image_id']})

    with ImageConveyor(UrlLoader(), pathes, 1000) as conveyor, open(data_file_path + '_log.log', 'w') as log_file:
        i = 0
        start_time = time.time()
        loaded_num, failed_num = 0, 0
        failed_url = []
        for images in conveyor:
            for img in images:
                i += 1
                if img['object'] is None:
                    failed_num += 1
                    failed_url.append(img['path'])
                    string = "download {} failed; {:.2f} sec, failed {:.2f}%".format(i, time.time() - start_time,
                                                                                     100 * failed_num / i)
                    print(string, file=sys.stderr)
                    log_file.write(string + "\n")
                    continue
                dir = data_file_path + "\\{}".format(img['label_id'])
                if not os.path.exists(dir):
                    os.makedirs(dir)
                try:
                    cv2.imwrite(os.path.join(dir, "{}.jpg".format(img['image_id'])), img['object'])
                except Exception:
                    failed_num += 1
                    failed_url.append(img['path'])
                    string = "download {} failed; {:.2f} sec, failed {:.2f}%".format(i, time.time() - start_time,
                                                                                     100 * failed_num / i)
                    print(string, file=sys.stderr)
                    log_file.write(string + "\n")
                    failed_url.append(img['path'])
                    continue

                del img['object']

                loaded_num += 1
                string = "download {} succeed; {:.2f} sec, failed {:.2f}%".format(i, time.time() - start_time,
                                                                                  100 * failed_num / i)
                print(string)
                log_file.write(string + "\n")
                log_file.flush()

        print("elapsed: {} sec; loaded: {}, failed: {} ({} % of failed)".format(time.time() - start_time, loaded_num,
                                                                                failed_num,
                                                                                100 * failed_num / len(pathes)))

        with open(data_file_path + "_failed_urls_log.log", 'w') as url_log_file:
            for url in failed_url:
                url_log_file.write(url + "\n")
            url_log_file.flush()


if __name__ == "__main__":
    load('validation')

