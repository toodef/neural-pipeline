import os
import sys
import time
import cv2

from image_conveyor import ImageConveyor, ImageLoader, UrlLoader
import json


class TestLoader(ImageLoader):
    def load(self, url):
        return url


def main():
    with open("train.json", 'r') as file:
        data = json.load(file)

    pathes = []
    for i, item in enumerate(data['images']):
        pathes.append({'path': item['url'][0], 'label_id': data['annotations'][item['image_id'] - 1]['label_id'],
                       'image_id': item['image_id']})

    with ImageConveyor(UrlLoader(), pathes, 1000) as conveyor, open('log.txt', 'w') as log_file:
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
                    string = "download {} failed; {:.2f} sec, failed {:.2f}%".format(i,
                                                                                                      time.time() - start_time,
                                                                                                      100 * failed_num / i)
                    print(string, file=sys.stderr)
                    log_file.write(string + "\n")
                    continue
                dir = "data\\{}".format(img['label_id'])
                if not os.path.exists(dir):
                    os.makedirs(dir)
                try:
                    cv2.imwrite(os.path.join(dir, "{}.jpg".format(img['image_id'])), img['object'])
                except Exception:
                    failed_num += 1
                    failed_url.append(img['path'])
                    string = "download {} failed; {:.2f} sec, failed {:.2f}%".format(i,
                                                                                                      time.time() - start_time,
                                                                                                      100 * failed_num / i)
                    print(string, file=sys.stderr)
                    log_file.write(string + "\n")
                    failed_url.append(img['path'])
                    continue

                del img['object']

                loaded_num += 1
                string = "download {} succeed; {:.2f} sec, failed {:.2f}%".format(i,
                                                                                                   time.time() - start_time,
                                                                                                   100 * failed_num / i)
                print(string)
                log_file.write(string + "\n")
                log_file.flush()

        print("elapsed: {} sec; loaded: {}, failed: {} ({} % of failed)".format(time.time() - start_time, loaded_num,
                                                                                failed_num,
                                                                                100 * failed_num / len(pathes)))

        with open("failed_urls_log.txt", 'w') as url_log_file:
            for url in failed_url:
                url_log_file.write(url)
            url_log_file.flush()


if __name__ == "__main__":
    main()
