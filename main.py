import os

import sys

import time

from image_conveyor import ImageConveyor, ImageLoader, UrlLoader
import json


class TestLoader(ImageLoader):
    def load(self, url):
        return url


def main():
    with open("C:\\Users\\fedot\\Downloads\\train.json", 'r') as file:
        data = json.load(file)

    pathes = []
    for i, item in enumerate(data['images']):
        pathes.append({'path': item['url'][0], 'label_id': data['annotations'][item['image_id'] - 1]['label_id']})

    with ImageConveyor(UrlLoader(), pathes, 4) as conveyor:
        i = 0
        start_time = time.time()
        loaded_num, failed_num = 0, 0
        failed_url = []
        for images in conveyor:
            for img in images:
                i += 1
                if img['object'] is None:
                    print("download {} failed".format(i), file=sys.stderr)
                    failed_num += 1
                    failed_url.append(img['path'])
                    continue
                dir = "C:\\workspace\\projects\\nn\\furniture_segmentation\\data\\{}".format(img['label_id'])
                if not os.path.exists(dir):
                    os.makedirs(dir)
                try:
                    img['object'].convert('RGB').save(os.path.join(dir, "{}.jpg".format(i)), "JPEG", quality=90, optimize=True, progressive=True)
                except Exception:
                    print("download {} failed".format(i), file=sys.stderr)
                    failed_url.append(img['path'])
                    failed_num += 1
                    continue

                print("download {} succeed".format(i))
                loaded_num += 1

        print("elapsed: {} sec; loaded: {}, failed: {} ({} % of failed)".format(time.time() - start_time, loaded_num, failed_num, 100 * failed_num / len(pathes)))

        with open("C:\\workspace\\projects\\nn\\furniture_segmentation\\data\\log.txt") as log_file:
            for url in failed_url:
                log_file.write(url)
            log_file.flush()

if __name__ == "__main__":
    main()
