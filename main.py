import os

import sys

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
        for images in conveyor:
            for img in images:
                i += 1
                if img['object'] is None:
                    print("download {} failed".format(i), file=sys.stderr)
                    continue
                print("download {} succeed".format(i))
                dir = "C:\\workspace\\projects\\nn\\furniture_segmentation\\data\\{}".format(img['label_id'])
                if not os.path.exists(dir):
                    os.makedirs(dir)
                img['object'].convert('RGB').save(os.path.join(dir, "{}.jpg".format(i)), "JPEG", quality=90, optimize=True, progressive=True)


if __name__ == "__main__":
    main()
