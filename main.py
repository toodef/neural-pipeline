from image_conveyor import ImageConveyor


class TestLoader:
    def load(self, url):
        return url


def main():
    pathes = ['a', 'b', 'c', 'd', 'e', 'f']

    with ImageConveyor(TestLoader(), pathes, 4) as conveyor:
        for images in conveyor:
            print(images)


if __name__ == "__main__":
    main()
