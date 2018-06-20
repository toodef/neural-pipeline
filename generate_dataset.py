import json
import os

validation_dir = r"C:\workspace\nn_projects\furniture_segmentation\workdir\validation"
train_dir = r"C:\workspace\nn_projects\furniture_segmentation\workdir\train"


def get_pathes(directory, classes):
    res = []
    for cur_class in classes:
        res += [{'path': os.path.join(os.path.join(directory, str(cur_class)), file), 'target': int(cur_class) - 1}
                for file in os.listdir(os.path.join(directory, str(cur_class)))]
    return res


train_classes = [int(d) for d in os.listdir(train_dir)]
train_dataset = {'labels': {str(l): l - 1 for l in train_classes}, 'data': get_pathes(train_dir, train_classes)}
validation_classes = [int(d) for d in os.listdir(validation_dir)]
validation_dataset = {'labels': {str(l): l - 1 for l in train_classes}, 'data': get_pathes(validation_dir, validation_classes)}

with open(r'C:\workspace\nn_projects\furniture_segmentation\workdir\train.json', 'w') as out:
    json.dump(train_dataset, out, indent=2)
with open(r'C:\workspace\nn_projects\furniture_segmentation\workdir\validation.json', 'w') as out:
    json.dump(validation_dataset, out, indent=2)
with open(r'C:\workspace\nn_projects\furniture_segmentation\workdir\datast.json', 'w') as out:
    json.dump(train_dataset, out, indent=2)
    json.dump(validation_dataset, out, indent=2)
