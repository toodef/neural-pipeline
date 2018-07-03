import json
import os

target_dir = r"Z:\datasets\fure"
validation_dir = r"Z:\datasets\fure\validation"
train_dir = r"Z:\datasets\fure\train"


def get_pathes(directory, classes):
    res = []
    for cur_class in classes:
        res += [{'path': os.path.relpath(os.path.join(os.path.join(directory, str(cur_class)), file), target_dir), 'target': int(cur_class) - 1}
                for file in os.listdir(os.path.join(directory, str(cur_class)))]
    return res


train_classes = [int(d) for d in os.listdir(train_dir)]
train_dataset = {'labels': {str(l): l - 1 for l in train_classes}, 'data': get_pathes(train_dir, train_classes)}
validation_classes = [int(d) for d in os.listdir(validation_dir)]
validation_dataset = {'labels': {str(l): l - 1 for l in train_classes}, 'data': get_pathes(validation_dir, validation_classes)}

overall_dataset = {'labels': dict(train_dataset['labels'], **validation_dataset['labels']), 'data': train_dataset['data'] + validation_dataset['data']}


with open(os.path.join(target_dir, 'train.json'), 'w') as out:
    json.dump(train_dataset, out, indent=2)
with open(os.path.join(target_dir, 'validation.json'), 'w') as out:
    json.dump(validation_dataset, out, indent=2)
with open(os.path.join(target_dir, 'dataset.json'), 'w') as out:
    json.dump(overall_dataset, out, indent=2)
