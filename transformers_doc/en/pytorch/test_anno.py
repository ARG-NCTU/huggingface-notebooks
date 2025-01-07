from datasets import Features, Value, Sequence

# Specify the correct schema for your dataset
features = Features({
    'image_id': Value('int32'),
    'image_path': Value('string'),
    'width': Value('int32'),
    'height': Value('int32'),
    'objects': {
        'id': Sequence(Value('int32')),
        'area': Sequence(Value('int32')), 
        'bbox': Sequence(Sequence(Value('int32'), length=4)), 
        'category': Sequence(Value('int32'))
    }
})

from datasets import load_dataset
dataset = load_dataset(
    'json', 
    data_files={'train': 'data/instances_train2024.jsonl', 
                'validation': 'data/instances_val2024.jsonl'},
    features=features
)

# Test all image paths are valid
import os
for example in dataset['train']:
    assert os.path.exists(example['image_path']), f"Image path {example['image_path']} does not exist"

# Test all bounding boxes are within the image
for example in dataset['train']:
    for bbox, width, height in zip(example['objects']['bbox'], example['width'], example['height']):
        assert 0 <= bbox[0] <= width, f"Bounding box {bbox} is outside the image"
        assert 0 <= bbox[1] <= height, f"Bounding box {bbox} is outside the image"
        assert 0 <= bbox[0] + bbox[2] <= width, f"Bounding box {bbox} is outside the image"
        assert 0 <= bbox[1] + bbox[3] <= height, f"Bounding box {bbox} is outside the image"

# Test all bounding boxes are valid
for example in dataset['train']:
    for bbox in example['objects']['bbox']:
        assert bbox[2] > 0, f"Bounding box {bbox} has width <= 0"
        assert bbox[3] > 0, f"Bounding box {bbox} has height <= 0"

