from datasets import Features, Value, Sequence
from datasets import load_dataset
from transformers import AutoImageProcessor
import numpy as np
import os
from PIL import Image, ImageDraw
import albumentations
import numpy as np
import torch

from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer

import json

from transformers import pipeline
import requests

import torchvision

from transformers import AutoModelForObjectDetection, AutoImageProcessor
import evaluate
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt
from transformers import pipeline, AutoImageProcessor, AutoModelForObjectDetection
from torchvision.ops import box_iou

############# Data Loading #############

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


dataset = load_dataset(
    'json', 
    data_files={'train': 'data/instances_train2024.jsonl', 
                'validation': 'data/instances_val2024.jsonl'},
    features=features
)

print('\n\n')
print(dataset["train"][0])
print('\n\n')

def load_classes():
    class_list = []
    with open("data/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()



image = Image.open(dataset["train"][0]["image_path"])
annotations = dataset["train"][0]["objects"]
draw = ImageDraw.Draw(image)

# categories = dataset["train"].features["objects"].feature["category"].names

id2label = {index: x for index, x in enumerate(class_list, start=0)}
label2id = {v: k for k, v in id2label.items()}

for i in range(len(annotations["id"])):
    box = annotations["bbox"][i - 1]
    class_idx = annotations["category"][i - 1]
    x, y, w, h = tuple(box)
    draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
    draw.text((x, y), id2label[class_idx], fill="white")

# save the image
image.save("visualize_anno.jpg")

############# Preprocessing #############

checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        # albumentations.HorizontalFlip(p=1.0),
        albumentations.HorizontalFlip(p=0.5),
        # albumentations.RandomBrightnessContrast(p=1.0),
        albumentations.RandomBrightnessContrast(p=0.5),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)

def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations

# Create an empty placeholder image
def create_empty_image(width=640, height=480, color=(0, 0, 0)):
    return np.zeros((height, width, 3), dtype=np.uint8)  # Black image by default


# transforming a batch
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image_path, objects in zip(examples["image_path"], examples["objects"]):
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f'{image_path} does not exist, using a placeholder image.')

            # Try opening the image
            image = Image.open(image_path)
            image = np.array(image.convert("RGB"))[:, :, ::-1]
        
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(e)
            # Use a black placeholder image if the actual image is missing or cannot be opened
            image = create_empty_image()
        
        # pass_bbox = False
        # for bbox in objects["bbox"]:
        #     x_min, y_min, w, h = bbox
        #     if x_min < 0 or y_min < 0 or w <= 0 or h <= 0 or x_min + w > image.shape[1] or y_min + h > image.shape[0]:
        #         pass_bbox = True
        
        # if pass_bbox:
        #     area.append(objects["area"])
        #     images.append(image)
        #     bboxes.append(objects["bbox"])
        #     categories.append(objects["category"])

        # else:
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")

dataset["train"] = dataset["train"].with_transform(transform_aug_ann)
print('\n\n')
print(dataset["train"][15])
print('\n\n')

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    # encoding = image_processor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

############# Training #############

model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir="detr-resnet-50-finetuned-100-epochs-lifebuoy-dataset",
    per_device_train_batch_size=8,
    num_train_epochs=100,
    fp16=False,
    save_steps=5 * len(dataset["train"]) // 8,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
    hub_model_id="ARG-NCTU/detr-resnet-50-finetuned-100-epochs-lifebuoy-dataset",
)


# Function to find the latest checkpoint
def get_latest_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        # Sort checkpoints based on the epoch number and return the latest one
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        latest_checkpoint = checkpoints[-1]
        print(f"Resuming from the latest checkpoint: {latest_checkpoint}")
        return os.path.join(output_dir, latest_checkpoint)
    else:
        return None


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=collate_fn,
#     train_dataset=dataset["train"],
#     tokenizer=image_processor,
# )

# Custom Trainer class to handle custom push logic
class CustomTrainer(Trainer):
    def on_epoch_end(self):
        super().on_epoch_end()
        # Push the model to the hub every 5 epochs
        if self.state.epoch % 5 == 0:
            print(f"Pushing model to the hub at epoch {self.state.epoch}...")
            self.push_to_hub(commit_message=f"Checkpoint at epoch {int(self.state.epoch)}")

# Initialize Trainer with collate function, etc.
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=collate_fn,
    tokenizer=image_processor,
)

# Check if any checkpoint exists in the output directory
latest_checkpoint = get_latest_checkpoint(training_args.output_dir)

# Resume training from the latest checkpoint or start from scratch
if latest_checkpoint:
    print("Resuming from the latest checkpoint...")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("Starting training from scratch...")
    trainer.train()

trainer.push_to_hub(commit_message='detr-resnet-50-finetuned-lifebuoy-dataset 100 epoch')

############# Evaluation #############

# format annotations the same as for training, no need for data augmentation
def val_formatted_anns(image_id, objects):
    annotations = []
    for i in range(0, len(objects["id"])):
        new_ann = {
            "id": objects["id"][i],
            "category_id": objects["category"][i],
            "iscrowd": 0,
            "image_id": image_id,
            "area": objects["area"][i],
            "bbox": objects["bbox"][i],
        }
        annotations.append(new_ann)

    return annotations


# Save images and annotations into the files torchvision.datasets.CocoDetection expects
def save_annotation_file_images(dataset, mode="val"):
    output_json = {}
    path_output = f"{os.getcwd()}/output/"

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    if mode == "val":
        path_anno = os.path.join(path_output, "lifebuoy_ann_val.json")
    else:
        path_anno = os.path.join(path_output, "lifebuoy_ann_val_real.json")
    categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
    output_json["images"] = []
    output_json["annotations"] = []
    for example in dataset:
        ann = val_formatted_anns(example["image_id"], example["objects"])
        if not os.path.exists(example["image_path"]):
            continue
        image_example = Image.open(example["image_path"])
        output_json["images"].append(
            {
                "id": example["image_id"],
                "width": image_example.width,
                "height": image_example.height,
                "file_name": f"{example['image_id']}.png",
            }
        )
        output_json["annotations"].extend(ann)
    output_json["categories"] = categories_json

    with open(path_anno, "w") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    for image_path, img_id in zip(dataset["image_path"], dataset["image_id"]):
        if not os.path.exists(image_path):
            continue
        im = Image.open(image_path)
        path_img = os.path.join(path_output, f"{img_id}.png")
        im.save(path_img)

    return path_output, path_anno

os.system("pip3 install pycocotools")


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, ann_file):
        super().__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target: converting target to DETR format,
        # resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return {"pixel_values": pixel_values, "labels": target}

def eval(eval_dataset, mode="val"):
    im_processor = AutoImageProcessor.from_pretrained("ARG-NCTU/detr-resnet-50-finetuned-100-epochs-lifebuoy-dataset")
    path_output, path_anno = save_annotation_file_images(eval_dataset, mode)
    test_ds_coco_format = CocoDetection(path_output, im_processor, path_anno)
    
    model = AutoModelForObjectDetection.from_pretrained("ARG-NCTU/detr-resnet-50-finetuned-100-epochs-lifebuoy-dataset")
    module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
    val_dataloader = torch.utils.data.DataLoader(
        test_ds_coco_format, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            pixel_values = batch["pixel_values"]
            pixel_mask = batch["pixel_mask"]
    
            labels = [
                {k: v for k, v in t.items()} for t in batch["labels"]
            ]  # these are in DETR format, resized + normalized
    
            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            results = im_processor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api
    
            module.add(prediction=results, reference=labels)
            del batch
    
    results = module.compute()
    print(results)

eval(dataset["validation"], mode="val")

############# Inference #############

# url = "https://i.imgur.com/2lnWoly.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

image = Image.open(dataset["validation"][0]["image_path"])

image_processor = AutoImageProcessor.from_pretrained("ARG-NCTU/detr-resnet-50-finetuned-lifebuoy-dataset")
model = AutoModelForObjectDetection.from_pretrained("ARG-NCTU/detr-resnet-50-finetuned-lifebuoy-dataset")

with torch.no_grad():
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )

draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x, y, x2, y2 = tuple(box)
    draw.rectangle((x, y, x2, y2), outline="red", width=1)
    draw.text((x, y), model.config.id2label[label.item()], fill="white")

# save the image
path = dataset["validation"][0]["image_path"]
image.save(f"{path}_out.jpg")

############# Confusion Matrix #############

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the object detector model and move to GPU
image_processor = AutoImageProcessor.from_pretrained("ARG-NCTU/detr-resnet-50-finetuned-100-epochs-lifebuoy-dataset")
model = AutoModelForObjectDetection.from_pretrained("ARG-NCTU/detr-resnet-50-finetuned-100-epochs-lifebuoy-dataset").to(device)

# Initialize confusion matrix counts
TP = FP = TN = FN = 0

# Loop through the validation dataset to calculate confusion matrix for each image
for i in range(len(dataset["validation"])):
    if not os.path.exists(dataset["validation"][i]["image_path"]):
        continue
    # Load the image
    image = Image.open(dataset["validation"][i]["image_path"])

    # Perform inference
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt").to(device)  # Move inputs to GPU
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(device)  # Move target sizes to GPU
        results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    # Ground truth bounding boxes in [x1, y1, w, h] format
    gt_boxes = dataset["validation"][i]["objects"]["bbox"]  # True bounding boxes for the current image
    
    # Convert ground truth boxes from [x1, y1, w, h] to [x1, y1, x2, y2]
    ground_truth_boxes = []
    for box in gt_boxes:
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
        ground_truth_boxes.append([x1, y1, x2, y2])
    
    ground_truth_boxes = torch.tensor(ground_truth_boxes).to(device)  # Convert to tensor and move to GPU
    
    # Ground truth labels: 0 = lifebuoy, other categories = background
    ground_truth_labels = torch.tensor([category for category in dataset["validation"][i]["objects"]["category"]]).to(device)

    # Convert predicted boxes to tensor and move to GPU
    pred_boxes = torch.tensor([box.tolist() for box in results['boxes']]).to(device)
    
    # Calculate IoU between predicted and ground truth boxes
    if len(pred_boxes) > 0 and len(ground_truth_boxes) > 0:
        iou_matrix = box_iou(pred_boxes, ground_truth_boxes)

        # Iterate over each detection and compare with ground truth
        for j, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            max_iou, gt_idx = torch.max(iou_matrix[j], dim=0)
            # print(max_iou)
            if max_iou > 0.5:  # IoU threshold
                if label == ground_truth_labels[gt_idx]:  # Correct label (True Positive)
                    TP += 1
                else:  # Incorrect label (False Positive)
                    FP += 1
            else:
                FN += 1  # Lifebuoy not detected correctly (False Negative)
    
    # If there are no predicted bounding boxes, count all as False Negatives
    if len(pred_boxes) == 0 and len(ground_truth_boxes) > 0:
        FN += len(ground_truth_boxes)

# Calculate TN (True Negative)
TN = 0  # Assuming all images without lifebuoys are true negatives

# Create confusion matrix
confusion_matrix = [[TP, FP], [FN, TN]]

# Plot the confusion matrix
plt.figure(figsize=(6,6))
plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = [0, 1]
plt.xticks(tick_marks, ['Lifebuoy', 'Background'])
plt.yticks(tick_marks, ['Pred Lifebuoy', 'Pred Background'])

# Annotate the confusion matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix[i][j], horizontalalignment='center', color='black')

plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
plt.savefig('confusion_mat.png')

