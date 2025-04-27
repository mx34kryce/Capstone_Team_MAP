import os
import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

def get_predictions_coco_format(model, coco_dataset_path, image_folder, output_path, confidence_threshold=0.7):
    """
    Use a pre-trained Faster R-CNN model to get predictions on COCO dataset
    and save them in COCO annotation format.
    
    Args:
        model: Pre-trained Faster R-CNN model
        coco_dataset_path: Path to COCO dataset annotations (needed for image ids)
        image_folder: Path to the folder containing images
        output_path: Path to save the output annotation file
        confidence_threshold: Threshold for filtering predictions by confidence
    """
    # Load COCO dataset to get image info (need image IDs)
    coco = COCO(coco_dataset_path)
    image_ids = list(coco.imgs.keys())
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # COCO categories (same as COCO dataset)
    categories = [
        {"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}, {"id": 3, "name": "car"}, 
        # ... add all 80 COCO categories here
    ]
    
    # Initialize results list
    results = []
    
    # Process each image
    for img_id in tqdm(image_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_folder, img_info['file_name'])
        
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            img_tensor = torchvision.transforms.functional.to_tensor(image).to(device)
            
            # Make prediction
            with torch.no_grad():
                prediction = model([img_tensor])
            
            # Process each detected object
            boxes = prediction[0]['boxes'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()
            
            # Filter by confidence threshold
            keep = scores >= confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # Convert to COCO format (xmin, ymin, width, height)
            for box, score, label in zip(boxes, scores, labels):
                xmin, ymin, xmax, ymax = box
                width = xmax - xmin
                height = ymax - ymin
                
                # Create annotation in COCO format
                annotation = {
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [float(xmin), float(ymin), float(width), float(height)],
                    "score": float(score),
                }
                results.append(annotation)
        
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    # Save results to file
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Saved {len(results)} predictions to {output_path}")

def main():
    # Paths
    coco_annotation_path = "/home/porsche3/Jong/capstone/coco2017/annotations/captions_val2017.json"  # Change to your annotation path
    coco_images_folder = "/home/porsche3/Jong/capstone/coco2017/val2017"  # Change to your image folder
    output_annotation_path = "/home/porsche3/Jong/capstone/faster_rcnn_predictions.json"  # Output file
    
    # Load pre-trained Faster R-CNN
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    # Get predictions and save to file
    get_predictions_coco_format(
        model=model,
        coco_dataset_path=coco_annotation_path,
        image_folder=coco_images_folder,
        output_path=output_annotation_path,
        confidence_threshold=0.7
    )

if __name__ == "__main__":
    main()