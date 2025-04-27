# coco_loader.py
import json
import os
from collections import defaultdict

def load_coco_annotations(filepath):
    """COCO 형식의 Ground Truth annotation 파일을 로드합니다."""
    if not os.path.exists(filepath):
        print(f"오류: GT annotation 파일을 찾을 수 없습니다 - {filepath}")
        return None, None, None
    try:
        with open(filepath, 'r') as f:
            coco_data = json.load(f)

        images = {img['id']: img for img in coco_data.get('images', [])}
        annotations = defaultdict(list)
        for ann in coco_data.get('annotations', []):
            annotations[ann['image_id']].append(ann)
        categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
        print(f"GT 로드 완료: {len(images)}개 이미지, {len(coco_data.get('annotations', []))}개 annotation")
        return images, annotations, categories
    except Exception as e:
        print(f"오류: GT annotation 파일 로드 중 오류 발생 - {e}")
        return None, None, None

def load_predictions(filepath):
    """모델 예측 annotation 파일을 로드합니다."""
    if not os.path.exists(filepath):
        print(f"오류: 예측 annotation 파일을 찾을 수 없습니다 - {filepath}")
        return None
    try:
        with open(filepath, 'r') as f:
            predictions = json.load(f)
        # 예측 결과를 image_id 기준으로 그룹화
        predictions_by_image = defaultdict(list)
        for pred in predictions:
            # 입력 요구사항 형식에 맞게 변환 (필요시)
            formatted_pred = {
                "image_id": pred.get("image_id"),
                "category_id": int(pred.get("category_id")),
                "bbox": [float(c) for c in pred.get("bbox", [])],
                "score": float(pred.get("score")),
                # 필요하다면 원본 ID나 다른 정보 추가
                "id": pred.get("id", None) # 수정/삭제 등을 위해 고유 ID가 있으면 유용
            }
            if formatted_pred["image_id"] is not None:
                 predictions_by_image[formatted_pred["image_id"]].append(formatted_pred)

        print(f"예측 로드 완료: {len(predictions)}개 예측")
        return predictions_by_image
    except Exception as e:
        print(f"오류: 예측 annotation 파일 로드 중 오류 발생 - {e}")
        return None

def get_image_path(image_info, image_dir):
    """이미지 정보와 디렉토리 경로를 받아 이미지 파일 경로를 반환합니다."""
    if not image_info or 'file_name' not in image_info:
        return None
    return os.path.join(image_dir, image_info['file_name'])

# 예시 사용법 (테스트용)
if __name__ == '__main__':
    gt_file = 'annotations/instances_val2017.json'
    pred_file = 'faster_rcnn_predictions.json'
    image_dir = 'val2017'

    images, gt_annotations, categories = load_coco_annotations(gt_file)
    predictions = load_predictions(pred_file)

    if images and gt_annotations and predictions:
        # 특정 이미지에 대한 정보 가져오기 (예: 첫 번째 이미지)
        example_image_id = next(iter(images))
        print(f"\n--- 예시 이미지 ID: {example_image_id} ---")

        # GT 정보
        print("GT Annotations:")
        for ann in gt_annotations.get(example_image_id, []):
            print(f"  - Category: {categories.get(ann['category_id'], {}).get('name', 'Unknown')}, BBox: {ann['bbox']}")

        # 예측 정보
        print("Predictions:")
        for pred in predictions.get(example_image_id, []):
             print(f"  - Category ID: {pred['category_id']}, BBox: {pred['bbox']}, Score: {pred['score']:.4f}")

        # 이미지 경로
        image_path = get_image_path(images.get(example_image_id), image_dir)
        print(f"Image Path: {image_path}")
