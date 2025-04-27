# map_calculator.py
import numpy as np
from collections import defaultdict

def calculate_iou(box1, box2):
    """
    두 바운딩 박스 간의 IoU(Intersection over Union)를 계산합니다.
    box 형식: [xmin, ymin, width, height]
    """
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2 = x1_1 + w1, y1_1 + h1
    x2_1, y2_1, w2, h2 = box2
    x2_2, y2_2 = x2_1 + w2, y2_1 + h2

    # 교차 영역 좌표 계산
    xi_1 = max(x1_1, x2_1)
    yi_1 = max(y1_1, y2_1)
    xi_2 = min(x1_2, x2_2)
    yi_2 = min(y1_2, y2_2)

    # 교차 영역 넓이
    inter_width = max(0, xi_2 - xi_1)
    inter_height = max(0, yi_2 - yi_1)
    inter_area = inter_width * inter_height

    # 각 박스 넓이
    area1 = w1 * h1
    area2 = w2 * h2

    # 합집합 넓이
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    # IoU 계산
    iou = inter_area / union_area
    return iou

def calculate_ap(rec, prec):
    """Average Precision (AP)를 계산합니다."""
    # Precision-Recall 곡선 아래 면적 계산 (11점 보간법 또는 모든 점 사용)
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # Precision 값을 오른쪽에서 왼쪽으로 가면서 누적 최댓값으로 만듭니다.
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    # 구간별 면적 계산
    i = np.where(mrec[1:] != mrec[:-1])[0] # Recall 값이 변하는 지점
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calculate_map(gt_annotations_img, pred_annotations_img, categories, iou_threshold=0.5):
    """
    단일 이미지 또는 전체 데이터셋에 대한 mAP를 계산합니다.
    (여기서는 단일 이미지 처리를 가정하고 단순화된 예시를 제공합니다.
     실제 COCO mAP는 여러 IoU 임계값과 클래스에 대해 계산됩니다.)

    Args:
        gt_annotations_img (list): 특정 이미지에 대한 GT annotation 리스트.
                                    각 annotation은 {'bbox': [...], 'category_id': ...} 포함.
        pred_annotations_img (list): 특정 이미지에 대한 예측 annotation 리스트.
                                     각 annotation은 {'bbox': [...], 'category_id': ..., 'score': ...} 포함.
        categories (dict): 카테고리 ID와 정보를 매핑하는 딕셔너리.
        iou_threshold (float): TP/FP 판정을 위한 IoU 임계값.

    Returns:
        float: 계산된 mAP 값 (여기서는 단일 클래스 AP 또는 단순 평균 AP).
        dict: 클래스별 AP 값.
    """
    aps = {}
    if not categories:
        print("오류: 카테고리 정보가 없습니다.")
        return 0.0, {}

    for category_id, category_info in categories.items():
        # 현재 클래스에 해당하는 GT와 예측 필터링
        gt = [ann for ann in gt_annotations_img if ann['category_id'] == category_id]
        preds = [pred for pred in pred_annotations_img if pred['category_id'] == category_id]

        if not gt and not preds:
            continue # 이 클래스에 대한 GT와 예측이 모두 없으면 건너뜀
        if not preds:
            aps[category_id] = 0.0 # 예측이 없으면 AP는 0
            continue
        if not gt:
             # GT는 없는데 예측만 있는 경우, 모든 예측은 FP
             # 이 경우 AP 계산 방식에 따라 다를 수 있으나, 보통 0으로 처리
             aps[category_id] = 0.0
             continue


        # 예측을 score 기준으로 내림차순 정렬
        preds.sort(key=lambda x: x['score'], reverse=True)

        nd = len(gt) # 해당 클래스의 총 GT 개수
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        gt_matched = np.zeros(len(gt)) # 각 GT가 매칭되었는지 추적

        for i, pred in enumerate(preds):
            best_iou = 0.0
            best_gt_idx = -1

            # 현재 예측과 가장 IoU가 높은 GT 찾기
            for j, gt_ann in enumerate(gt):
                iou = calculate_iou(pred['bbox'], gt_ann['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            # IoU 임계값을 넘고, 해당 GT가 아직 다른 예측과 매칭되지 않았다면 TP
            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                tp[i] = 1.
                gt_matched[best_gt_idx] = 1
            else:
                fp[i] = 1.

        # Precision, Recall 계산
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        rec = tp_cumsum / (nd + 1e-10) # Recall
        prec = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10) # Precision

        # AP 계산
        ap = calculate_ap(rec, prec)
        aps[category_id] = ap

    # mAP 계산 (모든 클래스 AP의 평균)
    mean_ap = np.mean(list(aps.values())) if aps else 0.0

    return mean_ap, aps

# 예시 사용법 (테스트용)
if __name__ == '__main__':
    # 가상의 데이터 생성
    gt_ann = [
        {'bbox': [10, 10, 50, 50], 'category_id': 1},
        {'bbox': [100, 100, 60, 60], 'category_id': 1},
        {'bbox': [200, 200, 70, 70], 'category_id': 2},
    ]
    pred_ann = [
        {'bbox': [12, 12, 48, 48], 'category_id': 1, 'score': 0.9},
        {'bbox': [110, 110, 55, 55], 'category_id': 1, 'score': 0.8},
        {'bbox': [50, 50, 30, 30], 'category_id': 1, 'score': 0.7}, # FP
        {'bbox': [210, 210, 65, 65], 'category_id': 2, 'score': 0.95},
        {'bbox': [300, 300, 40, 40], 'category_id': 3, 'score': 0.88}, # 다른 클래스 예측 (GT 없음)
    ]
    categories_test = {
        1: {'id': 1, 'name': 'cat'},
        2: {'id': 2, 'name': 'dog'},
        3: {'id': 3, 'name': 'person'} # GT에는 없지만 예측에는 있는 클래스
    }

    iou_thresh = 0.5
    mean_ap, aps = calculate_map(gt_ann, pred_ann, categories_test, iou_thresh)

    print(f"--- mAP 계산 결과 (IoU={iou_thresh}) ---")
    print(f"mAP: {mean_ap:.4f}")
    print("클래스별 AP:")
    for cat_id, ap in aps.items():
        cat_name = categories_test.get(cat_id, {}).get('name', 'Unknown')
        print(f"  - {cat_name} (ID: {cat_id}): {ap:.4f}")

    # IoU 계산 테스트
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 10, 10]
    iou = calculate_iou(box1, box2)
    print(f"\nIoU 계산 테스트: box1={box1}, box2={box2} -> IoU = {iou:.4f} (예상: 0.1429)")
    box3 = [0, 0, 10, 10]
    box4 = [0, 0, 10, 10]
    iou = calculate_iou(box3, box4)
    print(f"IoU 계산 테스트: box3={box3}, box4={box4} -> IoU = {iou:.4f} (예상: 1.0)")
    box5 = [0, 0, 10, 10]
    box6 = [20, 20, 10, 10]
    iou = calculate_iou(box5, box6)
    print(f"IoU 계산 테스트: box5={box5}, box6={box6} -> IoU = {iou:.4f} (예상: 0.0)")