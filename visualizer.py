# visualizer.py
from PIL import Image, ImageDraw, ImageFont
import random

# 색상 팔레트 (카테고리별로 다른 색상 사용 위함)
# https://material.io/design/color/the-color-system.html#tools-for-picking-colors
DEFAULT_COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
    '#C00000', '#00C000', '#0000C0', '#C0C000', '#C000C0', '#00C0C0',
    '#400000', '#004000', '#000040', '#404000', '#400040', '#004040',
    '#FFA500', '#FFD700', '#ADFF2F', '#7FFF00', '#00FA9A', '#00CED1',
    '#1E90FF', '#DA70D6', '#FF69B4', '#FF1493', '#DC143C', '#A52A2A',
]

def get_color(category_id):
    """카테고리 ID에 따라 색상을 반환합니다."""
    idx = category_id % len(DEFAULT_COLORS)
    return DEFAULT_COLORS[idx]

def draw_annotations(image_path, gt_annotations, pred_annotations, categories,
                     confidence_threshold=0.5, iou_threshold_for_match=0.5,
                     show_gt=True, show_pred=True):
    """
    이미지에 Ground Truth와 예측 Annotation을 그립니다.

    Args:
        image_path (str): 이미지 파일 경로.
        gt_annotations (list): GT annotation 리스트.
        pred_annotations (list): 예측 annotation 리스트.
        categories (dict): 카테고리 정보 딕셔너리.
        confidence_threshold (float): 표시할 예측의 최소 confidence 점수.
        iou_threshold_for_match (float): GT와 예측 매칭 확인용 IoU 임계값 (시각화용).
        show_gt (bool): GT 박스 표시 여부.
        show_pred (bool): 예측 박스 표시 여부.

    Returns:
        PIL.Image.Image: Annotation이 그려진 이미지 객체.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        # 폰트 설정 (시스템에 따라 경로 수정 필요)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        # GT 그리기
        if show_gt and gt_annotations:
            for ann in gt_annotations:
                bbox = ann['bbox']
                cat_id = ann['category_id']
                cat_info = categories.get(cat_id, {})
                label = cat_info.get('name', f'ID:{cat_id}')
                color = get_color(cat_id)

                xmin, ymin, w, h = bbox
                xmax, ymax = xmin + w, ymin + h
                # GT 박스는 실선으로 그림
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
                text = f"GT: {label}"
                text_bbox = draw.textbbox((xmin, ymin), text, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((xmin, ymin), text, fill="white", font=font)

        # 예측 그리기
        if show_pred and pred_annotations:
            # 예측을 score 기준으로 내림차순 정렬 (겹칠 때 높은 score가 위에 오도록)
            pred_annotations.sort(key=lambda x: x['score'], reverse=True)

            for pred in pred_annotations:
                if pred['score'] >= confidence_threshold:
                    bbox = pred['bbox']
                    cat_id = pred['category_id']
                    score = pred['score']
                    cat_info = categories.get(cat_id, {})
                    label = cat_info.get('name', f'ID:{cat_id}')
                    color = get_color(cat_id)

                    xmin, ymin, w, h = bbox
                    xmax, ymax = xmin + w, ymin + h
                    # 예측 박스는 점선으로 그림
                    # 점선 구현이 복잡하므로 여기서는 다른 두께 또는 스타일로 구분
                    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3) # 약간 더 두껍게

                    text = f"Pred: {label} ({score:.2f})"
                    text_bbox = draw.textbbox((xmin, ymax - 15 if ymax > 15 else ymax), text, font=font) # 박스 아래쪽에 표시
                    draw.rectangle(text_bbox, fill=color)
                    draw.text((xmin, ymax - 15 if ymax > 15 else ymax), text, fill="black", font=font)

        return image

    except FileNotFoundError:
        print(f"오류: 이미지 파일을 찾을 수 없습니다 - {image_path}")
        # 빈 이미지 또는 오류 이미지 반환
        return Image.new('RGB', (300, 200), color = 'grey')
    except Exception as e:
        print(f"오류: 이미지에 annotation 그리는 중 오류 발생 - {e}")
        return Image.new('RGB', (300, 200), color = 'grey')

def draw_pr_curve(precision, recall, title="Precision-Recall Curve"):
    """
    Precision-Recall Curve를 그립니다. (matplotlib 사용 예시)
    GUI 환경에서는 Tkinter Canvas에 직접 그리거나 matplotlib의 FigureCanvasTkAgg를 사용할 수 있습니다.
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        # plt.show() # GUI 환경에서는 show() 대신 Figure 객체를 반환하거나 Canvas에 그려야 함
        # 여기서는 임시로 Figure 객체를 반환하는 형태로 가정
        fig = plt.gcf()
        return fig
    except ImportError:
        print("오류: matplotlib 라이브러리가 설치되지 않았습니다. PR Curve 시각화를 사용하려면 설치해주세요.")
        return None
    except Exception as e:
        print(f"오류: PR Curve 그리는 중 오류 발생 - {e}")
        return None

# 예시 사용법 (테스트용)
if __name__ == '__main__':
    # 가상의 데이터 및 경로 설정 (실제 경로로 변경 필요)
    img_path = 'val2017/000000000139.jpg' # 실제 존재하는 이미지 경로 사용
    if not os.path.exists(img_path):
        print(f"테스트 이미지 없음: {img_path}. 빈 이미지를 사용합니다.")
        img_path = None
        # 임시 빈 이미지 생성
        temp_img = Image.new('RGB', (640, 480), color = 'white')
        temp_img.save("temp_test_image.jpg")
        img_path = "temp_test_image.jpg"


    gt_ann_test = [
        {'bbox': [100, 100, 150, 150], 'category_id': 1},
        {'bbox': [300, 200, 100, 80], 'category_id': 2},
    ]
    pred_ann_test = [
        {'bbox': [110, 110, 140, 140], 'category_id': 1, 'score': 0.95},
        {'bbox': [310, 210, 90, 70], 'category_id': 2, 'score': 0.88},
        {'bbox': [50, 50, 60, 60], 'category_id': 1, 'score': 0.6}, # 낮은 confidence
        {'bbox': [400, 100, 50, 50], 'category_id': 3, 'score': 0.7}, # 다른 클래스
    ]
    categories_test = {
        1: {'id': 1, 'name': 'ObjectA'},
        2: {'id': 2, 'name': 'ObjectB'},
        3: {'id': 3, 'name': 'ObjectC'}
    }

    # Annotation 그리기 테스트
    annotated_image = draw_annotations(img_path, gt_ann_test, pred_ann_test, categories_test,
                                       confidence_threshold=0.7, show_gt=True, show_pred=True)

    if annotated_image:
        annotated_image.save("annotated_image_test.jpg")
        print("Annotation이 그려진 이미지가 annotated_image_test.jpg 로 저장되었습니다.")
        # annotated_image.show() # 로컬에서 직접 실행 시 이미지 보기

    # PR Curve 그리기 테스트 (가상 데이터)
    recall_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    precision_test = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55])
    pr_figure = draw_pr_curve(precision_test, recall_test)
    if pr_figure:
        pr_figure.savefig("pr_curve_test.png")
        print("PR Curve 이미지가 pr_curve_test.png 로 저장되었습니다.")

    # 임시 이미지 파일 삭제
    if os.path.exists("temp_test_image.jpg"):
        os.remove("temp_test_image.jpg")
