import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

def show_annotation(json_file, img_dir, index=0):
    """
    COCO 데이터셋의 예측 결과 중 하나만 시각화하는 함수
    
    Args:
        json_file (str): 예측 결과가 저장된 JSON 파일 경로
        img_dir (str): 이미지가 저장된 디렉토리 경로
        index (int): 시각화할 예측 결과의 인덱스 (기본값: 0)
    """
    # JSON 파일에서 예측 결과 로드
    with open(json_file, 'r') as f:
        predictions = json.load(f)
    
    # 특정 인덱스의 예측 결과 가져오기
    if index >= len(predictions):
        print(f"인덱스가 범위를 벗어납니다. 전체 예측 결과 수: {len(predictions)}")
        return
    
    prediction = predictions[index]
    
    # 이미지 ID 가져오기
    image_id = prediction.get('image_id')
    if not image_id:
        print("예측 결과에 image_id가 없습니다.")
        return
    
    # COCO 형식의 이미지 파일명 (예: 000000123456.jpg)
    image_file = os.path.join(img_dir, f"{int(image_id):012d}.jpg")
    
    if not os.path.exists(image_file):
        print(f"이미지 파일을 찾을 수 없습니다: {image_file}")
        return
    
    # 이미지 로드
    image = Image.open(image_file)
    
    # 시각화 준비
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # COCO 클래스 레이블 (필요에 따라 수정)
    coco_classes = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
        # 나머지 클래스들도 필요에 따라 추가
    }
    
    # 예측 결과 시각화 (경계 상자, 클래스, 신뢰도 점수)
    if 'bbox' in prediction and 'category_id' in prediction:
        bbox = prediction['bbox']  # [x, y, width, height] 형식
        category_id = prediction['category_id']
        score = prediction.get('score', 0)
        
        # 경계 상자 그리기
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 클래스 이름과 신뢰도 점수 표시
        class_name = coco_classes.get(category_id, f"Class {category_id}")
        ax.text(
            bbox[0], bbox[1] - 5,
            f"{class_name}: {score:.2f}",
            color='white', fontsize=12, backgroundcolor='red'
        )
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return fig

# 실행 예시
if __name__ == "__main__":
    # 파일 경로 설정
    predictions_file = "/home/porsche3/Jong/capstone/faster_rcnn_predictions.json"
    image_directory = "/home/porsche3/Jong/capstone/coco2017/val2017"  # 실제 이미지 디렉토리 경로로 수정 필요
    
    # 첫 번째 예측 결과 시각화
    show_annotation(predictions_file, image_directory, index=0)
    def save_annotation(json_file, img_dir, output_dir, index=0):
        """
        COCO 데이터셋의 예측 결과를 시각화하고 저장하는 함수
        
        Args:
            json_file (str): 예측 결과가 저장된 JSON 파일 경로
            img_dir (str): 이미지가 저장된 디렉토리 경로
            output_dir (str): 시각화된 이미지를 저장할 디렉토리 경로
            index (int): 시각화할 예측 결과의 인덱스 (기본값: 0)
        """
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 시각화 실행
        fig = show_annotation(json_file, img_dir, index)
        
        if fig:
            # 이미지 ID 가져오기
            with open(json_file, 'r') as f:
                predictions = json.load(f)
            
            prediction = predictions[index]
            image_id = prediction.get('image_id')
            
            # 파일 저장
            output_path = os.path.join(output_dir, f"annotation_{int(image_id):012d}.png")
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)
            print(f"이미지가 저장되었습니다: {output_path}")
            return output_path
        
        return None

    # 함수 사용 예시
    if __name__ == "__main__":
        # 파일 경로 설정
        predictions_file = "/home/porsche3/Jong/capstone/faster_rcnn_predictions.json"
        image_directory = "/home/porsche3/Jong/capstone/coco2017/val2017"
        output_directory = "/home/porsche3/Jong/capstone/output_annotations"
        
        # 예측 결과 시각화 및 저장
        save_annotation(predictions_file, image_directory, output_directory, index=0)
