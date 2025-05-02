# interactive_canvas.py
import tkinter as tk
from PIL import Image, ImageTk
import math # 크기 조절 시 거리 계산 등에 사용 가능

# visualizer 모듈의 색상 함수 재사용 또는 여기서 정의
DEFAULT_COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
    '#C00000', '#00C000', '#0000C0', '#C0C000', '#C000C0', '#00C0C0',
    '#400000', '#004000', '#000040', '#404000', '#400040', '#004040',
    '#FFA500', '#FFD700', '#ADFF2F', '#7FFF00', '#00FA9A', '#00CED1',
    '#1E90FF', '#DA70D6', '#FF69B4', '#FF1493', '#DC143C', '#A52A2A',
]
GT_COLOR_OFFSET = 5 # GT 색상을 예측과 다르게 하기 위한 오프셋 (선택적)

def get_color(category_id, is_gt=False):
    """카테고리 ID에 따라 색상을 반환합니다. GT 여부에 따라 오프셋 적용 가능."""
    offset = GT_COLOR_OFFSET if is_gt else 0
    idx = (category_id + offset) % len(DEFAULT_COLORS)
    return DEFAULT_COLORS[idx]

class InteractiveCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.image_on_canvas = None
        self.tk_image = None
        self.gt_annotations = []    # Ground Truth annotations
        self.pred_annotations = []  # Prediction annotations (수정 가능)
        self.categories = {}
        self.confidence_threshold = 0.5
        self.visible_class_ids = set() # 현재 보여줄 클래스 ID 집합

        # 드래그 상태 변수
        self._drag_data = {"x": 0, "y": 0, "item": None, "type": None, "ann_idx": -1, "is_gt": False, "original_bbox": None}
        self._selected_pred_idx = -1 # 현재 선택된 '예측' annotation의 인덱스

        # 이벤트 바인딩
        self.bind("<ButtonPress-1>", self.on_button_press)
        self.bind("<B1-Motion>", self.on_move_press)
        self.bind("<ButtonRelease-1>", self.on_button_release)
        self.bind("<Motion>", self.on_mouse_motion) # 마우스 커서 변경용

        # 콜백 함수 (annotation 변경 시 호출)
        self.annotation_update_callback = None

    def set_annotation_update_callback(self, callback):
        self.annotation_update_callback = callback

    def load_image(self, image_path):
        try:
            pil_image = Image.open(image_path).convert("RGB")
            # Canvas 크기에 맞게 이미지 리사이즈 (선택 사항, 비율 유지하며)
            # canvas_w, canvas_h = self.winfo_width(), self.winfo_height()
            # if canvas_w > 1 and canvas_h > 1: # 초기 크기 1x1 방지
            #     img_w, img_h = pil_image.size
            #     ratio = min(canvas_w / img_w, canvas_h / img_h)
            #     new_w, new_h = int(img_w * ratio), int(img_h * ratio)
            #     pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

            self.tk_image = ImageTk.PhotoImage(pil_image)
            if self.image_on_canvas:
                self.delete(self.image_on_canvas)
            # 이미지를 캔버스 중앙에 배치 (선택적)
            # x_pos = (self.winfo_width() - self.tk_image.width()) // 2
            # y_pos = (self.winfo_height() - self.tk_image.height()) // 2
            # self.image_on_canvas = self.create_image(x_pos, y_pos, anchor="nw", image=self.tk_image)
            self.image_on_canvas = self.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.config(scrollregion=self.bbox(self.image_on_canvas)) # 스크롤 영역 설정

            # 이미지가 로드되면 기존 annotation 삭제
            self.clear_annotations()
        except Exception as e:
            print(f"Error: Image load failed - {e}")
            self.delete("all")
            self.tk_image = None
            self.image_on_canvas = None

    def set_data(self, gt_annotations, pred_annotations, categories, confidence_threshold, visible_class_ids):
        """GT, 예측, 카테고리, 임계값, 가시성 정보를 설정하고 다시 그립니다."""
        self.gt_annotations = gt_annotations
        self.pred_annotations = pred_annotations # 수정 가능한 예측 annotation 리스트
        self.categories = categories
        self.confidence_threshold = confidence_threshold
        self.visible_class_ids = visible_class_ids # 보여줄 클래스 ID 업데이트
        self.redraw_annotations()

    def redraw_annotations(self):
        """GT와 예측 annotations를 조건에 맞게 다시 그립니다."""
        self.delete("annotation") # 기존 annotation 관련 객체 삭제
        self._selected_pred_idx = -1 # 선택 상태 초기화

        handle_size = 6 # 크기 조절 핸들 크기
        min_bbox_size = 5 # 최소 bbox 크기

        # GT 그리기 (실선)
        for idx, ann in enumerate(self.gt_annotations):
            cat_id = ann['category_id']
            if cat_id not in self.visible_class_ids: # 가시성 체크
                continue

            bbox = ann['bbox']
            label = self.categories.get(cat_id, {}).get('name', f'ID:{cat_id}')
            color = get_color(cat_id, is_gt=True)

            xmin, ymin, w, h = bbox
            # bbox 크기 유효성 검사 (음수 또는 너무 작은 크기 방지)
            if w < min_bbox_size or h < min_bbox_size: continue
            xmax, ymax = xmin + w, ymin + h

            # GT 태그: annotation, gt, gt_idx_N
            gt_tag = f"gt_idx_{idx}"
            # 바운딩 박스 (실선)
            rect_id = self.create_rectangle(xmin, ymin, xmax, ymax, outline=color, width=2, tags=("annotation", "gt", gt_tag, "bbox"))
            # 레이블 텍스트
            text_content = f"GT: {label}"
            text_id = self.create_text(xmin + 2, ymin + 2, anchor="nw", text=text_content, fill=color, font=("Arial", 8), tags=("annotation", "gt", gt_tag, "label"))
            # GT는 수정 불가하므로 핸들 없음

        # 예측 그리기 (점선, 수정 가능)
        for idx, ann in enumerate(self.pred_annotations):
            cat_id = ann['category_id']
            score = ann['score']
            # Confidence 및 가시성 체크
            if score < self.confidence_threshold or cat_id not in self.visible_class_ids:
                continue

            bbox = ann['bbox']
            label = self.categories.get(cat_id, {}).get('name', f'ID:{cat_id}')
            color = get_color(cat_id, is_gt=False)

            xmin, ymin, w, h = bbox
            # bbox 크기 유효성 검사
            if w < min_bbox_size or h < min_bbox_size: continue
            xmax, ymax = xmin + w, ymin + h

            # 예측 태그: annotation, pred, pred_idx_N
            pred_tag = f"pred_idx_{idx}"
            # 바운딩 박스 (점선)
            rect_id = self.create_rectangle(xmin, ymin, xmax, ymax, outline=color, width=2, dash=(4, 2), tags=("annotation", "pred", pred_tag, "bbox"))
            # 레이블 텍스트
            text_content = f"{label} ({score:.2f})"
            text_id = self.create_text(xmin + 2, ymin - 10 if ymin > 10 else ymin + 2, anchor="nw", text=text_content, fill=color, font=("Arial", 8, "bold"), tags=("annotation", "pred", pred_tag, "label"))

            # 크기 조절 핸들 추가 (예측에만)
            handles = [
                (xmin, ymin, "resize_tl"), (xmax, ymin, "resize_tr"),
                (xmin, ymax, "resize_bl"), (xmax, ymax, "resize_br")
            ]
            for hx, hy, htype in handles:
                self.create_rectangle(hx - handle_size/2, hy - handle_size/2, hx + handle_size/2, hy + handle_size/2,
                                      fill=color, outline='black', tags=("annotation", "pred", pred_tag, htype, "handle"))

            # 내부용: 각 annotation에 연결된 canvas 객체 ID 저장 (업데이트 시 필요)
            ann['canvas_ids'] = self.find_withtag(pred_tag)


    def clear_annotations(self):
        self.delete("annotation")
        self.gt_annotations = []
        self.pred_annotations = []
        self._selected_pred_idx = -1

    def get_selected_pred_index(self):
        """현재 선택된 '예측' annotation의 인덱스를 반환합니다."""
        return self._selected_pred_idx

    def on_mouse_motion(self, event):
        """마우스 커서 모양을 변경합니다."""
        active_item = self.find_closest(event.x, event.y)[0]
        tags = self.gettags(active_item)

        if "handle" in tags:
            if "resize_tl" in tags or "resize_br" in tags:
                self.config(cursor="size_nw_se")
            elif "resize_tr" in tags or "resize_bl" in tags:
                self.config(cursor="size_ne_sw")
            else:
                self.config(cursor="fleur") # 이동 커서
        elif "bbox" in tags and "pred" in tags: # 예측 bbox 위
             self.config(cursor="fleur")
        else:
            self.config(cursor="") # 기본 커서


    def on_button_press(self, event):
        # 클릭된 위치의 가장 위에 있는 객체 찾기
        item = self.find_closest(event.x, event.y)[0]
        tags = self.gettags(item)

        self._selected_pred_idx = -1 # 클릭 시 일단 선택 해제

        if "annotation" in tags and "pred" in tags: # 예측 annotation 관련 객체 클릭
            self._drag_data["item"] = item
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y
            self._drag_data["is_gt"] = False

            # 어떤 종류의 객체인지 확인 (bbox, resize handle 등)
            drag_type = None
            if "bbox" in tags: drag_type = "move"
            elif "resize_tl" in tags: drag_type = "resize_tl"
            elif "resize_tr" in tags: drag_type = "resize_tr"
            elif "resize_bl" in tags: drag_type = "resize_bl"
            elif "resize_br" in tags: drag_type = "resize_br"
            # elif "label" in tags: drag_type = "move" # 레이블 클릭 시에도 이동 가능하게 하려면

            self._drag_data["type"] = drag_type

            # 선택된 annotation 인덱스 찾기
            for tag in tags:
                if tag.startswith("pred_idx_"):
                    try:
                        pred_idx = int(tag.split("_")[-1])
                        if 0 <= pred_idx < len(self.pred_annotations):
                            self._drag_data["ann_idx"] = pred_idx
                            self._selected_pred_idx = pred_idx # 예측 annotation 선택됨
                            self._drag_data["original_bbox"] = list(self.pred_annotations[pred_idx]['bbox']) # 원본 bbox 저장
                            print(f"Prediction {pred_idx} selected for {drag_type or 'selection'}")
                            # 선택 시각적 피드백 (예: 테두리 강조)
                            self.itemconfig(self.find_withtag(f"pred_idx_{pred_idx} and bbox")[0], width=4)
                        break
                    except (IndexError, ValueError):
                        pass
        else:
            # 배경 또는 GT 클릭 시 드래그 상태 초기화
            self._drag_data = {"x": 0, "y": 0, "item": None, "type": None, "ann_idx": -1, "is_gt": False, "original_bbox": None}
            # 모든 예측 bbox 테두리 원래대로 복원
            for item_id in self.find_withtag("pred and bbox"):
                self.itemconfig(item_id, width=2)
            print("Selection cleared")


    def on_move_press(self, event):
        """마우스 드래그 중 bbox 이동 또는 크기 조절"""
        if self._drag_data["item"] is None or self._drag_data["ann_idx"] == -1 or self._drag_data["is_gt"]:
            return

        pred_idx = self._drag_data["ann_idx"]
        drag_type = self._drag_data["type"]
        if not drag_type: return # 이동/크기조절 대상이 아니면 무시

        ann = self.pred_annotations[pred_idx]
        original_bbox = self._drag_data["original_bbox"]
        xmin, ymin, w, h = original_bbox # 항상 원본 기준으로 계산
        xmax, ymax = xmin + w, ymin + h

        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]

        new_bbox = list(original_bbox) # 새 bbox 계산용
        min_size = 10 # 최소 크기

        # 이동 로직
        if drag_type == "move":
            new_bbox[0] = xmin + delta_x
            new_bbox[1] = ymin + delta_y
        # 크기 조절 로직
        elif drag_type == "resize_br": # 오른쪽 아래
            new_bbox[2] = max(min_size, w + delta_x)
            new_bbox[3] = max(min_size, h + delta_y)
        elif drag_type == "resize_tl": # 왼쪽 위
            new_w = max(min_size, w - delta_x)
            new_h = max(min_size, h - delta_y)
            new_bbox[0] = xmax - new_w # xmin 변경
            new_bbox[1] = ymax - new_h # ymin 변경
            new_bbox[2] = new_w
            new_bbox[3] = new_h
        elif drag_type == "resize_tr": # 오른쪽 위
            new_bbox[2] = max(min_size, w + delta_x) # width 변경
            new_h = max(min_size, h - delta_y)
            new_bbox[1] = ymax - new_h # ymin 변경
            new_bbox[3] = new_h
        elif drag_type == "resize_bl": # 왼쪽 아래
            new_w = max(min_size, w - delta_x)
            new_bbox[0] = xmax - new_w # xmin 변경
            new_bbox[2] = new_w
            new_bbox[3] = max(min_size, h + delta_y) # height 변경

        # 임시: 현재 annotation 데이터 업데이트 (릴리스 시 최종 확정)
        # ann['bbox'] = new_bbox # 실시간 반영 시 주석 해제 (성능 저하 가능)

        # 캔버스 객체들 실시간 업데이트
        self.update_canvas_objects(pred_idx, new_bbox)


    def update_canvas_objects(self, pred_idx, bbox):
        """주어진 bbox에 따라 해당 예측 annotation의 캔버스 객체들을 업데이트합니다."""
        pred_tag = f"pred_idx_{pred_idx}"
        xmin, ymin, w, h = bbox
        xmax, ymax = xmin + w, ymin + h
        handle_size = 6

        # 연결된 모든 객체 찾기 (더 효율적인 방법: 생성 시 ID 저장)
        item_ids = self.find_withtag(pred_tag)
        if not item_ids: return

        for item_id in item_ids:
            tags = self.gettags(item_id)
            if "bbox" in tags:
                self.coords(item_id, xmin, ymin, xmax, ymax)
            elif "label" in tags:
                # 레이블 위치 조정 (예: 박스 위쪽 중앙 또는 왼쪽 위)
                self.coords(item_id, xmin + 2, ymin - 10 if ymin > 10 else ymin + 2)
            elif "handle" in tags:
                if "resize_tl" in tags: self.coords(item_id, xmin - handle_size/2, ymin - handle_size/2, xmin + handle_size/2, ymin + handle_size/2)
                elif "resize_tr" in tags: self.coords(item_id, xmax - handle_size/2, ymin - handle_size/2, xmax + handle_size/2, ymin + handle_size/2)
                elif "resize_bl" in tags: self.coords(item_id, xmin - handle_size/2, ymax - handle_size/2, xmin + handle_size/2, ymax + handle_size/2)
                elif "resize_br" in tags: self.coords(item_id, xmax - handle_size/2, ymax - handle_size/2, xmax + handle_size/2, ymax + handle_size/2)


    def on_button_release(self, event):
        """마우스 버튼 뗄 때 최종 bbox 업데이트 및 콜백 호출"""
        if self._drag_data["item"] is None or self._drag_data["ann_idx"] == -1 or self._drag_data["is_gt"]:
            self._drag_data = {"x": 0, "y": 0, "item": None, "type": None, "ann_idx": -1, "is_gt": False, "original_bbox": None}
            return # 드래그 상태 아니면 종료

        pred_idx = self._drag_data["ann_idx"]
        drag_type = self._drag_data["type"]

        # 최종 위치/크기 계산 (on_move_press와 동일한 로직으로 최종 bbox 계산)
        original_bbox = self._drag_data["original_bbox"]
        xmin, ymin, w, h = original_bbox
        xmax, ymax = xmin + w, ymin + h
        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]
        new_bbox = list(original_bbox)
        min_size = 10

        if drag_type == "move":
            new_bbox[0] = xmin + delta_x
            new_bbox[1] = ymin + delta_y
        elif drag_type == "resize_br":
            new_bbox[2] = max(min_size, w + delta_x)
            new_bbox[3] = max(min_size, h + delta_y)
        elif drag_type == "resize_tl":
            new_w = max(min_size, w - delta_x)
            new_h = max(min_size, h - delta_y)
            new_bbox[0] = xmax - new_w
            new_bbox[1] = ymax - new_h
            new_bbox[2] = new_w
            new_bbox[3] = new_h
        elif drag_type == "resize_tr":
            new_bbox[2] = max(min_size, w + delta_x)
            new_h = max(min_size, h - delta_y)
            new_bbox[1] = ymax - new_h
            new_bbox[3] = new_h
        elif drag_type == "resize_bl":
            new_w = max(min_size, w - delta_x)
            new_bbox[0] = xmax - new_w
            new_bbox[2] = new_w
            new_bbox[3] = max(min_size, h + delta_y)

        # 소수점 정리 및 annotation 데이터 업데이트
        final_bbox = [round(c, 2) for c in new_bbox]
        self.pred_annotations[pred_idx]['bbox'] = final_bbox
        print(f"Prediction {pred_idx} {drag_type} finished. New bbox: {final_bbox}")

        # 캔버스 객체 최종 위치 업데이트 (필수)
        self.update_canvas_objects(pred_idx, final_bbox)

        # 콜백 호출하여 GUI에 변경 알림
        if self.annotation_update_callback:
            # 콜백에는 수정된 annotation 전체를 전달
            self.annotation_update_callback(pred_idx, self.pred_annotations[pred_idx])

        # 드래그 상태 초기화
        self._drag_data = {"x": 0, "y": 0, "item": None, "type": None, "ann_idx": -1, "is_gt": False, "original_bbox": None}
        # 선택 상태는 유지 (edit label 등 후속 작업 가능)
        # 선택 피드백 유지 (테두리 두께 등)


# 예시 사용법 (테스트용)
if __name__ == '__main__':
    import os
    root = tk.Tk()
    root.title("Interactive Canvas Test")
    root.geometry("800x600")

    canvas = InteractiveCanvas(root, bg="white")
    canvas.pack(fill="both", expand=True)

    # 테스트용 이미지 로드 (실제 경로 사용)
    script_dir = os.path.dirname(__file__) # 현재 스크립트 디렉토리
    test_image_path = os.path.join(script_dir, 'val2017/000000000139.jpg') # 상대 경로 사용 시도
    # 절대 경로 예시: test_image_path = 'c:/path/to/your/images/val2017/000000000139.jpg'

    if os.path.exists(test_image_path):
        canvas.load_image(test_image_path)
    else:
        print(f"Test image not found: {test_image_path}")
        # 대체 이미지 또는 오류 메시지
        canvas.create_text(400, 300, text="Test Image Not Found", anchor="center")


    # 테스트용 annotation 데이터 설정
    test_gt = [
        {'image_id': 139, 'category_id': 1, 'bbox': [50.0, 50.0, 100.0, 80.0], 'id': 101},
    ]
    test_preds = [
        {'image_id': 139, 'category_id': 1, 'bbox': [100.0, 100.0, 150.0, 150.0], 'score': 0.95, 'id': 1},
        {'image_id': 139, 'category_id': 2, 'bbox': [300.0, 200.0, 100.0, 80.0], 'score': 0.88, 'id': 2},
        {'image_id': 139, 'category_id': 1, 'bbox': [400.0, 50.0, 70.0, 120.0], 'score': 0.70, 'id': 3},
    ]
    test_categories = {
        1: {'id': 1, 'name': 'ObjectA'},
        2: {'id': 2, 'name': 'ObjectB'},
    }
    visible_classes = {1, 2} # 모든 클래스 보이게 설정

    def on_update(index, annotation):
        print(f"Callback: Prediction {index} updated: {annotation}")

    canvas.set_annotation_update_callback(on_update)
    canvas.set_data(test_gt, test_preds, test_categories, 0.5, visible_classes)

    root.mainloop()