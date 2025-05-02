# interactive_canvas.py
import tkinter as tk
from PIL import Image, ImageTk
import math

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
        self.original_pil_image = None # 원본 이미지 저장
        self.display_pil_image = None # 화면 표시용 리사이즈된 이미지
        self.tk_image = None
        self.image_on_canvas = None # Canvas상의 이미지 객체 ID

        self.gt_annotations = []
        self.pred_annotations = []
        self.categories = {}
        self.confidence_threshold = 0.5
        self.visible_class_ids = set()

        # 좌표 변환용 변수
        self.display_scale = 1.0
        self.display_offset = (0, 0) # (x_offset, y_offset)

        # 드래그 상태 변수
        self._drag_data = {"x": 0, "y": 0, "item": None, "type": None, "ann_idx": -1, "is_gt": False, "original_bbox": None, "start_canvas_coords": (0,0)}
        self._selected_pred_idx = -1

        # 이벤트 바인딩
        self.bind("<ButtonPress-1>", self.on_button_press)
        self.bind("<B1-Motion>", self.on_move_press)
        self.bind("<ButtonRelease-1>", self.on_button_release)
        self.bind("<Motion>", self.on_mouse_motion)
        self.bind("<Configure>", self.on_resize) # 캔버스 크기 변경 감지

        # 콜백 함수
        self.annotation_update_callback = None

    def set_annotation_update_callback(self, callback):
        self.annotation_update_callback = callback

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """캔버스 좌표를 원본 이미지 좌표로 변환"""
        if self.display_scale == 0: return 0, 0
        img_x = (canvas_x - self.display_offset[0]) / self.display_scale
        img_y = (canvas_y - self.display_offset[1]) / self.display_scale
        return img_x, img_y

    def _image_to_canvas_coords(self, img_x, img_y):
        """원본 이미지 좌표를 캔버스 좌표로 변환"""
        canvas_x = img_x * self.display_scale + self.display_offset[0]
        canvas_y = img_y * self.display_scale + self.display_offset[1]
        return canvas_x, canvas_y

    def load_image(self, image_path):
        try:
            self.original_pil_image = Image.open(image_path).convert("RGB")
            self.tk_image = None # 이전 tk 이미지 해제
            if self.image_on_canvas:
                self.delete(self.image_on_canvas)
                self.image_on_canvas = None

            # 이미지가 로드되면 기존 annotation 삭제
            self.clear_annotations()
            # 이미지 리사이즈 및 표시 (Configure 이벤트에서도 호출됨)
            self._resize_and_display_image()

        except Exception as e:
            print(f"Error: Image load failed - {e}")
            self.delete("all")
            self.original_pil_image = None
            self.tk_image = None
            self.image_on_canvas = None

    def _resize_and_display_image(self):
        """캔버스 크기에 맞춰 이미지를 리사이즈하고 표시"""
        if not self.original_pil_image:
            return

        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1: # 초기화 전 크기 방지
            return

        img_w, img_h = self.original_pil_image.size
        if img_w == 0 or img_h == 0: return

        # 비율 유지하며 스케일 계산
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        self.display_scale = min(scale_w, scale_h)

        # 새 크기 및 중앙 정렬 오프셋 계산
        new_w = int(img_w * self.display_scale)
        new_h = int(img_h * self.display_scale)
        self.display_offset = ((canvas_w - new_w) // 2, (canvas_h - new_h) // 2)

        # 이미지 리사이즈 및 Tk 객체 생성
        self.display_pil_image = self.original_pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_pil_image)

        # 캔버스에 이미지 표시/업데이트
        if self.image_on_canvas:
            self.coords(self.image_on_canvas, self.display_offset[0], self.display_offset[1])
            self.itemconfig(self.image_on_canvas, image=self.tk_image)
        else:
            self.image_on_canvas = self.create_image(self.display_offset[0], self.display_offset[1], anchor="nw", image=self.tk_image)

        # 스크롤 영역 설정 (이미지 크기에 맞게)
        self.config(scrollregion=(self.display_offset[0], self.display_offset[1],
                                   self.display_offset[0] + new_w, self.display_offset[1] + new_h))

        # 이미지 크기 변경 시 annotation 다시 그리기
        self.redraw_annotations()

    def on_resize(self, event):
        """캔버스 크기가 변경될 때 호출"""
        # 너비 또는 높이가 변경되었을 때만 리사이즈 수행 (불필요한 호출 방지)
        if hasattr(self, '_last_width') and hasattr(self, '_last_height'):
             if self._last_width == event.width and self._last_height == event.height:
                 return
        self._last_width = event.width
        self._last_height = event.height

        self._resize_and_display_image()

    def set_data(self, gt_annotations, pred_annotations, categories, confidence_threshold, visible_class_ids):
        """데이터 설정 및 다시 그리기"""
        self.gt_annotations = gt_annotations
        self.pred_annotations = pred_annotations
        self.categories = categories
        self.confidence_threshold = confidence_threshold
        self.visible_class_ids = visible_class_ids
        self.redraw_annotations()

    def redraw_annotations(self):
        """Annotations를 현재 스케일과 오프셋에 맞게 다시 그립니다."""
        self.delete("annotation")
        self._selected_pred_idx = -1
        if not self.original_pil_image: return # 이미지가 없으면 그리지 않음

        handle_size = 6 # 캔버스 픽셀 기준
        min_bbox_size = 5 # 원본 이미지 픽셀 기준

        # GT 그리기
        for idx, ann in enumerate(self.gt_annotations):
            cat_id = ann['category_id']
            if cat_id not in self.visible_class_ids: continue

            bbox = ann['bbox'] # 원본 이미지 좌표계 [xmin, ymin, w, h]
            label = self.categories.get(cat_id, {}).get('name', f'ID:{cat_id}')
            color = get_color(cat_id, is_gt=True)

            xmin_img, ymin_img, w_img, h_img = bbox
            if w_img < min_bbox_size or h_img < min_bbox_size: continue
            xmax_img, ymax_img = xmin_img + w_img, ymin_img + h_img

            # 캔버스 좌표로 변환
            xmin_c, ymin_c = self._image_to_canvas_coords(xmin_img, ymin_img)
            xmax_c, ymax_c = self._image_to_canvas_coords(xmax_img, ymax_img)

            gt_tag = f"gt_idx_{idx}"
            rect_id = self.create_rectangle(xmin_c, ymin_c, xmax_c, ymax_c, outline=color, width=2, tags=("annotation", "gt", gt_tag, "bbox"))
            text_content = f"GT: {label}"
            # 텍스트 위치도 변환된 좌표 사용
            self.create_text(xmin_c + 2, ymin_c + 2, anchor="nw", text=text_content, fill=color, font=("Arial", 8), tags=("annotation", "gt", gt_tag, "label"))

        # 예측 그리기
        for idx, ann in enumerate(self.pred_annotations):
            cat_id = ann['category_id']
            score = ann['score']
            if score < self.confidence_threshold or cat_id not in self.visible_class_ids: continue

            bbox = ann['bbox'] # 원본 이미지 좌표계
            label = self.categories.get(cat_id, {}).get('name', f'ID:{cat_id}')
            color = get_color(cat_id, is_gt=False)

            xmin_img, ymin_img, w_img, h_img = bbox
            if w_img < min_bbox_size or h_img < min_bbox_size: continue
            xmax_img, ymax_img = xmin_img + w_img, ymin_img + h_img

            # 캔버스 좌표로 변환
            xmin_c, ymin_c = self._image_to_canvas_coords(xmin_img, ymin_img)
            xmax_c, ymax_c = self._image_to_canvas_coords(xmax_img, ymax_img)

            pred_tag = f"pred_idx_{idx}"
            rect_id = self.create_rectangle(xmin_c, ymin_c, xmax_c, ymax_c, outline=color, width=2, dash=(4, 2), tags=("annotation", "pred", pred_tag, "bbox"))
            text_content = f"{label} ({score:.2f})"
            # 텍스트 위치 계산 (캔버스 좌표 기준)
            text_y = ymin_c - 10 if ymin_c > (self.display_offset[1] + 10) else ymin_c + 2
            self.create_text(xmin_c + 2, text_y, anchor="nw", text=text_content, fill=color, font=("Arial", 8, "bold"), tags=("annotation", "pred", pred_tag, "label"))

            # 핸들 위치 (캔버스 좌표 기준)
            handles = [
                (xmin_c, ymin_c, "resize_tl"), (xmax_c, ymin_c, "resize_tr"),
                (xmin_c, ymax_c, "resize_bl"), (xmax_c, ymax_c, "resize_br")
            ]
            for hx, hy, htype in handles:
                self.create_rectangle(hx - handle_size/2, hy - handle_size/2, hx + handle_size/2, hy + handle_size/2,
                                      fill=color, outline='black', tags=("annotation", "pred", pred_tag, htype, "handle"))

    def clear_annotations(self):
        self.delete("annotation")
        self._selected_pred_idx = -1

    def get_selected_pred_index(self):
        return self._selected_pred_idx

    def on_mouse_motion(self, event):
        """마우스 커서 모양 변경 (캔버스 좌표 기준)"""
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
        """클릭 시 객체 선택 및 드래그 시작 (캔버스 좌표 사용, 원본 bbox 저장)"""
        item = self.find_closest(event.x, event.y)[0]
        tags = self.gettags(item)

        self._selected_pred_idx = -1

        if "annotation" in tags and "pred" in tags:
            self._drag_data["item"] = item
            self._drag_data["x"] = event.x # 캔버스 클릭 x
            self._drag_data["y"] = event.y # 캔버스 클릭 y
            self._drag_data["start_canvas_coords"] = (event.x, event.y) # 드래그 시작 캔버스 좌표
            self._drag_data["is_gt"] = False

            drag_type = None
            if "bbox" in tags: drag_type = "move"
            elif "resize_tl" in tags: drag_type = "resize_tl"
            elif "resize_tr" in tags: drag_type = "resize_tr"
            elif "resize_bl" in tags: drag_type = "resize_bl"
            elif "resize_br" in tags: drag_type = "resize_br"

            self._drag_data["type"] = drag_type

            for tag in tags:
                if tag.startswith("pred_idx_"):
                    try:
                        pred_idx = int(tag.split("_")[-1])
                        if 0 <= pred_idx < len(self.pred_annotations):
                            self._drag_data["ann_idx"] = pred_idx
                            self._selected_pred_idx = pred_idx
                            self._drag_data["original_bbox"] = list(self.pred_annotations[pred_idx]['bbox'])
                            print(f"Prediction {pred_idx} selected for {drag_type or 'selection'}")
                            bbox_items = self.find_withtag(f"pred_idx_{pred_idx} and bbox")
                            if bbox_items:
                                self.itemconfig(bbox_items[0], width=4)
                        break
                    except (IndexError, ValueError):
                        pass
        else:
            self._drag_data = {"x": 0, "y": 0, "item": None, "type": None, "ann_idx": -1, "is_gt": False, "original_bbox": None, "start_canvas_coords": (0,0)}
            for item_id in self.find_withtag("pred and bbox"):
                self.itemconfig(item_id, width=2)
            print("Selection cleared")

    def on_move_press(self, event):
        """드래그 중 bbox 이동/크기 조절 (원본 이미지 좌표계에서 계산 후 캔버스 업데이트)"""
        if self._drag_data["item"] is None or self._drag_data["ann_idx"] == -1 or self._drag_data["is_gt"]:
            return

        pred_idx = self._drag_data["ann_idx"]
        drag_type = self._drag_data["type"]
        if not drag_type: return

        original_bbox = self._drag_data["original_bbox"]
        xmin_img, ymin_img, w_img, h_img = original_bbox
        xmax_img, ymax_img = xmin_img + w_img, ymin_img + h_img

        canvas_dx = event.x - self._drag_data["start_canvas_coords"][0]
        canvas_dy = event.y - self._drag_data["start_canvas_coords"][1]

        img_dx = canvas_dx / self.display_scale
        img_dy = canvas_dy / self.display_scale

        new_bbox_img = list(original_bbox)
        min_size_img = 10

        if drag_type == "move":
            new_bbox_img[0] = xmin_img + img_dx
            new_bbox_img[1] = ymin_img + img_dy
        elif drag_type == "resize_br":
            new_bbox_img[2] = max(min_size_img, w_img + img_dx)
            new_bbox_img[3] = max(min_size_img, h_img + img_dy)
        elif drag_type == "resize_tl":
            new_w = max(min_size_img, w_img - img_dx)
            new_h = max(min_size_img, h_img - img_dy)
            new_bbox_img[0] = xmax_img - new_w
            new_bbox_img[1] = ymax_img - new_h
            new_bbox_img[2] = new_w
            new_bbox_img[3] = new_h
        elif drag_type == "resize_tr":
            new_bbox_img[2] = max(min_size_img, w_img + img_dx)
            new_h = max(min_size_img, h_img - img_dy)
            new_bbox_img[1] = ymax_img - new_h
            new_bbox_img[3] = new_h
        elif drag_type == "resize_bl":
            new_w = max(min_size_img, w_img - img_dx)
            new_bbox_img[0] = xmax_img - new_w
            new_bbox_img[2] = new_w
            new_bbox_img[3] = max(min_size_img, h_img + img_dy)

        self.update_canvas_objects(pred_idx, new_bbox_img)

    def update_canvas_objects(self, pred_idx, bbox_img):
        """주어진 원본 이미지 bbox에 따라 캔버스 객체들을 업데이트합니다."""
        pred_tag = f"pred_idx_{pred_idx}"
        xmin_img, ymin_img, w_img, h_img = bbox_img
        xmax_img, ymax_img = xmin_img + w_img, ymin_img + h_img
        handle_size = 6

        xmin_c, ymin_c = self._image_to_canvas_coords(xmin_img, ymin_img)
        xmax_c, ymax_c = self._image_to_canvas_coords(xmax_img, ymax_img)

        item_ids = self.find_withtag(pred_tag)
        if not item_ids: return

        for item_id in item_ids:
            tags = self.gettags(item_id)
            if "bbox" in tags:
                self.coords(item_id, xmin_c, ymin_c, xmax_c, ymax_c)
            elif "label" in tags:
                text_y = ymin_c - 10 if ymin_c > (self.display_offset[1] + 10) else ymin_c + 2
                self.coords(item_id, xmin_c + 2, text_y)
            elif "handle" in tags:
                if "resize_tl" in tags: self.coords(item_id, xmin_c - handle_size/2, ymin_c - handle_size/2, xmin_c + handle_size/2, ymin_c + handle_size/2)
                elif "resize_tr" in tags: self.coords(item_id, xmax_c - handle_size/2, ymin_c - handle_size/2, xmax_c + handle_size/2, ymin_c + handle_size/2)
                elif "resize_bl" in tags: self.coords(item_id, xmin_c - handle_size/2, ymax_c - handle_size/2, xmin_c + handle_size/2, ymax_c + handle_size/2)
                elif "resize_br" in tags: self.coords(item_id, xmax_c - handle_size/2, ymax_c - handle_size/2, xmax_c + handle_size/2, ymax_c + handle_size/2)

    def on_button_release(self, event):
        """마우스 뗄 때 최종 bbox 업데이트 (원본 이미지 좌표계) 및 콜백 호출"""
        if self._drag_data["item"] is None or self._drag_data["ann_idx"] == -1 or self._drag_data["is_gt"]:
            self._drag_data = {"x": 0, "y": 0, "item": None, "type": None, "ann_idx": -1, "is_gt": False, "original_bbox": None, "start_canvas_coords": (0,0)}
            return

        pred_idx = self._drag_data["ann_idx"]
        drag_type = self._drag_data["type"]

        original_bbox = self._drag_data["original_bbox"]
        xmin_img, ymin_img, w_img, h_img = original_bbox
        xmax_img, ymax_img = xmin_img + w_img, ymin_img + h_img

        canvas_dx = event.x - self._drag_data["start_canvas_coords"][0]
        canvas_dy = event.y - self._drag_data["start_canvas_coords"][1]
        img_dx = canvas_dx / self.display_scale
        img_dy = canvas_dy / self.display_scale

        new_bbox_img = list(original_bbox)
        min_size_img = 10

        if drag_type == "move":
            new_bbox_img[0] = xmin_img + img_dx
            new_bbox_img[1] = ymin_img + img_dy
        elif drag_type == "resize_br":
            new_bbox_img[2] = max(min_size_img, w_img + img_dx)
            new_bbox_img[3] = max(min_size_img, h_img + img_dy)
        elif drag_type == "resize_tl":
            new_w = max(min_size_img, w_img - img_dx)
            new_h = max(min_size_img, h_img - img_dy)
            new_bbox_img[0] = xmax_img - new_w
            new_bbox_img[1] = ymax_img - new_h
            new_bbox_img[2] = new_w
            new_bbox_img[3] = new_h
        elif drag_type == "resize_tr":
            new_bbox_img[2] = max(min_size_img, w_img + img_dx)
            new_h = max(min_size_img, h_img - img_dy)
            new_bbox_img[1] = ymax_img - new_h
            new_bbox_img[3] = new_h
        elif drag_type == "resize_bl":
            new_w = max(min_size_img, w_img - img_dx)
            new_bbox_img[0] = xmax_img - new_w
            new_bbox_img[2] = new_w
            new_bbox_img[3] = max(min_size_img, h_img + img_dy)

        final_bbox_img = [round(c, 2) for c in new_bbox_img]
        self.pred_annotations[pred_idx]['bbox'] = final_bbox_img
        print(f"Prediction {pred_idx} {drag_type} finished. New image bbox: {final_bbox_img}")

        self.update_canvas_objects(pred_idx, final_bbox_img)

        if self.annotation_update_callback:
            self.annotation_update_callback(pred_idx, self.pred_annotations[pred_idx])

        self._drag_data = {"x": 0, "y": 0, "item": None, "type": None, "ann_idx": -1, "is_gt": False, "original_bbox": None, "start_canvas_coords": (0,0)}

if __name__ == '__main__':
    import os
    root = tk.Tk()
    root.title("Interactive Canvas Test")
    root.geometry("800x600")

    canvas = InteractiveCanvas(root, bg="white")
    canvas.pack(fill="both", expand=True)

    script_dir = os.path.dirname(__file__)
    test_image_path = os.path.join(script_dir, 'val2017/000000000139.jpg')

    if os.path.exists(test_image_path):
        canvas.load_image(test_image_path)
    else:
        print(f"Test image not found: {test_image_path}")
        canvas.create_text(400, 300, text="Test Image Not Found", anchor="center")

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
    visible_classes = {1, 2}

    def on_update(index, annotation):
        print(f"Callback: Prediction {index} updated: {annotation}")

    canvas.set_annotation_update_callback(on_update)
    canvas.set_data(test_gt, test_preds, test_categories, 0.5, visible_classes)

    root.mainloop()