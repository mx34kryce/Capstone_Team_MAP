# interactive_canvas.py
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw # ImageDraw 추가
import math
import os

# visualizer 모듈의 색상 함수 재사용 또는 여기서 정의
DEFAULT_COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
    '#C00000', '#00C000', '#0000C0', '#C0C000', '#C000C0', '#00C0C0',
    '#400000', '#004000', '#000040', '#404000', '#400040', '#004040',
    '#FFA500', '#FFD700', '#ADFF2F', '#7FFF00', '#00FA9A', '#00CED1',
    '#1E90FF', '#DA70D6', '#FF69B4', '#FF1493', '#DC143C', '#A52A2A',
]
GT_COLOR_OFFSET = 0 # GT 색상을 예측과 동일하게 설정

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

        # 좌표 변환 및 표시용 변수
        self.display_scale = 1.0
        self.display_offset = (0, 0) # (x_offset, y_offset) 캔버스 좌상단 기준 이미지의 좌상단 위치

        # 드래그 상태 변수 (bbox 수정용)
        self._drag_data = {"x": 0, "y": 0, "item": None, "type": None, "ann_idx": -1, "is_gt": False, "original_bbox": None, "start_canvas_coords": (0,0)}
        self._selected_pred_idx = -1

        # 패닝(이동) 상태 변수
        self._pan_data = {"x": 0, "y": 0, "active": False}

        # 이벤트 바인딩
        self.bind("<ButtonPress-1>", self.on_button_press)      # Annotation 선택/드래그 시작
        self.bind("<B1-Motion>", self.on_move_press)            # Annotation 드래그
        self.bind("<ButtonRelease-1>", self.on_button_release)  # Annotation 드래그 종료
        self.bind("<Motion>", self.on_mouse_motion)             # 커서 변경
        self.bind("<Configure>", self.on_resize)                # 캔버스 크기 변경

        # 휠 확대/축소 (Windows, macOS)
        self.bind("<MouseWheel>", self.on_mouse_wheel)
        # 휠 확대/축소 (Linux)
        self.bind("<Button-4>", lambda e: self.on_mouse_wheel(e, custom_delta=120))
        self.bind("<Button-5>", lambda e: self.on_mouse_wheel(e, custom_delta=-120))

        # 이미지 패닝 (휠 클릭 드래그 - 보통 마우스 가운데 버튼)
        self.bind("<ButtonPress-2>", self.on_pan_start)    # 휠 버튼 누름 (패닝 시작)
        self.bind("<B2-Motion>", self.on_pan_drag)         # 휠 버튼 누른 채 드래그 (패닝 중)
        self.bind("<ButtonRelease-2>", self.on_pan_end)    # 휠 버튼 뗌 (패닝 종료)

        self.annotation_update_callback = None
        
        # 줌 관련 설정
        self.min_zoom = 0.1 # 최소 줌 레벨
        self.max_zoom = 5.0 # 최대 줌 레벨
        self.zoom_factor_base = 1.05 # 줌 스텝 (휠 한 칸당 변경 비율)


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

            self.clear_annotations()
            # 이미지 로드 시 스케일과 오프셋 초기화
            self.display_scale = 1.0
            self.display_offset = (0,0)
            self._fit_image_to_canvas() # 초기 핏팅

        except Exception as e:
            print(f"Error: Image load failed - {e}")
            self.delete("all")
            self.original_pil_image = None
            self.tk_image = None
            self.image_on_canvas = None
            self.display_scale = 1.0
            self.display_offset = (0,0)

    def _fit_image_to_canvas(self):
        """캔버스 크기에 맞춰 이미지 스케일과 오프셋을 계산하고 화면을 업데이트합니다."""
        if not self.original_pil_image:
            self._update_display() # 이미지가 없으면 화면 클리어
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
        # 이미지를 캔버스에 맞추되, max_zoom을 넘지 않도록 함
        self.display_scale = min(scale_w, scale_h, self.max_zoom) 
        self.display_scale = max(self.display_scale, self.min_zoom) # min_zoom 제한

        # 새 크기 및 중앙 정렬 오프셋 계산
        scaled_img_w = int(img_w * self.display_scale)
        scaled_img_h = int(img_h * self.display_scale)
        self.display_offset = ((canvas_w - scaled_img_w) // 2, (canvas_h - scaled_img_h) // 2)

        self._update_display()

    def _update_display(self):
        """현재 display_scale과 display_offset을 사용하여 이미지를 그리고 annotation을 업데이트합니다."""
        if not self.original_pil_image:
            if self.image_on_canvas:
                self.delete(self.image_on_canvas)
                self.image_on_canvas = None
            self.tk_image = None
            self.delete("annotation") # Annotation도 삭제
            return

        img_w, img_h = self.original_pil_image.size

        # 현재 스케일로 표시될 이미지 크기
        scaled_w = int(img_w * self.display_scale)
        scaled_h = int(img_h * self.display_scale)

        if scaled_w <= 0 or scaled_h <= 0: # 너무 작아지면 그리지 않음
            if self.image_on_canvas: self.delete(self.image_on_canvas)
            self.image_on_canvas = None
            self.tk_image = None
            self.delete("annotation")
            return
        
        try:
            # 원본 이미지에서 현재 스케일에 맞게 리사이즈
            self.display_pil_image = self.original_pil_image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(self.display_pil_image)
        except Exception as e:
            # print(f"Debug: Error resizing image for display: {e}, scaled_w={scaled_w}, scaled_h={scaled_h}")
            # 오류 발생 시, 이전 tk_image를 유지하거나, 이미지를 지울 수 있습니다.
            # 여기서는 이미지를 지우는 대신, 오류 메시지를 출력하고 넘어갑니다.
            # self.tk_image = None # 주석 처리하여 이전 이미지 유지 시도
            if self.image_on_canvas: self.delete(self.image_on_canvas)
            self.image_on_canvas = None
            self.delete("annotation")
            return


        # 캔버스에 이미지 표시/업데이트
        # display_offset은 캔버스 좌상단에서 이미지가 시작될 위치
        if self.image_on_canvas:
            self.coords(self.image_on_canvas, self.display_offset[0], self.display_offset[1])
            self.itemconfig(self.image_on_canvas, image=self.tk_image)
        else:
            self.image_on_canvas = self.create_image(self.display_offset[0], self.display_offset[1],
                                                     anchor="nw", image=self.tk_image)
        
        # 이미지를 다른 요소들보다 아래에 배치 (Annotation이 위에 오도록)
        if self.image_on_canvas:
            self.tag_lower(self.image_on_canvas)

        self.redraw_annotations()


    def on_resize(self, event):
        """캔버스 크기가 변경될 때 호출"""
        if hasattr(self, '_last_width') and hasattr(self, '_last_height'):
             if self._last_width == event.width and self._last_height == event.height:
                 return
        self._last_width = event.width
        self._last_height = event.height
        
        self._fit_image_to_canvas()

    def on_mouse_wheel(self, event, custom_delta=None):
        if not self.original_pil_image: return

        # 스크롤 방향 결정
        if custom_delta is not None: # Linux Button-4/5 용
            delta = custom_delta
        elif os.name == 'nt': # Windows
            delta = event.delta
        elif hasattr(event, 'delta'): # macOS 등 event.delta가 있는 경우
             delta = event.delta
        else: # 그 외 (delta 속성이 없는 경우, 예: 일부 Linux 환경의 일반 휠)
            # 이 경우는 Button-4/5 바인딩으로 처리되거나, delta를 추론해야 함
            # 여기서는 custom_delta가 제공되지 않으면 무시
            return


        # 마우스 커서의 캔버스 좌표
        canvas_x, canvas_y = event.x, event.y

        # 확대/축소 전, 마우스 커서 위치에 해당하는 원본 이미지 좌표
        img_x_at_cursor, img_y_at_cursor = self._canvas_to_image_coords(canvas_x, canvas_y)

        # 새 스케일 계산
        if delta > 0: # Zoom in
            new_scale = self.display_scale * self.zoom_factor_base
        else: # Zoom out
            new_scale = self.display_scale / self.zoom_factor_base

        new_scale = max(self.min_zoom, min(self.max_zoom, new_scale))

        if abs(new_scale - self.display_scale) < 1e-6 : # 스케일 변화가 거의 없으면 무시
            return

        self.display_scale = new_scale

        # 새 스케일에서, 이전의 이미지 좌표 (img_x_at_cursor, img_y_at_cursor)가
        # 여전히 마우스 커서 (canvas_x, canvas_y) 아래에 오도록 display_offset 조정
        self.display_offset = (
            canvas_x - img_x_at_cursor * self.display_scale,
            canvas_y - img_y_at_cursor * self.display_scale
        )

        self._update_display()

    def on_pan_start(self, event):
        if not self.original_pil_image: return
        # Annotation 드래그 중이 아닐 때만 패닝 활성화
        if self._drag_data["item"] is None:
            self.config(cursor="fleur")
            self._pan_data["x"] = event.x
            self._pan_data["y"] = event.y
            self._pan_data["active"] = True

    def on_pan_drag(self, event):
        if not self._pan_data["active"] or not self.original_pil_image: return

        dx = event.x - self._pan_data["x"]
        dy = event.y - self._pan_data["y"]

        self.display_offset = (self.display_offset[0] + dx, self.display_offset[1] + dy)

        self._pan_data["x"] = event.x
        self._pan_data["y"] = event.y

        self._update_display()

    def on_pan_end(self, event):
        if self._pan_data["active"]:
            self.config(cursor="") # 기본 커서로 복원 (또는 on_mouse_motion이 처리하도록)
            self._pan_data["active"] = False
            self.on_mouse_motion(event) # 패닝 종료 후 커서 즉시 업데이트


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
            color = get_color(cat_id, is_gt=True) # is_gt는 이제 색상 자체를 바꾸지 않음

            xmin_img, ymin_img, w_img, h_img = bbox
            if w_img < min_bbox_size or h_img < min_bbox_size: continue
            xmax_img, ymax_img = xmin_img + w_img, ymin_img + h_img

            # 캔버스 좌표로 변환
            xmin_c, ymin_c = self._image_to_canvas_coords(xmin_img, ymin_img)
            xmax_c, ymax_c = self._image_to_canvas_coords(xmax_img, ymax_img)

            gt_tag = f"gt_idx_{idx}"
            # GT 박스 내부를 stipple로 채워 투명도 효과 부여
            rect_id = self.create_rectangle(xmin_c, ymin_c, xmax_c, ymax_c, 
                                            outline=color, width=2, fill=color, stipple="gray50", 
                                            tags=("annotation", "gt", gt_tag, "bbox"))
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
            # 예측 박스는 내부를 채우지 않음 (기존과 동일)
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
        if self._pan_data["active"]: # 패닝 중이면 커서 변경 안 함 (fleur 유지)
            return

        active_item = self.find_closest(event.x, event.y)[0] if self.find_closest(event.x, event.y) else None
        tags = self.gettags(active_item) if active_item else []


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
        # 패닝 중이면 bbox 선택/드래그 비활성화
        if self._pan_data["active"]:
            return

        item = self.find_closest(event.x, event.y)[0] if self.find_closest(event.x, event.y) else None
        tags = self.gettags(item) if item else []


        self._selected_pred_idx = -1
        # 이전 선택된 bbox 테두리 원래대로
        for item_id in self.find_withtag("pred and bbox"):
            self.itemconfig(item_id, width=2)


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
                            # print(f"Prediction {pred_idx} selected for {drag_type or 'selection'}")
                            bbox_items = self.find_withtag(f"pred_idx_{pred_idx} and bbox")
                            if bbox_items:
                                self.itemconfig(bbox_items[0], width=4) # 선택된 bbox 강조
                        break
                    except (IndexError, ValueError):
                        pass
        else:
            self._drag_data = {"x": 0, "y": 0, "item": None, "type": None, "ann_idx": -1, "is_gt": False, "original_bbox": None, "start_canvas_coords": (0,0)}
            # print("Selection cleared")

    def on_move_press(self, event):
        """드래그 중 bbox 이동/크기 조절 (원본 이미지 좌표계에서 계산 후 캔버스 업데이트)"""
        # 패닝 중이면 bbox 이동/크기 조절 비활성화
        if self._pan_data["active"]:
            return

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
        min_size_img = 10 # 원본 이미지 기준 최소 크기

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
        # 패닝으로 인해 버튼이 해제된 경우는 _pan_data["active"]가 true일 수 있음
        # 여기서는 bbox 드래그에 대한 처리이므로, _drag_data["item"] 기준으로 판단
        if self._drag_data["item"] is None or self._drag_data["ann_idx"] == -1 or self._drag_data["is_gt"]:
            # 만약 패닝 중이었다면, 패닝 종료는 on_pan_end에서 처리됨
            # 여기서는 bbox 드래그가 아니었음을 확인하고 초기화
            if not self._pan_data["active"]: # 패닝으로 인한 release가 아닐 때만 초기화
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
        # print(f"Prediction {pred_idx} {drag_type} finished. New image bbox: {final_bbox_img}")

        self.update_canvas_objects(pred_idx, final_bbox_img)

        if self.annotation_update_callback:
            self.annotation_update_callback(pred_idx, self.pred_annotations[pred_idx])

        self._drag_data = {"x": 0, "y": 0, "item": None, "type": None, "ann_idx": -1, "is_gt": False, "original_bbox": None, "start_canvas_coords": (0,0)}

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Interactive Canvas Test")
    root.geometry("800x600")

    canvas = InteractiveCanvas(root, bg="lightgrey") # 배경색 변경하여 이미지와 구분
    canvas.pack(fill="both", expand=True)

    # 테스트를 위해 현재 스크립트 파일이 있는 디렉토리를 기준으로 이미지 경로 설정
    # 실제 사용 시에는 GUI를 통해 이미지 경로를 받아오거나 설정해야 함
    try:
        script_dir = os.path.dirname(__file__)
        # 예시 이미지 경로 (COCO val2017 데이터셋의 이미지 중 하나)
        # 이 이미지가 실제로 해당 경로에 있어야 테스트 가능
        test_image_path = os.path.join(script_dir, '..', 'val2017', '000000000139.jpg') # 경로 수정
        # test_image_path = "path/to/your/test/image.jpg" # 실제 이미지 경로로 변경
        
        # 임시 테스트용 빈 이미지 생성 (위 경로에 이미지가 없을 경우)
        if not os.path.exists(test_image_path):
            print(f"Test image not found at: {test_image_path}")
            print("Creating a dummy image for testing.")
            dummy_img_path = "dummy_test_image.png"
            try:
                img = Image.new('RGB', (640, 480), color = 'skyblue')
                draw = ImageDraw.Draw(img)
                draw.text((10,10), "Dummy Test Image", fill=(0,0,0))
                img.save(dummy_img_path)
                test_image_path = dummy_img_path
            except Exception as e_dummy:
                print(f"Failed to create dummy image: {e_dummy}")
                test_image_path = None


        if test_image_path and os.path.exists(test_image_path):
            canvas.load_image(test_image_path)
        else:
            canvas.create_text(400, 300, text="Test Image Not Found.\nPlease check the path.", anchor="center", font=("Arial", 12))

    except NameError: # __file__이 정의되지 않은 환경 (예: 일부 인터프리터)
        print("Cannot determine script directory. Please set test_image_path manually.")
        canvas.create_text(400, 300, text="Image path error.", anchor="center")


    test_gt = [
        {'image_id': 139, 'category_id': 1, 'bbox': [50.0, 50.0, 100.0, 80.0], 'id': 101},
        {'image_id': 139, 'category_id': 2, 'bbox': [200.0, 150.0, 120.0, 100.0], 'id': 102},
    ]
    test_preds = [
        {'image_id': 139, 'category_id': 1, 'bbox': [60.0, 60.0, 90.0, 70.0], 'score': 0.95, 'id': 1},
        {'image_id': 139, 'category_id': 2, 'bbox': [210.0, 160.0, 110.0, 90.0], 'score': 0.88, 'id': 2},
        {'image_id': 139, 'category_id': 1, 'bbox': [300.0, 50.0, 70.0, 120.0], 'score': 0.70, 'id': 3},
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

    # 확대/축소 및 패닝 테스트 안내
    info_text = "Mouse Wheel: Zoom In/Out\nMiddle Mouse Button Drag: Pan Image"
    # canvas.create_text(10, 10, anchor="nw", text=info_text, font=("Arial", 10), fill="black")
    # 위 create_text는 이미지 로딩 후 _update_display()에 의해 redraw_annotations()가 호출되면서 지워질 수 있음
    # GUI의 다른 부분에 표시하는 것이 좋음. 여기서는 테스트 목적으로 콘솔에 출력
    print("\n" + info_text + "\n")


    root.mainloop()

    # 테스트용 임시 이미지 삭제
    if 'dummy_img_path' in locals() and os.path.exists(dummy_img_path):
        try:
            os.remove(dummy_img_path)
            print(f"Removed dummy image: {dummy_img_path}")
        except Exception as e_remove:
            print(f"Error removing dummy image: {e_remove}")