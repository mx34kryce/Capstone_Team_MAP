# interactive_canvas.py
import tkinter as tk
from PIL import Image, ImageTk
import os

# visualizer 모듈의 색상 함수 재사용 또는 여기서 정의
DEFAULT_COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
]
def get_color(category_id):
    idx = category_id % len(DEFAULT_COLORS)
    return DEFAULT_COLORS[idx]

class InteractiveCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.scale = 1.0             # 현재 확대 비율
        self.pil_image = None        # 원본 PIL.Image
        self.image_on_canvas = None
        self.tk_image = None
        self.annotations = [] # 현재 표시/수정 중인 annotation 리스트 (예측만 해당될 수 있음)
        self.categories = {}
        self.confidence_threshold = 0.5 # 초기값

        # 드래그 상태 변수
        self._drag_data = {"x": 0, "y": 0, "item": None, "type": None} # item: 캔버스 객체 ID, type: 'move' or 'resize_...'
        self._selected_item = None # 현재 선택된 annotation의 캔버스 객체 ID
        self._selected_ann_idx = -1 # 현재 선택된 annotation의 self.annotations 인덱스

        # 이벤트 바인딩
        self.bind("<ButtonPress-1>", self.on_button_press)
        self.bind("<B1-Motion>", self.on_move_press)
        self.bind("<ButtonRelease-1>", self.on_button_release)
        self.bind("<MouseWheel>", self.on_mousewheel)
        # Linux 바인딩(마우스 휠휠)
        self.bind("<Button-4>",  self.on_mousewheel)
        self.bind("<Button-5>",  self.on_mousewheel)

        # 콜백 함수 (annotation 변경 시 호출)
        self.annotation_update_callback = None

    def set_annotation_update_callback(self, callback):
        self.annotation_update_callback = callback

    def load_image(self, image_path):
        """이미지를 로드하고 스케일·어노테이션을 초기화합니다."""
        try:
            self.pil_image = Image.open(image_path).convert("RGB")
            self.scale = 1.0
            self._update_image()
            self.clear_annotations()
        except Exception as e:
            print(f"오류: 이미지 로드 실패 - {e}")
            self.delete("all")
            self.tk_image = None
            self.image_on_canvas = None

    def set_data(self, annotations, categories, confidence_threshold):
        self.annotations = annotations # 수정 가능한 예측 annotation 리스트
        self.categories = categories
        self.confidence_threshold = confidence_threshold
        self.redraw_annotations()

    def redraw_annotations(self):
        self.delete("annotation") # 기존 annotation 관련 객체 삭제 (태그 사용)

        for idx, ann in enumerate(self.annotations):
            if ann['score'] >= self.confidence_threshold:
                bbox = ann['bbox']
                cat_id = ann['category_id']
                score = ann['score']
                label = self.categories.get(cat_id, {}).get('name', f'ID:{cat_id}')
                color = get_color(cat_id)

                xmin, ymin, w, h = bbox
                xmax, ymax = xmin + w, ymin + h

                # 바운딩 박스 그리기 (태그 추가)
                rect_id = self.create_rectangle(xmin, ymin, xmax, ymax, outline=color, width=2, tags=("annotation", f"ann_{idx}", "bbox"))
                # 레이블 텍스트 그리기 (태그 추가)
                text_content = f"{label} ({score:.2f})"
                text_id = self.create_text(xmin + 2, ymin + 2, anchor="nw", text=text_content, fill=color, tags=("annotation", f"ann_{idx}", "label"))

                # 크기 조절 핸들 추가 (선택 사항)
                # 예: 네 귀퉁이에 작은 사각형 추가
                handle_size = 6
                self.create_rectangle(xmin - handle_size/2, ymin - handle_size/2, xmin + handle_size/2, ymin + handle_size/2, fill=color, outline='black', tags=("annotation", f"ann_{idx}", "resize_tl"))
                self.create_rectangle(xmax - handle_size/2, ymin - handle_size/2, xmax + handle_size/2, ymin + handle_size/2, fill=color, outline='black', tags=("annotation", f"ann_{idx}", "resize_tr"))
                self.create_rectangle(xmin - handle_size/2, ymax - handle_size/2, xmin + handle_size/2, ymax + handle_size/2, fill=color, outline='black', tags=("annotation", f"ann_{idx}", "resize_bl"))
                self.create_rectangle(xmax - handle_size/2, ymax - handle_size/2, xmax + handle_size/2, ymax + handle_size/2, fill=color, outline='black', tags=("annotation", f"ann_{idx}", "resize_br"))

    def _update_image(self):
        # 리사이즈된 이미지를 PhotoImage 로 변환
        w, h = self.pil_image.size
        sw, sh = int(w * self.scale), int(h * self.scale)
        resized = self.pil_image.resize((sw, sh), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        if self.image_on_canvas:
            self.delete(self.image_on_canvas)
        self.image_on_canvas = self.create_image(0, 0, anchor="nw", image=self.tk_image)
        # 축소·확대 후 annotation 재그리기
        self.redraw_annotations()

    def clear_annotations(self):
        self.delete("annotation")
        self.annotations = []
        self._selected_item = None
        self._selected_ann_idx = -1

    def on_button_press(self, event):
        # 클릭된 위치의 가장 위에 있는 객체 찾기
        item = self.find_closest(event.x, event.y)[0] # find_closest는 튜플 반환
        tags = self.gettags(item)

        if "annotation" in tags:
            self._drag_data["item"] = item
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y

            # 어떤 종류의 객체인지 확인 (bbox, resize handle 등)
            if "bbox" in tags:
                self._drag_data["type"] = "move"
            elif "resize_tl" in tags: self._drag_data["type"] = "resize_tl"
            elif "resize_tr" in tags: self._drag_data["type"] = "resize_tr"
            elif "resize_bl" in tags: self._drag_data["type"] = "resize_bl"
            elif "resize_br" in tags: self._drag_data["type"] = "resize_br"
            else:
                 self._drag_data["type"] = None # 레이블 등 다른 부분 클릭

            # 선택된 annotation 인덱스 찾기
            for tag in tags:
                if tag.startswith("ann_"):
                    try:
                        self._selected_ann_idx = int(tag.split("_")[1])
                        self._selected_item = item # 선택 표시 등에 사용 가능
                        print(f"Annotation {self._selected_ann_idx} 선택됨")
                        break
                    except (IndexError, ValueError):
                        pass
        else:
            # 배경 클릭 시 선택 해제
            self._drag_data["item"] = None
            self._drag_data["type"] = None
            self._selected_item = None
            self._selected_ann_idx = -1
            print("선택 해제됨")


    def on_move_press(self, event):
        if self._drag_data["item"] is None or self._selected_ann_idx == -1:
            return

        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]

        ann = self.annotations[self._selected_ann_idx]
        bbox = ann['bbox']
        xmin, ymin, w, h = bbox
        xmax, ymax = xmin + w, ymin + h

        new_bbox = list(bbox) # 복사본 사용

        # 이동 또는 크기 조절 로직
        if self._drag_data["type"] == "move":
            new_bbox[0] = xmin + delta_x
            new_bbox[1] = ymin + delta_y
            # 연관된 모든 객체 이동 (박스, 레이블, 핸들)
            item_tag = f"ann_{self._selected_ann_idx}"
            self.move(item_tag, delta_x, delta_y)

        elif self._drag_data["type"] == "resize_br": # 오른쪽 아래 핸들
            new_w = max(1, w + delta_x) # 최소 크기 1 보장
            new_h = max(1, h + delta_y)
            new_bbox[2] = new_w
            new_bbox[3] = new_h
            self.coords(self._drag_data["item"], xmin + new_w - 3, ymin + new_h - 3, xmin + new_w + 3, ymin + new_h + 3) # 핸들 위치 업데이트
            # 박스 및 다른 핸들 위치 업데이트 필요... (구현 복잡)
            self.coords(self.find_withtag(f"ann_{self._selected_ann_idx} and bbox")[0], xmin, ymin, xmin + new_w, ymin + new_h)


        # TODO: 다른 resize 핸들 로직 추가

        # 실제 annotation 데이터 업데이트 (임시, ButtonRelease에서 최종 업데이트 권장)
        # self.annotations[self._selected_ann_idx]['bbox'] = new_bbox

        # 현재 드래그 위치 업데이트
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

        # 실시간 업데이트가 필요하면 여기서 redraw 또는 부분 업데이트
        # self.redraw_annotations() # 성능 문제 유발 가능

    def on_button_release(self, event):
        if self._drag_data["item"] is None or self._selected_ann_idx == -1:
            return

        # 최종 위치/크기 계산 및 annotation 데이터 업데이트
        # on_move_press에서 계산된 최종 위치를 기반으로 bbox 업데이트
        # 예시: 이동의 경우
        if self._drag_data["type"] == "move":
            final_coords = self.coords(self.find_withtag(f"ann_{self._selected_ann_idx} and bbox")[0])
            if len(final_coords) == 4:
                xmin, ymin, xmax, ymax = final_coords
                new_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                self.annotations[self._selected_ann_idx]['bbox'] = [round(c, 2) for c in new_bbox] # 소수점 정리
                print(f"Annotation {self._selected_ann_idx} 이동 완료: {self.annotations[self._selected_ann_idx]['bbox']}")

                # 콜백 호출하여 변경 알림
                if self.annotation_update_callback:
                    self.annotation_update_callback(self._selected_ann_idx, self.annotations[self._selected_ann_idx])

        # TODO: Resize 완료 시 bbox 업데이트 로직 추가

        # 드래그 상태 초기화
        self._drag_data = {"x": 0, "y": 0, "item": None, "type": None}
        # 선택 상태는 유지할 수 있음 (다른 동작을 위해)

        # 최종 상태로 다시 그리기 (필요시)
        self.redraw_annotations()
        
    def on_mousewheel(self, event):
        # Windows: event.delta, Linux: event.num
        factor = 1.0
        if hasattr(event, 'delta'):
            factor = 1.0 + (event.delta / 1200)   # 10% 단위로 조절
        elif event.num == 4:
            factor = 1.1
        elif event.num == 5:
            factor = 0.9
        # 스케일 범위 제한
        new_scale = min(max(self.scale * factor, 0.2), 5.0)
        if abs(new_scale - self.scale) < 1e-3:
            return
        self.scale = new_scale
        self._update_image()
    
    def redraw_annotations(self):
        self.delete("annotation")
        for idx, ann in enumerate(self.annotations):
            if ann['score'] < self.confidence_threshold: continue
            xmin, ymin, w, h = ann['bbox']
            # 스케일 적용
            xmin, ymin, w, h = xmin*self.scale, ymin*self.scale, w*self.scale, h*self.scale
            xmax, ymax = xmin + w, ymin + h
            color = get_color(ann['category_id'])
            self.create_rectangle(xmin, ymin, xmax, ymax,
                                  outline=color, width=2,
                                  tags=("annotation", f"ann_{idx}", "bbox"))
            self.create_text(xmin+2, ymin+2,
                             text=f"{self.categories[ann['category_id']]['name']} ({ann['score']:.2f})",
                             fill=color,
                             anchor="nw",
                             tags=("annotation", f"ann_{idx}", "label"))
            # … resize 핸들 등도 동일하게 스케일 적용 …



# 예시 사용법 (테스트용)
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Interactive Canvas Test")
    root.geometry("800x600")

    canvas = InteractiveCanvas(root, bg="white")
    canvas.pack(fill="both", expand=True)

    # 테스트용 이미지 로드 (실제 경로 사용)
    test_image_path = 'val2017/000000000139.jpg'
    if os.path.exists(test_image_path):
        canvas.load_image(test_image_path)
    else:
        print(f"테스트 이미지 없음: {test_image_path}")

    # 테스트용 annotation 데이터 설정
    test_annotations = [
        {'image_id': 139, 'category_id': 1, 'bbox': [100.0, 100.0, 150.0, 150.0], 'score': 0.95, 'id': 1},
        {'image_id': 139, 'category_id': 2, 'bbox': [300.0, 200.0, 100.0, 80.0], 'score': 0.88, 'id': 2},
    ]
    test_categories = {
        1: {'id': 1, 'name': 'ObjectA'},
        2: {'id': 2, 'name': 'ObjectB'},
    }

    def on_update(index, annotation):
        print(f"Callback: Annotation {index} 업데이트됨: {annotation}")

    canvas.set_annotation_update_callback(on_update)
    canvas.set_data(test_annotations, test_categories, 0.5)

    root.mainloop()