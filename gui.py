# annotator_gui.py
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import os
import copy
import numpy as np  # For PR curve data

# Matplotlib imports for PR Curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 다른 모듈 임포트
import coco_loader
import map_calculator
from interactive_canvas import InteractiveCanvas


class AnnotatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Annotation Tool")
        master.geometry("1400x900")
        master.tk.call('source', 'azure.tcl')
        master.tk.call("set_theme", "dark")

        # 데이터 변수
        self.gt_images = None
        self.gt_annotations = None
        self.categories = None
        self.pred_annotations_all = None
        self.current_image_id = None
        self.current_image_path = None
        self.current_gt_anns = []
        self.current_pred_anns = []
        self.image_dir = ""
        self.class_visibility = {}
        self.instance_visibility = {}
        self.class_expanded = {}
        self.current_pr_prec = None
        self.current_pr_rec = None
        self.selected_pr_class_id = None
        self.instance_numbers = {}

        # 이미지 메타데이터
        self.image_metadata = {}  # 이미지 ID를 키로, 메타데이터 딕셔너리를 값으로 가짐

        # 썸네일 관련 변수
        self.thumbnail_cache = {}
        self.thumbnail_size = (64, 64) # 썸네일 크기
        self.placeholder_thumbnail = None
        self.scroll_debounce_id = None

        # 탐색기 뷰 관련 변수
        self.explorer_canvas = None
        self.explorer_scrollbar_y = None
        self.all_image_ids_ordered = [] # 정렬된 이미지 ID 목록
        self.canvas_item_map = {} # image_id -> {'thumb': canvas_id, 'text': canvas_id, 'bg': canvas_id}
        self.item_height_in_explorer = self.thumbnail_size[1] + 40 # 각 아이템의 높이 (썸네일 + 텍스트 + 메타데이터 + 패딩)
        self.item_padding = 2
        self.selected_explorer_image_id = None
        self.style = None # ttk.Style 객체를 저장할 인스턴스 변수


        # UI 요소 생성
        self._create_widgets()
        self._create_placeholder_thumbnail()


        # 초기 상태 설정
        self._update_ui_state()

    def _create_placeholder_thumbnail(self):
        # 로드 전 또는 실패 시 표시할 기본 이미지 생성
        try:
            img = Image.new("RGB", self.thumbnail_size, "gray80")
            # 간단한 X 표시 추가 (선택 사항)
            # from PIL import ImageDraw
            # draw = ImageDraw.Draw(img)
            # draw.line((0, 0) + self.thumbnail_size, fill="black", width=1)
            # draw.line((0, self.thumbnail_size[1], self.thumbnail_size[0], 0), fill="black", width=1)
            self.placeholder_thumbnail = ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error creating placeholder thumbnail: {e}")
            self.placeholder_thumbnail = None


    def _create_widgets(self):
        self.style = ttk.Style(self.master) # self.style에 할당

        # --- Top Frame ---
        top_frame = ttk.Frame(self.master, padding="5 5 5 0")
        top_frame.pack(side=tk.TOP, fill=tk.X)

        btn_load_gt = ttk.Button(top_frame, text="Load GT Annotations", command=self.load_gt_data)
        btn_load_gt.pack(side=tk.LEFT, padx=5)
        btn_load_pred = ttk.Button(top_frame, text="Load Predictions", command=self.load_pred_data)
        btn_load_pred.pack(side=tk.LEFT, padx=5)
        btn_load_img_dir = ttk.Button(top_frame, text="Select Image Directory", command=self.select_image_dir)
        btn_load_img_dir.pack(side=tk.LEFT, padx=5)

        btn_reset = ttk.Button(top_frame, text="Reset Annotations", command=self.reset_annotations, state=tk.DISABLED)
        btn_reset.pack(side=tk.LEFT, padx=5)
        self.reset_btn = btn_reset

        # 전체 map계산 버튼
        self.calc_dataset_map_btn = ttk.Button(top_frame, text="Calculate Dataset mAP", command=self.calculate_dataset_map, 
                                               state=tk.DISABLED)
        self.calc_dataset_map_btn.pack(side=tk.LEFT, padx=5)

        # --- Bottom Frame ---
        bottom_frame = ttk.Frame(self.master, padding="5 0 5 5")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(bottom_frame, text="Ready", anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.progress_bar = ttk.Progressbar(bottom_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)

        # --- Main Frame (Left, Center, Right 분할) ---
        main_frame = ttk.Frame(self.master, padding="5 5 5 5")
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # --- Left Frame: 이미지 목록 및 ap score, 클래스 가시성 ---
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.grid(row=0, column=0, sticky="ns")

        # 이미지 목록 (캔버스 기반 탐색기 뷰로 변경)
        ttk.Label(left_frame, text="Images:").pack(anchor="w")
        explorer_outer_frame = ttk.Frame(left_frame)
        explorer_outer_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.explorer_canvas = tk.Canvas(explorer_outer_frame, background=self.style.lookup('TFrame', 'background'), highlightthickness=0)
        self.explorer_scrollbar_y = ttk.Scrollbar(explorer_outer_frame, orient="vertical", command=self.explorer_canvas.yview)
        self.explorer_canvas.configure(yscrollcommand=self.explorer_scrollbar_y.set)

        self.explorer_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.explorer_canvas.pack(side=tk.LEFT, fill="both", expand=True)

        self.explorer_canvas.bind('<Configure>', lambda e: self._populate_explorer_view()) # 캔버스 크기 변경 시 재구성
        # 스크롤 이벤트는 yscrollcommand를 통해 _on_explorer_scroll에서 처리되도록 할 것이므로 직접 바인딩은 제거
        # 대신, yscrollcommand가 호출될 때 _on_explorer_scroll이 트리거되도록 scrollbar의 command와 canvas의 yscrollcommand를 설정

        # mAP Display
        self.map_label = ttk.Label(left_frame, text="Current Image AP: N/A", font=("Arial", 10))
        self.map_label.pack(anchor="w", pady=5)

        # --- Dataset mAP Label ---
        self.dataset_map_label = ttk.Label(left_frame, text="Dataset mAP: N/A", font=("Arial", 10))
        self.dataset_map_label.pack(anchor="w", pady=(2, 10))

        # 클래스 가시성
        visibility_frame = ttk.LabelFrame(left_frame, text="Class Visibility", padding="5")
        visibility_frame.pack(fill="both", expand=True, pady=5)

        # 현재 테마의 TFrame 배경색을 가져옵니다.
        frame_bg_color = self.style.lookup('TFrame', 'background')

        vis_canvas = tk.Canvas(visibility_frame, borderwidth=0, background=frame_bg_color, highlightthickness=0)
        self.class_checkbox_frame = ttk.Frame(vis_canvas, padding="2") # ttk.Frame은 기본적으로 테마의 배경색을 따릅니다.
        vis_scrollbar = ttk.Scrollbar(visibility_frame, orient="vertical", command=vis_canvas.yview)        
        vis_canvas.configure(yscrollcommand=vis_scrollbar.set)

        vis_scrollbar.pack(side="right", fill="y")
        vis_canvas.pack(side="left", fill="both", expand=True)
        vis_canvas_window = vis_canvas.create_window((0, 0), window=self.class_checkbox_frame, anchor="nw")

        def _configure_vis_canvas(event):
            vis_canvas.configure(scrollregion=vis_canvas.bbox("all"))
            vis_canvas.itemconfig(vis_canvas_window, width=event.width)

        def _configure_checkbox_frame(event):
            vis_canvas.configure(scrollregion=vis_canvas.bbox("all"))

        vis_canvas.bind("<Configure>", _configure_vis_canvas)
        self.class_checkbox_frame.bind("<Configure>", _configure_checkbox_frame)

        # --- Right Frame: 컨트롤, 정보, PR Curve ---
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.grid(row=0, column=2, sticky="ns")

        ttk.Label(right_frame, text="Controls", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 10))

        # Confidence Threshold
        self.conf_frame = ttk.LabelFrame(right_frame, text="Confidence Threshold", padding="5")
        self.conf_frame.pack(fill=tk.X, pady=5)
        self.conf_value_label = ttk.Label(self.conf_frame, text="0.50", width=5, anchor="e")
        self.conf_value_label.pack(side=tk.RIGHT, padx=(5, 0))
        btn_conf_minus = ttk.Button(self.conf_frame, text="-", width=2, command=lambda: self.adjust_slider(self.conf_slider, -0.01))
        btn_conf_minus.pack(side=tk.LEFT)
        self.conf_slider = ttk.Scale(self.conf_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=self.on_threshold_change)
        self.conf_slider.set(0.5)
        self.conf_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        btn_conf_plus = ttk.Button(self.conf_frame, text="+", width=2, command=lambda: self.adjust_slider(self.conf_slider, 0.01))
        btn_conf_plus.pack(side=tk.LEFT)

        # IoU Threshold
        self.iou_frame = ttk.LabelFrame(right_frame, text="IoU Threshold (for mAP)", padding="5")
        self.iou_frame.pack(fill=tk.X, pady=5)
        self.iou_value_label = ttk.Label(self.iou_frame, text="0.50", width=5, anchor="e")
        self.iou_value_label.pack(side=tk.RIGHT, padx=(5, 0))
        btn_iou_minus = ttk.Button(self.iou_frame, text="-", width=2, command=lambda: self.adjust_slider(self.iou_slider, -0.05))
        btn_iou_minus.pack(side=tk.LEFT)
        self.iou_slider = ttk.Scale(self.iou_frame, from_=0.05, to=0.95, orient=tk.HORIZONTAL, command=self.on_threshold_change)
        self.iou_slider.set(0.5)
        self.iou_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        btn_iou_plus = ttk.Button(self.iou_frame, text="+", width=2, command=lambda: self.adjust_slider(self.iou_slider, 0.05))
        btn_iou_plus.pack(side=tk.LEFT)

        # --- PR Curve Section ---
        pr_curve_frame = ttk.LabelFrame(right_frame, text="Precision-Recall Curve", padding="5")
        pr_curve_frame.pack(fill="both", expand=True, pady=5)

        # PR Curve Class Selector
        pr_class_select_frame = ttk.Frame(pr_curve_frame)
        pr_class_select_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(pr_class_select_frame, text="Class:").pack(side=tk.LEFT, padx=(0, 5))
        self.pr_class_var = tk.StringVar()
        self.pr_class_combobox = ttk.Combobox(pr_class_select_frame, textvariable=self.pr_class_var, state="readonly", width=25)
        self.pr_class_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.pr_class_combobox.bind("<<ComboboxSelected>>", self.on_pr_class_select)

        # Matplotlib Canvas Placeholder
        self.pr_fig = Figure(figsize=(4, 3), dpi=100)
        self.pr_ax = self.pr_fig.add_subplot(111)
        self.pr_canvas_widget = FigureCanvasTkAgg(self.pr_fig, master=pr_curve_frame)
        self.pr_canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        self.pr_ax.set_title("PR Curve")
        self.pr_ax.set_xlabel("Recall")
        self.pr_ax.set_ylabel("Precision")
        self.pr_ax.set_xlim(0, 1)
        self.pr_ax.set_ylim(0, 1.05)
        self.pr_ax.grid(True)
        self.pr_fig.tight_layout()
        self.pr_canvas_widget.draw()

        # Edit Label Button
        self.edit_label_button = ttk.Button(right_frame, text="Edit Selected Label", command=self.edit_selected_label, state=tk.DISABLED)
        self.edit_label_button.pack(fill=tk.X, pady=5)

        # Save Annotations Button
        self.save_button = ttk.Button(right_frame, text="Save Modified Annotations", command=self.save_annotations, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, pady=5)

        # --- Center Frame: 이미지 및 Annotation 표시 ---
        center_frame = ttk.Frame(main_frame, padding="5")
        center_frame.grid(row=0, column=1, sticky="nsew")
        center_frame.columnconfigure(0, weight=1)
        center_frame.rowconfigure(0, weight=1)

        self.canvas = InteractiveCanvas(center_frame, bg="lightgrey")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.set_annotation_update_callback(self.on_annotation_update)

    def _on_explorer_scroll(self, *args):
        # Canvas의 yview를 먼저 호출하여 스크롤을 적용
        # 이 메서드는 scrollbar의 command나 canvas의 yscrollcommand에 직접 연결되지 않고,
        # yscrollcommand에 의해 _update_explorer_view_items가 호출되도록 하는 것이 더 일반적임.
        # 여기서는 ttk.Scrollbar의 command가 self.explorer_canvas.yview로 설정되어 있고,
        # self.explorer_canvas의 yscrollcommand가 self.explorer_scrollbar_y.set으로 되어 있음.
        # 스크롤 발생 시 썸네일 업데이트를 위해 yscrollcommand를 가로채거나,
        # 또는 스크롤 후 명시적으로 업데이트 함수를 호출해야 함.
        # ttk.Scrollbar는 command를 직접 실행하므로, canvas.yview를 호출하고 그 다음에 우리 로직을 실행.
        
        self.explorer_canvas.yview(*args) # 실제 스크롤 적용

        if self.scroll_debounce_id:
            self.master.after_cancel(self.scroll_debounce_id)
        self.scroll_debounce_id = self.master.after_idle(self._update_explorer_view_items)


    def _load_thumbnail(self, image_id_str):
        image_id = int(image_id_str)
        if not self.image_dir or not self.gt_images or image_id not in self.gt_images:
            return self.placeholder_thumbnail

        if image_id_str in self.thumbnail_cache:
            # 캐시에 플레이스홀더가 아닌 실제 이미지가 있는지 확인
            if self.thumbnail_cache[image_id_str] != self.placeholder_thumbnail:
                return self.thumbnail_cache[image_id_str]
            # 캐시에 플레이스홀더가 있다면, 다시 로드 시도 (예: 이미지 파일이 나중에 생겼을 수 있음)

        image_info = self.gt_images[image_id]
        img_path = coco_loader.get_image_path(image_info, self.image_dir)

        if not img_path or not os.path.exists(img_path):
            self.thumbnail_cache[image_id_str] = self.placeholder_thumbnail # 실패 시 플레이스홀더 캐싱
            return self.placeholder_thumbnail
        
        try:
            img = Image.open(img_path)
            img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
            photo_img = ImageTk.PhotoImage(img)
            self.thumbnail_cache[image_id_str] = photo_img
            return photo_img
        except Exception as e:
            print(f"Error loading thumbnail for image ID {image_id} ({img_path}): {e}")
            self.thumbnail_cache[image_id_str] = self.placeholder_thumbnail # 예외 발생 시 플레이스홀더 캐싱
            return self.placeholder_thumbnail

    def _update_explorer_view_items(self):
        if not self.image_dir or not self.gt_images or not self.all_image_ids_ordered:
            self.explorer_canvas.delete("all_items")
            self.canvas_item_map.clear()
            # 스크롤 영역도 초기화
            self.explorer_canvas.config(scrollregion=(0, 0, self.explorer_canvas.winfo_width(), 0))
            return

        canvas_width = self.explorer_canvas.winfo_width()
        if canvas_width <= 1: # 아직 캔버스가 제대로 그려지지 않음
            self.master.after(50, self._update_explorer_view_items) # 잠시 후 다시 시도
            return

        # 1. 보이는 아이템 인덱스 범위 결정
        try:
            scroll_top_fraction, _ = self.explorer_canvas.yview()
        except tk.TclError: # 위젯이 파괴되었거나 아직 준비되지 않은 경우
            return
            
        content_height = len(self.all_image_ids_ordered) * self.item_height_in_explorer
        visible_top_y = scroll_top_fraction * content_height
        
        first_visible_idx = int(visible_top_y / self.item_height_in_explorer)
        # 한 화면에 보이는 아이템 수 + 버퍼
        num_items_in_view_approx = int(self.explorer_canvas.winfo_height() / self.item_height_in_explorer) + 1 

        # 2. 렌더링/캐싱할 아이템 범위 결정 (현재 보이는 10개 + 위/아래 10개 캐싱)
        # 실제로는 "그릴" 범위를 결정하고, _load_thumbnail이 내부적으로 캐싱함.
        # "현재 있는 인덱스에 있는 10개" -> first_visible_idx 부터 10개
        # "미리 위쪽 10개, 아래쪽 10개"
        # 따라서 first_visible_idx - 10 부터 first_visible_idx + 10 + (현재 화면에 보이는 갯수) 까지 로드/그리기
        
        # 그릴 범위: 보이는 첫 아이템 기준 -10 ~ +10 + 화면에 보이는 갯수
        # 이렇게 하면 화면에 보이는 것과 그 주변이 그려짐
        render_buffer = 10 
        render_start_idx = max(0, first_visible_idx - render_buffer)
        render_end_idx = min(len(self.all_image_ids_ordered), first_visible_idx + num_items_in_view_approx + render_buffer)

        # 3. 기존 아이템 정리 및 새 아이템 그리기
        current_ids_to_render = set(self.all_image_ids_ordered[i] for i in range(render_start_idx, render_end_idx))
        
        # 삭제할 아이템 (현재 맵에 있지만 그릴 범위에 없는 것)
        ids_to_remove = []
        for img_id_str_map in self.canvas_item_map:
            img_id_map = int(img_id_str_map)
            if img_id_map not in current_ids_to_render:
                ids_to_remove.append(img_id_str_map)

        for img_id_str_map_to_remove in ids_to_remove:
            item_refs = self.canvas_item_map.pop(img_id_str_map_to_remove)
            if item_refs.get('bg'): self.explorer_canvas.delete(item_refs['bg'])
            if item_refs.get('thumb'): self.explorer_canvas.delete(item_refs['thumb'])
            if item_refs.get('text'): self.explorer_canvas.delete(item_refs['text'])
            
        # 추가/업데이트할 아이템
        for i in range(render_start_idx, render_end_idx):
            image_id = self.all_image_ids_ordered[i]
            image_id_str = str(image_id)
            
            y_pos = i * self.item_height_in_explorer + self.item_padding

            # 배경 사각형 (선택 및 클릭용)
            bg_color = "blue" if image_id == self.selected_explorer_image_id else self.style.lookup('TFrame', 'background')
            text_color = "white" if image_id == self.selected_explorer_image_id else self.style.lookup('TLabel', 'foreground')

            if image_id_str not in self.canvas_item_map:
                self.canvas_item_map[image_id_str] = {}
                
                bg_rect = self.explorer_canvas.create_rectangle(
                    self.item_padding, y_pos, 
                    canvas_width - self.item_padding, y_pos + self.item_height_in_explorer - self.item_padding * 2,
                    fill=bg_color, outline="", tags=("item_bg", f"item_bg_{image_id_str}")
                )
                self.canvas_item_map[image_id_str]['bg'] = bg_rect
                self.explorer_canvas.tag_bind(bg_rect, "<Button-1>", lambda e, img_id=image_id: self._handle_explorer_item_click(img_id))

                thumbnail_obj = self._load_thumbnail(image_id_str)
                thumb_item = self.explorer_canvas.create_image(
                    self.item_padding + self.thumbnail_size[0] / 2, 
                    y_pos + (self.item_height_in_explorer - self.item_padding*2) / 2, 
                    image=thumbnail_obj, tags=("item_thumb", f"item_thumb_{image_id_str}")
                )
                self.canvas_item_map[image_id_str]['thumb'] = thumb_item

                # 파일명과 메타데이터 정보를 포함한 텍스트 생성
                filename = self.image_metadata.get(image_id, {}).get("filename", f"ID: {image_id}")
                metadata = self.image_metadata.get(image_id, {})
                ap_score = metadata.get("ap", "N/A")
                class_count = metadata.get("classes", 0)
                instance_count = metadata.get("instances", 0)
                
                # AP score를 숫자로 포맷팅 (N/A가 아닌 경우)
                if ap_score != "N/A" and isinstance(ap_score, (int, float)):
                    ap_display = f"{ap_score:.3f}"
                else:
                    ap_display = str(ap_score)
                
                # 멀티라인 텍스트 생성
                display_text = f"{filename}\nAP: {ap_display} | Classes: {class_count} | Instances: {instance_count}"
                
                text_item = self.explorer_canvas.create_text(
                    self.item_padding + self.thumbnail_size[0] + 10, 
                    y_pos + (self.item_height_in_explorer - self.item_padding*2) / 2,
                    text=display_text, anchor="w", fill=text_color,
                    width=canvas_width - (self.thumbnail_size[0] + 20), # 텍스트 줄바꿈 너비
                    tags=("item_text", f"item_text_{image_id_str}")
                )
                self.canvas_item_map[image_id_str]['text'] = text_item
            else: # 이미 아이템이 존재하면 색상 등 업데이트
                self.explorer_canvas.itemconfig(self.canvas_item_map[image_id_str]['bg'], fill=bg_color)
                
                # 텍스트 내용도 업데이트 (메타데이터가 변경될 수 있음)
                filename = self.image_metadata.get(image_id, {}).get("filename", f"ID: {image_id}")
                metadata = self.image_metadata.get(image_id, {})
                ap_score = metadata.get("ap", "N/A")
                class_count = metadata.get("classes", 0)
                instance_count = metadata.get("instances", 0)
                
                if ap_score != "N/A" and isinstance(ap_score, (int, float)):
                    ap_display = f"{ap_score:.3f}"
                else:
                    ap_display = str(ap_score)
                
                display_text = f"{filename}\nAP: {ap_display} | Classes: {class_count} | Instances: {instance_count}"
                
                self.explorer_canvas.itemconfig(self.canvas_item_map[image_id_str]['text'], 
                                               fill=text_color, text=display_text)
                
                # 썸네일이 플레이스홀더였다가 실제 이미지로 변경된 경우 업데이트
                current_thumb_on_canvas = self.explorer_canvas.itemcget(self.canvas_item_map[image_id_str]['thumb'], "image")
                new_thumb_obj = self._load_thumbnail(image_id_str) # 캐시에서 가져오거나 로드
                if str(new_thumb_obj) != current_thumb_on_canvas : # PhotoImage 객체는 문자열 표현로 비교
                     self.explorer_canvas.itemconfig(self.canvas_item_map[image_id_str]['thumb'], image=new_thumb_obj)

        # 스크롤 영역 업데이트
        self.explorer_canvas.config(scrollregion=(0, 0, canvas_width, content_height))
        self.explorer_canvas.addtag_all("all_items")


    def _handle_explorer_item_click(self, image_id):
        if self.selected_explorer_image_id == image_id: # 이미 선택된 아이템 다시 클릭
            return

        # 이전 선택 해제
        if self.selected_explorer_image_id is not None:
            old_id_str = str(self.selected_explorer_image_id)
            if old_id_str in self.canvas_item_map and self.canvas_item_map[old_id_str].get('bg'):
                self.explorer_canvas.itemconfig(self.canvas_item_map[old_id_str]['bg'], fill=self.style.lookup('TFrame', 'background'))
                self.explorer_canvas.itemconfig(self.canvas_item_map[old_id_str]['text'], fill=self.style.lookup('TLabel', 'foreground'))


        self.selected_explorer_image_id = image_id
        
        # 새 아이템 선택 표시
        new_id_str = str(image_id)
        if new_id_str in self.canvas_item_map and self.canvas_item_map[new_id_str].get('bg'):
            self.explorer_canvas.itemconfig(self.canvas_item_map[new_id_str]['bg'], fill="blue") # 선택 색상
            self.explorer_canvas.itemconfig(self.canvas_item_map[new_id_str]['text'], fill="white")


        self.on_image_select_logic(image_id) # 실제 이미지 로딩 및 처리 로직 호출
        self._update_ui_state()


    def update_status(self, message, progress=None):
        self.status_label.config(text=message)
        if progress is not None:
            self.progress_bar['value'] = progress
        self.master.update_idletasks()

    def adjust_slider(self, slider, delta):
        current_value = slider.get()
        new_value = round(current_value + delta, 2)

        if slider == self.conf_slider:
            min_val, max_val = 0.0, 1.0
        elif slider == self.iou_slider:
            min_val, max_val = 0.05, 0.95
        else:
            return

        new_value = max(min_val, min(max_val, new_value))
        slider.set(new_value)

    def _update_ui_state(self):
        gt_loaded = self.gt_images is not None and self.categories is not None
        pred_loaded = self.pred_annotations_all is not None
        img_dir_set = bool(self.image_dir)
        # selected_items = self.image_treeview.selection() # Treeview 제거됨
        # img_selected = bool(selected_items) and self.current_image_id is not None
        img_selected = self.selected_explorer_image_id is not None and self.current_image_id is not None

        can_calculate_map = gt_loaded and pred_loaded and img_dir_set and img_selected
        self.conf_slider.config(state=tk.NORMAL if can_calculate_map else tk.DISABLED)
        self.iou_slider.config(state=tk.NORMAL if can_calculate_map else tk.DISABLED)

        for child in self.conf_frame.winfo_children():
            if isinstance(child, ttk.Button):
                child.config(state=tk.NORMAL if can_calculate_map else tk.DISABLED)
        for child in self.iou_frame.winfo_children():
            if isinstance(child, ttk.Button):
                child.config(state=tk.NORMAL if can_calculate_map else tk.DISABLED)

        self.edit_label_button.config(state=tk.NORMAL if img_selected and pred_loaded else tk.DISABLED)
        self.save_button.config(state=tk.NORMAL if pred_loaded else tk.DISABLED)

        can_plot_pr = self.current_image_id is not None and self.categories is not None and self.current_pred_anns is not None
        self.pr_class_combobox.config(state=tk.NORMAL if can_plot_pr else tk.DISABLED)

        #dataset-map계산 버튼 로직
        can_calc_dataset = self.gt_images is not None and self.pred_annotations_all is not None
        self.calc_dataset_map_btn.config(state=tk.NORMAL if can_calc_dataset else tk.DISABLED)

        can_reset = (self.current_image_id is not None and self.pred_annotations_all is not None)
        self.reset_btn.config(state=tk.NORMAL if can_reset else tk.DISABLED)

    def load_gt_data(self):
        filepath = filedialog.askopenfilename(
            title="Select Ground Truth COCO JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return

        self.update_status("Loading GT annotations...", 0)
        self.gt_images, self.gt_annotations, self.categories = coco_loader.load_coco_annotations(filepath)
        self.update_status("Processing GT data...", 50)

        if self.gt_images and self.categories:
            if self.pred_annotations_all:  # 예측도 로드된 경우에만 메타데이터 계산
                self._calculate_all_images_metadata() # 내부에서 _populate_explorer_view 호출
            else:
                # image_metadata가 아직 없을 수 있으므로, gt_images 기반으로 임시 생성 또는 _calculate_all_images_metadata 호출 준비
                # 여기서는 _populate_explorer_view가 image_metadata를 사용하므로, 먼저 채워야 함.
                # 간단하게는 파일명만이라도 채우거나, _calculate_all_images_metadata가 일부라도 실행되도록.
                # 지금은 _calculate_all_images_metadata가 pred_annotations_all이 없을 때도 GT 기반으로 채우도록 수정됨.
                self._calculate_all_images_metadata() # GT만 있어도 메타데이터 일부 계산 및 UI 채우기

            self._populate_visibility_checkboxes() # current_gt_anns 등이 필요하므로 이미지 선택 후 호출되거나, 초기화 필요
            self._update_pr_class_selector()

            messagebox.showinfo("Success", f"Loaded {len(self.gt_images)} images and {len(self.categories)} categories from GT.")
            self.update_status(f"GT Loaded: {len(self.gt_images)} images, {len(self.categories)} categories.", 100)
        else:
            messagebox.showerror("Error", "Failed to load GT annotations or categories.")
            self.update_status("Error loading GT.", 0)
            self.all_image_ids_ordered.clear() # 데이터 로드 실패 시 목록 비우기
            self._update_explorer_view_items() # 빈 목록으로 캔버스 업데이트
        self._update_ui_state()

    def _populate_visibility_checkboxes(self):
        """현재 이미지의 클래스별 · 인스턴스별 체크박스 재구성
           - 이전 on/off 상태 유지
           - 이전 토글(펼침/접힘) 상태 유지
           - 클래스별로 컨테이너 프레임을 만들어 인스턴스가 해당 클래스 바로 아래에 붙도록 함
           - 현재 confidence threshold에 따라 필터링된 예측만 표시
        """
        # 1) 기존 체크박스 및 토글 상태 백업
        prev_class_states    = {cat_id: var.get() for cat_id, var in self.class_visibility.items()}
        prev_inst_states     = {key: var.get()    for key,    var in self.instance_visibility.items()}
        prev_expanded_states = self.class_expanded.copy()

        # 2) 프레임 초기화
        for w in self.class_checkbox_frame.winfo_children():
            w.destroy()
        self.class_visibility.clear()
        self.instance_visibility.clear()
        self.class_expanded.clear()

        # GT/Pred annotation이 하나도 없으면 바로 리턴
        if not (self.current_gt_anns or self.current_pred_anns):
            return

        # 현재 confidence threshold로 예측 필터링
        conf_thresh = self.conf_slider.get()
        filtered_pred_anns = [
            pred for pred in self.current_pred_anns 
            if pred.get('score', 0.0) >= conf_thresh
        ]

        # 3) 화면에 등장하는 클래스 ID들을 이름순으로 정렬 (필터링된 예측 기준)
        present_cats = {
            ann['category_id'] for ann in self.current_gt_anns
        } | {
            ann['category_id'] for ann in filtered_pred_anns  # 필터링된 예측 사용
        }
        sorted_categories = sorted(
            ((cid, info) for cid, info in self.categories.items() if cid in present_cats),
            key=lambda item: item[1]['name']
        )

        for cat_id, cat_info in sorted_categories:
            class_name = cat_info['name']

            # 3-1) 이전에 펼침/접힘 상태가 있으면 재사용, 없으면 기본(True=펼친 상태)
            is_expanded = prev_expanded_states.get(cat_id, True)

            # 3-2) 해당 클래스의 AP 계산 (현재 IoU & Confidence 슬라이더 값 기준)
            gt_cat  = [ann for ann in self.current_gt_anns   if ann['category_id'] == cat_id]
            pr_cat  = [ann for ann in filtered_pred_anns if ann['category_id'] == cat_id]  # 이미 필터링된 예측 사용
            iou_thr = self.iou_slider.get()
            prec, rec, _ = map_calculator.get_pr_arrays(
                gt_cat, pr_cat,
                category_id=cat_id,
                iou_threshold=iou_thr
            )
            ap_value = map_calculator.calculate_ap(rec, prec) if (prec is not None and rec is not None) else 0.0

            # ─────────────────────────────────────────────────────────────────────
            # 4) 클래스별 컨테이너 프레임 생성: class_frame + inst_frame을 이 안에 묶음
            container_frame = ttk.Frame(self.class_checkbox_frame)
            container_frame.pack(fill="x", anchor="w", pady=(0, 2))

            # 4-1) 클래스 헤더용 프레임 (토글 버튼 + 체크박스)
            class_frame = ttk.Frame(container_frame)
            class_frame.pack(fill="x", anchor="w")

            # 4-2) 인스턴스 목록 프레임 (처음엔 is_expanded 상태에 따라 pack or pack_forget)
            inst_frame = ttk.Frame(container_frame)
            if is_expanded:
                inst_frame.pack(fill="x", anchor="w", padx=20)
            #     collapsed 상태일 땐 pack 하지 않음 (토글 버튼에서 제어)

            # 4-3) 토글 함수 정의: 해당 container_frame 내부의 inst_frame만 토글
            def make_toggle_func(frame, btn, cid):
                def _toggle():
                    if frame.winfo_ismapped():
                        frame.pack_forget()
                        btn.config(text="+")
                        self.class_expanded[cid] = False
                    else:
                        frame.pack(fill="x", anchor="w", padx=20)
                        btn.config(text="−")
                        self.class_expanded[cid] = True
                return _toggle

            # 4-4) 토글 버튼 생성: collapsed면 "+", expanded면 "−"
            btn_text = "−" if is_expanded else "+"
            toggle_btn = ttk.Button(class_frame, width=1, text=btn_text)
            toggle_btn.pack(side="left")

            # 4-5) 클래스 체크박스 생성: 이전 on/off 상태 유지, 없으면 기본 True
            init_class_state = prev_class_states.get(cat_id, True)
            class_var = tk.BooleanVar(value=init_class_state)
            label_text = f"{class_name} (AP={ap_value:.4f}, Conf≥{conf_thresh:.2f})"
            class_cb = ttk.Checkbutton(
                class_frame,
                text=label_text,
                variable=class_var,
                command=self.on_visibility_change
            )
            class_cb.pack(side="left", fill="x", expand=True)
            self.class_visibility[cat_id] = class_var

            # 4-6) 토글 버튼에 콜백 연결
            toggle_btn.config(command=make_toggle_func(inst_frame, toggle_btn, cat_id))

            # ─────────────────────────────────────────────────────────────────────
            # 5) GT 인스턴스 체크박스 생성 (inst_frame 안에)
            gt_list = []
            for idx, ann in enumerate(self.current_gt_anns):
                if ann['category_id'] != cat_id:
                    continue
                key = f"gt_{idx}"
                num = self.instance_numbers.get(key, 0)
                gt_list.append((num, key))
            gt_list.sort(key=lambda x: x[0])

            for num, key in gt_list:
                label = f"GT_{class_name}_{num}"
                init_inst_state = prev_inst_states.get(key, True)
                iv = tk.BooleanVar(value=init_inst_state)
                cb = ttk.Checkbutton(
                    inst_frame,
                    text=label,
                    variable=iv,
                    command=self.on_visibility_change
                )
                cb.pack(anchor="w", pady=(0, 1))
                self.instance_visibility[key] = iv

            # ─────────────────────────────────────────────────────────────────────
            # 6) Prediction 인스턴스 체크박스 생성 (inst_frame 안에) - 필터링된 예측만 표시
            pr_list = []
            for idx, ann in enumerate(self.current_pred_anns):
                if ann['category_id'] != cat_id:
                    continue
                # confidence threshold 체크 - 필터링된 예측만 체크박스로 표시
                if ann.get('score', 0.0) < conf_thresh:
                    continue
                key = f"pred_{idx}"
                num = self.instance_numbers.get(key, 0)
                pr_list.append((num, key))
            pr_list.sort(key=lambda x: x[0])

            for num, key in pr_list:
                label = f"PR_{class_name}_{num}"
                init_inst_state = prev_inst_states.get(key, True)
                iv = tk.BooleanVar(value=init_inst_state)
                cb = ttk.Checkbutton(
                    inst_frame,
                    text=label,
                    variable=iv,
                    command=self.on_visibility_change
                )
                cb.pack(anchor="w", pady=(0, 1))
                self.instance_visibility[key] = iv

            # 7) 최종적으로 해당 클래스의 토글(expanded/collapsed) 상태 저장
            self.class_expanded[cat_id] = is_expanded

    def _update_pr_class_selector(self):
        self.pr_class_combobox.set('')
        if not self.categories:
            self.pr_class_combobox['values'] = []
            return

        # 현재 이미지의 GT 어노테이션에 포함된 클래스만 필터링
        if self.current_gt_anns:
            gt_class_ids = {ann['category_id'] for ann in self.current_gt_anns}
            available_classes = [
                self.categories[cat_id]['name'] 
                for cat_id in gt_class_ids 
                if cat_id in self.categories
            ]
            class_names = ["Overall"] + sorted(available_classes)
        else:
            # GT 어노테이션이 없는 경우 Overall만 표시
            class_names = ["Overall"]
        
        self.pr_class_combobox['values'] = class_names
        
        # 기존 선택이 새로운 목록에 있는지 확인
        current_selection = self.pr_class_var.get()
        if current_selection not in class_names:
            self.pr_class_combobox.set("Overall")
            self.selected_pr_class_id = "Overall"
        else:
            # 기존 선택 유지
            self.pr_class_var.set(current_selection)

    def on_visibility_change(self):
        self.update_visualization_and_map()

    def load_pred_data(self):
        filepath = filedialog.askopenfilename(
            title="Select Prediction COCO JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return

        self.update_status("Loading predictions...", 0)
        self.pred_annotations_all = coco_loader.load_predictions(filepath)
        self.update_status("Processing predictions...", 50)

        if self.pred_annotations_all is not None:
            messagebox.showinfo("Success", f"Loaded predictions for {len(self.pred_annotations_all)} images.")
            self.update_status(f"Predictions loaded for {len(self.pred_annotations_all)} images.", 100)

            if self.gt_images:  # GT도 로드된 경우 메타데이터 계산 및 Treeview 업데이트
                self._calculate_all_images_metadata() # 내부에서 _populate_explorer_view 호출

            if self.current_image_id: # 현재 선택된 이미지가 있다면 해당 이미지 정보 갱신
                self.load_annotations_for_current_image()
                self.update_visualization_and_map()
        else:
            messagebox.showerror("Error", "Failed to load predictions.")
            self.update_status("Error loading predictions.", 0)
        self._update_ui_state()

    def select_image_dir(self):
        dirpath = filedialog.askdirectory(title="Select Image Directory")
        if not dirpath:
            return
        
        if self.image_dir != dirpath: # 이미지 디렉토리가 실제로 변경된 경우에만 캐시 초기화
            self.image_dir = dirpath
            self.thumbnail_cache.clear() # 캐시 초기화
            messagebox.showinfo("Success", f"Image directory set to: {self.image_dir}")
            self.update_status(f"Image directory set: {self.image_dir}", 100)
            
            # 탐색기 뷰 업데이트 (썸네일 다시 로드 필요)
            if self.gt_images: 
                self._populate_explorer_view() # 전체 목록 다시 구성 및 그리기
            
            if self.current_image_id: 
                self.load_image_and_annotations(self.current_image_id) # 현재 이미지 다시 로드
        else: # 같은 디렉토리를 다시 선택한 경우
            self.update_status(f"Image directory remains: {self.image_dir}", 100)

        self._update_ui_state()

    def on_image_select_logic(self, new_image_id): # 기존 on_image_select의 로직 부분
        if new_image_id == self.current_image_id and self.canvas.has_image(): # 이미 로드된 동일 이미지면 무시 (선택 효과는 _handle_explorer_item_click에서 처리)
            return

        self.current_image_id = new_image_id
        print(f"Selected Image ID: {self.current_image_id}")
        if not self.image_dir:
            messagebox.showwarning("Warning", "Please select the image directory first.")
            # 선택 해제 로직 필요 시 추가
            self.current_image_id = None
            self.selected_explorer_image_id = None # UI 선택도 해제
            # 이전 선택 UI 복원
            self._update_explorer_view_items() # 선택 효과 업데이트
            self.update_status("Select image directory first.", 0)
            return

        self.update_status(f"Loading image ID: {self.current_image_id}...", 0)
        self.load_image_and_annotations(self.current_image_id)
        
        self.update_status(f"Image ID: {self.current_image_id} loaded.", 100)
        # 선택된 이미지에 등장하는 클래스만 필터링하여 체크박스 업데이트
        gt_ids   = {ann['category_id'] for ann in self.current_gt_anns}
        pred_ids = {ann['category_id'] for ann in self.current_pred_anns}
        # image_class_ids = gt_ids.union(pred_ids) # 사용되지 않음
        
        self._compute_instance_numbers()
        self._populate_visibility_checkboxes()
        self._update_pr_class_selector() # PR 곡선 클래스 선택기 업데이트
        self.update_visualization_and_map() # 메인 캔버스 및 mAP 업데이트
        # self._update_ui_state() # _handle_explorer_item_click 마지막에 호출됨

    def load_image_and_annotations(self, image_id):
        if not self.gt_images or image_id not in self.gt_images:
            print(f"Error: Cannot find info for image ID {image_id}")
            self.update_status(f"Error loading image ID {image_id}.", 0)
            return

        image_info = self.gt_images[image_id]
        self.current_image_path = coco_loader.get_image_path(image_info, self.image_dir)

        if not self.current_image_path or not os.path.exists(self.current_image_path):
            messagebox.showerror("Error", f"Image file not found: {self.current_image_path}")
            self.canvas.delete("all")
            self.current_image_id = None
            self.update_status(f"Image file not found: {self.current_image_path}", 0)
            return

        self.update_status(f"Loading image file: {os.path.basename(self.current_image_path)}", 20)
        self.canvas.load_image(self.current_image_path)
        self.update_status(f"Loading annotations for image {image_id}", 50)

        self.load_annotations_for_current_image()

        self.update_visualization_and_map()
        self.update_status(f"Image {image_id} ready.", 100)

    def load_annotations_for_current_image(self):
        self.current_gt_anns = self.gt_annotations.get(self.current_image_id, []) if self.gt_annotations else []

        if self.pred_annotations_all:
            self.current_pred_anns = copy.deepcopy(self.pred_annotations_all.get(self.current_image_id, []))
        else:
            self.current_pred_anns = []

        print(f"Loaded {len(self.current_gt_anns)} GT annotations and {len(self.current_pred_anns)} predictions for image {self.current_image_id}")

    def update_visualization_and_map(self):
        if not self.current_image_id or not self.categories:
            self.clear_pr_curve()
            return

        conf_thresh = self.conf_slider.get()
        iou_thresh_map = self.iou_slider.get()
        iou_thresh_pr = iou_thresh_map

        visible_class_ids = {cat_id for cat_id, var in self.class_visibility.items() if var.get()}

        visible_insts = {
            k for k, var in self.instance_visibility.items() if var.get()
        }
        self.canvas.set_data(
            self.current_gt_anns,
            self.current_pred_anns,
            self.categories,
            conf_thresh,
            visible_class_ids,
            visible_insts      # ← 추가 파라미터
        )

        filtered_preds_for_map = [p for p in self.current_pred_anns if p['score'] >= conf_thresh]

        if self.current_gt_anns and filtered_preds_for_map:
            mean_ap, class_aps = map_calculator.calculate_map(
                self.current_gt_anns, filtered_preds_for_map, self.categories, iou_thresh_map
            )
            self.map_label.config(text=f"Current Image AP (IoU={iou_thresh_map:.2f}): {mean_ap:.4f}")
            '''
            self.class_ap_text.config(state=tk.NORMAL)
            self.class_ap_text.delete(1.0, tk.END)
            if class_aps:
                sorted_class_aps = sorted(class_aps.items(), key=lambda item: item[1], reverse=True)
                for cat_id, ap in sorted_class_aps:
                    cat_name = self.categories.get(cat_id, {}).get('name', f'ID:{cat_id}')
                    self.class_ap_text.insert(tk.END, f"- {cat_name}: {ap:.4f}\n")
            else:
                self.class_ap_text.insert(tk.END, "No AP data available.")
            self.class_ap_text.config(state=tk.DISABLED)
            '''
        else:
            self.map_label.config(text=f"Current Image AP (IoU={iou_thresh_map:.2f}): N/A")
            '''
            self.class_ap_text.config(state=tk.NORMAL)
            self.class_ap_text.delete(1.0, tk.END)
            self.class_ap_text.insert(tk.END, "GT or Predictions missing for AP.")
            self.class_ap_text.config(state=tk.DISABLED)
            '''

        self.draw_pr_curve(iou_thresh_pr)

        self.update_status(f"Image {self.current_image_id} visualization updated.", 100)

    def on_threshold_change(self, value):
        try:
            conf_val = float(value) if self.conf_slider.cget("state") != tk.DISABLED else self.conf_slider.get()
            iou_val = float(value) if self.iou_slider.cget("state") != tk.DISABLED else self.iou_slider.get()

            self.conf_value_label.config(text=f"{self.conf_slider.get():.2f}")
            self.iou_value_label.config(text=f"{self.iou_slider.get():.2f}")

            if self.current_image_id:
                # threshold 변경 시 체크박스를 재구성하여 필터링된 객체를 반영
                self._compute_instance_numbers()
                self._populate_visibility_checkboxes()
                self.update_visualization_and_map()
                
        except Exception as e:
            print(f"Error in on_threshold_change: {e}")

    def on_annotation_update(self, index, updated_annotation):
        print(f"GUI: Prediction {index} updated in canvas: {updated_annotation}")
        if 0 <= index < len(self.current_pred_anns):
            self.current_pred_anns[index] = updated_annotation

            self.update_visualization_and_map()
            self.update_status(f"Annotation {index} updated. Recalculating AP.", 50)
        else:
            print(f"Error: Invalid prediction index {index}")
            self.update_status(f"Error updating annotation {index}.", 0)

    def on_pr_class_select(self, event=None):
        selected_name = self.pr_class_var.get()
        if selected_name == "Overall":
            self.selected_pr_class_id = "Overall"
        else:
            found_id = None
            for cat_id, cat_info in self.categories.items():
                if cat_info['name'] == selected_name:
                    found_id = cat_id
                    break
            if found_id is not None:
                self.selected_pr_class_id = found_id
            else:
                print(f"Warning: Could not find category ID for name '{selected_name}'")
                self.selected_pr_class_id = "Overall"
                self.pr_class_combobox.set("Overall")

        print(f"PR Curve class selected: {self.selected_pr_class_id}")
        if self.current_image_id:
            iou_thresh_pr = self.iou_slider.get()
            self.draw_pr_curve(iou_thresh_pr)

    def draw_pr_curve(self, iou_threshold):
        if not self.current_image_id or not self.categories or self.selected_pr_class_id is None:
            self.clear_pr_curve()
            return

        calc_cat_id = None if self.selected_pr_class_id == "Overall" else self.selected_pr_class_id
        
        # confidence threshold 기반 예측 필터링 추가
        conf_thresh = self.conf_slider.get()
        filtered_pred_anns = [
            pred for pred in self.current_pred_anns 
            if pred.get('score', 0.0) >= conf_thresh
        ]

        prec, rec, num_gt = map_calculator.get_pr_arrays(
            self.current_gt_anns,
            filtered_pred_anns,  # 필터링된 예측 전달
            category_id=calc_cat_id,
            iou_threshold=iou_threshold
        )

        self.pr_ax.clear()
        if prec is not None and rec is not None:
            self.pr_ax.plot(rec, prec, marker='.', linestyle='-')

            if calc_cat_id is not None:
                ap = map_calculator.calculate_ap(rec, prec)
                title = f"PR Curve: {self.categories[calc_cat_id]['name']} (AP={ap:.3f}, Conf≥{conf_thresh:.2f})"
            else:
                title = f"PR Curve: Overall (IoU={iou_threshold:.2f}, Conf≥{conf_thresh:.2f})"

            self.pr_ax.set_title(title, fontsize=9)
        else:
            title_suffix = f"(Conf≥{conf_thresh:.2f}, No Data)"
            self.pr_ax.set_title(f"PR Curve: {self.pr_class_var.get()} {title_suffix}", fontsize=9)

        self.pr_ax.set_xlabel("Recall")
        self.pr_ax.set_ylabel("Precision")
        self.pr_ax.set_xlim(0, 1)
        self.pr_ax.set_ylim(0, 1.05)
        self.pr_ax.grid(True)
        self.pr_fig.tight_layout()
        self.pr_canvas_widget.draw()

    def clear_pr_curve(self):
        self.pr_ax.clear()
        self.pr_ax.set_title("PR Curve")
        self.pr_ax.set_xlabel("Recall")
        self.pr_ax.set_ylabel("Precision")
        self.pr_ax.set_xlim(0, 1)
        self.pr_ax.set_ylim(0, 1.05)
        self.pr_ax.grid(True)
        self.pr_fig.tight_layout()
        self.pr_canvas_widget.draw()

    def edit_selected_label(self):
        selected_pred_idx = self.canvas.get_selected_pred_index()

        if selected_pred_idx == -1:
            messagebox.showwarning("Warning", "Please select a prediction annotation (dashed box) on the canvas first.")
            return

        if not (0 <= selected_pred_idx < len(self.current_pred_anns)):
            messagebox.showerror("Error", "Selected annotation index is out of bounds.")
            return

        current_ann = self.current_pred_anns[selected_pred_idx]
        current_cat_id = current_ann['category_id']
        current_label = self.categories.get(current_cat_id, {}).get('name', f'ID:{current_cat_id}')

        category_choices = {cat['name']: cat_id for cat_id, cat in sorted(self.categories.items(), key=lambda item: item[1]['name'])}
        category_names = list(category_choices.keys())

        dialog = tk.Toplevel(self.master)
        dialog.title("Edit Label")
        dialog.geometry("300x150")
        dialog.transient(self.master)
        dialog.grab_set()

        ttk.Label(dialog, text=f"Current label: {current_label}").pack(pady=10)
        ttk.Label(dialog, text="Select new label:").pack(pady=5)

        selected_label = tk.StringVar(dialog)
        if current_label in category_names:
            selected_label.set(current_label)

        combobox = ttk.Combobox(dialog, textvariable=selected_label, values=category_names, state="readonly", width=30)
        combobox.pack(pady=5)
        combobox.focus()

        new_cat_id = None

        def on_ok():
            nonlocal new_cat_id
            chosen_label = selected_label.get()
            if chosen_label in category_choices:
                new_cat_id = category_choices[chosen_label]
                dialog.destroy()
            else:
                messagebox.showwarning("Selection Error", "Please select a valid label from the list.", parent=dialog)

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ok_button = ttk.Button(button_frame, text="OK", command=on_ok)
        ok_button.pack(side=tk.LEFT, padx=5)
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
        cancel_button.pack(side=tk.LEFT, padx=5)

        dialog.wait_window()

        if new_cat_id is not None and new_cat_id != current_cat_id:
            new_label_name = self.categories.get(new_cat_id, {}).get('name', f'ID:{new_cat_id}')
            print(f"Updating label for prediction {selected_pred_idx}: {current_label} -> {new_label_name} (ID: {new_cat_id})")
            self.current_pred_anns[selected_pred_idx]['category_id'] = new_cat_id

            self.on_annotation_update(selected_pred_idx, self.current_pred_anns[selected_pred_idx])
            self.update_status(f"Label updated for annotation {selected_pred_idx}.", 100)

    def save_annotations(self):
        if not self.pred_annotations_all:
            messagebox.showerror("Error", "No predictions loaded to save.")
            self.update_status("Save failed: No predictions loaded.", 0)
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Modified Predictions",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not save_path:
            self.update_status("Save cancelled.", 0)
            return

        self.update_status("Preparing data for saving...", 0)
        if self.current_image_id is not None and self.current_image_id in self.pred_annotations_all:
            self.pred_annotations_all[self.current_image_id] = copy.deepcopy(self.current_pred_anns)

        output_predictions = []
        for img_id, preds in self.pred_annotations_all.items():
            for p in preds:
                clean_p = {k: v for k, v in p.items() if k not in ['canvas_ids']}
                output_predictions.append(clean_p)

        self.update_status("Saving annotations...", 50)
        try:
            import json
            with open(save_path, 'w') as f:
                json.dump(output_predictions, f, indent=4)
            messagebox.showinfo("Success", f"Modified predictions saved to:\n{save_path}")
            self.update_status(f"Annotations saved to {os.path.basename(save_path)}", 100)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {e}")
            self.update_status(f"Error saving annotations: {e}", 0)

    def _calculate_image_metadata(self, image_id):
        """단일 이미지에 대한 메타데이터를 계산합니다."""
        if not self.gt_images or image_id not in self.gt_images:
            return None

        gt_anns_img = self.gt_annotations.get(image_id, [])
        pred_anns_img = self.pred_annotations_all.get(image_id, []) if self.pred_annotations_all else []
        categories_img = self.categories
        iou_thresh = self.iou_slider.get()

        filename = self.gt_images[image_id].get('file_name', f'Image ID: {image_id}')
        instance_count = len(gt_anns_img)
        class_count = len(set(ann['category_id'] for ann in gt_anns_img))

        ap_score = 0.0
        if gt_anns_img and pred_anns_img and categories_img:
            conf_thresh = self.conf_slider.get()
            filtered_preds = [p for p in pred_anns_img if p['score'] >= conf_thresh]
            if filtered_preds:
                mean_ap, _ = map_calculator.calculate_map(gt_anns_img, filtered_preds, categories_img, iou_thresh)
                ap_score = mean_ap
            elif not gt_anns_img:
                ap_score = 0.0
        elif not gt_anns_img:
            ap_score = 0.0

        return {
            "filename": filename,
            "ap": ap_score,
            "classes": class_count,
            "instances": instance_count
        }

    def _calculate_all_images_metadata(self):
        """모든 이미지에 대한 메타데이터를 계산하여 self.image_metadata에 저장하고, 탐색기 뷰를 채웁니다."""
        self.image_metadata.clear()
        
        # GT 이미지가 로드되지 않았으면 아무것도 하지 않음
        if not self.gt_images:
            self.all_image_ids_ordered.clear()
            self._populate_explorer_view() # 탐색기 뷰 클리어
            return

        # GT 이미지는 있지만, 예측이나 카테고리가 없을 경우 기본 메타데이터 생성
        if not self.pred_annotations_all or not self.categories:
            self.update_status("Calculating basic metadata for images...", 0)
            for i, img_id in enumerate(self.gt_images.keys()):
                gt_anns_img = self.gt_annotations.get(img_id, [])
                self.image_metadata[img_id] = {
                    "filename": self.gt_images[img_id].get('file_name', f'Image ID: {img_id}'),
                    "ap": "N/A", # AP 계산 불가
                    "classes": len(set(ann['category_id'] for ann in gt_anns_img)),
                    "instances": len(gt_anns_img)
                }
                if (i + 1) % 20 == 0 or (i + 1) == len(self.gt_images):
                    self.update_status(f"Basic metadata... {i+1}/{len(self.gt_images)}", int((i+1)/len(self.gt_images)*100))
            self.update_status("Basic metadata calculation complete.", 100)
            self._populate_explorer_view() # 탐색기 뷰 채우기
            return

        # 모든 데이터가 있을 경우 AP 포함 전체 메타데이터 계산
        self.update_status("Calculating full metadata for all images...", 0)
        total_images = len(self.gt_images)
        for i, img_id in enumerate(self.gt_images.keys()):
            metadata = self._calculate_image_metadata(img_id) # 기존 AP 계산 로직 사용
            if metadata:
                self.image_metadata[img_id] = metadata
            if (i + 1) % 10 == 0 or (i + 1) == total_images:
                self.update_status(f"Calculating full metadata... {i+1}/{total_images}", int((i+1)/total_images*100))
        self.update_status("Full metadata calculation complete.", 100)
        self._populate_explorer_view() # 탐색기 뷰 채우기


    def _populate_explorer_view(self):
        """
        self.image_metadata를 기반으로 self.all_image_ids_ordered를 설정하고,
        캔버스의 스크롤 영역을 설정한 후 _update_explorer_view_items를 호출합니다.
        """
        if not self.image_metadata and self.gt_images: # 메타데이터는 없지만 GT 이미지는 있는 경우 (예: 초기 로드)
            # 파일 이름 순으로 정렬된 ID 목록 생성
            sorted_gt_items = sorted(self.gt_images.items(), key=lambda item: item[1].get('file_name', str(item[0])))
            self.all_image_ids_ordered = [img_id for img_id, _ in sorted_gt_items]
        elif self.image_metadata:
            # 메타데이터의 파일 이름으로 정렬 (AP 등 다른 기준으로 정렬하려면 여기 수정)
            # 현재는 filename으로 정렬된 image_id 리스트를 만듦
            sorted_meta_items = sorted(self.image_metadata.items(), key=lambda item: item[1].get("filename", str(item[0])))
            self.all_image_ids_ordered = [img_id for img_id, _ in sorted_meta_items]
        else:
            self.all_image_ids_ordered = []

        # 기존 캔버스 아이템 모두 삭제 및 맵 초기화
        self.explorer_canvas.delete("all_items") # 이 태그를 가진 모든 아이템 삭제
        self.canvas_item_map.clear()

        if not self.all_image_ids_ordered:
            self.explorer_canvas.config(scrollregion=(0,0,self.explorer_canvas.winfo_width(),0))
            return

        content_height = len(self.all_image_ids_ordered) * self.item_height_in_explorer
        canvas_width = self.explorer_canvas.winfo_width()
        if canvas_width <=1 : canvas_width = 200 # 초기 너비 방어 코드

        self.explorer_canvas.config(scrollregion=(0, 0, canvas_width, content_height))
        
        # 스크롤바의 command를 _on_explorer_scroll로 설정하여 스크롤 시 업데이트 되도록 함
        # _create_widgets에서 scrollbar의 command는 canvas.yview로, canvas의 yscrollcommand는 scrollbar.set으로 이미 설정됨.
        # 스크롤 이벤트 발생 시 _on_explorer_scroll을 호출하기 위해 yscrollcommand를 수정하거나,
        # 여기서는 scrollbar의 command를 직접 _on_explorer_scroll로 변경
        self.explorer_scrollbar_y.config(command=self._on_explorer_scroll)


        self._update_explorer_view_items() # 실제 아이템 그리기/업데이트


    def _sort_treeview_column(self, column_id):  # column_name 대신 column_id (Treeview 컬럼 식별자) 사용
        """Treeview 컬럼 헤더 클릭 시 정렬 수행 (기능 제거됨)"""
        pass # 정렬 기능 제거

    def calculate_dataset_map(self):
        #전체 이미지에 대해 mAP를 계산하여 다이얼로그로 보여줌.
        import copy
        from tkinter import messagebox
        # 현재 설정된 임계값
        conf_thresh = self.conf_slider.get()
        iou_thresh = self.iou_slider.get()

        # 모든 GT, Preds 리스트로 병합
        all_gt = []
        for anns in self.gt_annotations.values():
            all_gt.extend(anns)
        all_pred = []
        for preds in self.pred_annotations_all.values():
            for p in preds:
                if p['score'] >= conf_thresh:
                    all_pred.append(copy.deepcopy(p))

        # mAP 계산 (map_calculator.calculate_map 을 사용)
        mean_ap, _ = map_calculator.calculate_map(
            all_gt, all_pred, self.categories, iou_threshold=iou_thresh
        )

        # 결과 표시
        self.dataset_map_label.config(
            text=f"Dataset mAP (IoU={iou_thresh:.2f}), (Conf={conf_thresh:.2f}): {mean_ap:.4f}"
        )
        messagebox.showinfo(
            "Dataset mAP",
            f"Dataset mAP (IoU={iou_thresh:.2f}), (Conf={conf_thresh:.2f}): {mean_ap:.4f}"
        )

    def _compute_instance_numbers(self, iou_thresh=0.5):
        """GT↔PR 박스 매칭 후, 클래스별로 동일 번호 부여"""
        self.instance_numbers.clear()
        # 클래스별 처리
        present_cats = {
            ann['category_id'] for ann in self.current_gt_anns
        } | {
            ann['category_id'] for ann in self.current_pred_anns
        }
        for cat_id in present_cats:
            # 해당 클래스만 추출
            gt_list = [(i,ann) for i,ann in enumerate(self.current_gt_anns) if ann['category_id']==cat_id]
            pr_list = [(i,ann) for i,ann in enumerate(self.current_pred_anns) if ann['category_id']==cat_id]
            matched_gt = set()
            matched_pr = set()
            counter = 0

            # IoU 함수
            def iou(a, b):
                x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
                x2 = min(a[0]+a[2], b[0]+b[2]); y2 = min(a[1]+a[3], b[1]+b[3])
                inter = max(0, x2-x1) * max(0, y2-y1)
                union = a[2]*a[3] + b[2]*b[3] - inter
                return inter/union if union>0 else 0

            # 1) GT↔PR 매칭
            for gt_idx, gt_ann in gt_list:
                best_pr = None; best_iou = 0
                for pr_idx, pr_ann in pr_list:
                    if pr_idx in matched_pr: continue
                    val = iou(gt_ann['bbox'], pr_ann['bbox'])
                    if val>best_iou:
                        best_iou, best_pr = val, pr_idx
                if best_iou >= iou_thresh:
                    counter += 1
                    self.instance_numbers[f"gt_{gt_idx}"]   = counter
                    self.instance_numbers[f"pred_{best_pr}"] = counter
                    matched_gt.add(gt_idx); matched_pr.add(best_pr)

            # 2) 매칭되지 않은 GT
            for gt_idx, _ in gt_list:
                if gt_idx in matched_gt: continue
                counter += 1
                self.instance_numbers[f"gt_{gt_idx}"] = counter

            # 3) 매칭되지 않은 PR
            for pr_idx, _ in pr_list:
                if pr_idx in matched_pr: continue
                counter += 1
                self.instance_numbers[f"pred_{pr_idx}"] = counter

    def reset_annotations(self):
        """현재 선택한 이미지에 대해, 편집 전(로드 직후) 상태로 되돌립니다."""
        if not self.current_image_id:
            return
        # predictions를 다시 로드
        self.load_annotations_for_current_image()
        # 화면 갱신
        self.update_visualization_and_map()
        self.update_status("Annotations have been reset.", 100)

if __name__ == '__main__':
    root = tk.Tk()
    app = AnnotatorGUI(root)
    root.mainloop()