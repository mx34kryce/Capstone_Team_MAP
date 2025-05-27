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
        self.current_pr_prec = None
        self.current_pr_rec = None
        self.selected_pr_class_id = None
        self._copy = copy


        # 이미지 메타데이터 및 정렬 관련 변수
        self.image_metadata = {}  # 이미지 ID를 키로, 메타데이터 딕셔너리를 값으로 가짐
        self.sort_column = "filename"  # 초기 정렬 컬럼 (Treeview 컬럼 ID 기준)
        self.sort_order_asc = True  # True: 오름차순, False: 내림차순

        # UI 요소 생성
        self._create_widgets()

        # 초기 상태 설정
        self._update_ui_state()

    def _create_widgets(self):
        style = ttk.Style(self.master)

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

        btn_reset = ttk.Button(top_frame, text="Reset Annotations", command=self.reset_annotations, state=tk.DISABLED)
        btn_reset.pack(side=tk.LEFT, padx=5)
        self.reset_btn = btn_reset

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

        # --- Left Frame: 이미지 목록 및 클래스 가시성 ---
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.grid(row=0, column=0, sticky="ns")

        # 이미지 목록 (Treeview로 변경)
        ttk.Label(left_frame, text="Images:").pack(anchor="w")
        img_tree_frame = ttk.Frame(left_frame)
        img_tree_frame.pack(fill="both", expand=True, pady=(0, 10))  # expand=True로 변경

        self.image_treeview = ttk.Treeview(img_tree_frame,
                                           columns=("filename", "ap", "classes", "instances"),
                                           show="headings", selectmode=tk.BROWSE, height=15)  # exportselection -> selectmode, 값은 tk.BROWSE 또는 tk.EXTENDED 등
        self.image_treeview.heading("filename", text="File Name", command=lambda: self._sort_treeview_column("filename"))
        self.image_treeview.heading("ap", text="AP Score", command=lambda: self._sort_treeview_column("ap"))
        self.image_treeview.heading("classes", text="Classes", command=lambda: self._sort_treeview_column("classes"))
        self.image_treeview.heading("instances", text="Instances", command=lambda: self._sort_treeview_column("instances"))

        self.image_treeview.column("filename", width=150, anchor="w")
        self.image_treeview.column("ap", width=70, anchor="e")
        self.image_treeview.column("classes", width=60, anchor="e")
        self.image_treeview.column("instances", width=70, anchor="e")

        tree_scrollbar_y = ttk.Scrollbar(img_tree_frame, orient="vertical", command=self.image_treeview.yview)
        tree_scrollbar_x = ttk.Scrollbar(img_tree_frame, orient="horizontal", command=self.image_treeview.xview)
        self.image_treeview.configure(yscrollcommand=tree_scrollbar_y.set, xscrollcommand=tree_scrollbar_x.set)

        tree_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.image_treeview.pack(side=tk.LEFT, fill="both", expand=True)

        self.image_treeview.bind('<<TreeviewSelect>>', self.on_image_select)

        # 클래스 가시성
        visibility_frame = ttk.LabelFrame(left_frame, text="Class Visibility", padding="5")
        visibility_frame.pack(fill="both", expand=True, pady=5)

        # 현재 테마의 TFrame 배경색을 가져옵니다.
        frame_bg_color = style.lookup('TFrame', 'background')

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

        ttk.Label(right_frame, text="Controls & Info", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 10))

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

        # mAP Display
        self.map_label = ttk.Label(right_frame, text="Current Image AP: N/A", font=("Arial", 10))
        self.map_label.pack(anchor="w", pady=5)

        # --- Dataset mAP Label ---
        self.dataset_map_label = ttk.Label(right_frame, text="Dataset mAP: N/A", font=("Arial", 10))
        self.dataset_map_label.pack(anchor="w", pady=(2, 10))

        # Class AP Display
        class_ap_frame = ttk.LabelFrame(right_frame, text="Class APs", padding="5")
        class_ap_frame.pack(fill=tk.X, pady=5)
        self.class_ap_text = tk.Text(class_ap_frame, height=8, width=30, state=tk.DISABLED, relief=tk.FLAT)
        ap_scrollbar = ttk.Scrollbar(class_ap_frame, orient="vertical", command=self.class_ap_text.yview)
        self.class_ap_text.config(yscrollcommand=ap_scrollbar.set)
        ap_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.class_ap_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

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
        selected_items = self.image_treeview.selection()
        img_selected = bool(selected_items) and self.current_image_id is not None

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
                self._calculate_all_images_metadata()
            self._populate_image_treeview()

            self._populate_visibility_checkboxes()
            self._update_pr_class_selector()

            messagebox.showinfo("Success", f"Loaded {len(self.gt_images)} images and {len(self.categories)} categories from GT.")
            self.update_status(f"GT Loaded: {len(self.gt_images)} images, {len(self.categories)} categories.", 100)
        else:
            messagebox.showerror("Error", "Failed to load GT annotations or categories.")
            self.update_status("Error loading GT.", 0)
        self._update_ui_state()

    def _populate_visibility_checkboxes(self):
        """현재 이미지의 클래스별 · 인스턴스별 체크박스 재구성"""
        # 기존 위젯 전부 삭제
        for w in self.class_checkbox_frame.winfo_children():
            w.destroy()
        self.class_visibility.clear()
        self.instance_visibility.clear()

        if not self.current_gt_anns and not self.current_pred_anns:
            return

        # 이미지에 등장하는 클래스 ID 집합
        present_cats = {
            ann['category_id'] for ann in self.current_gt_anns
        } | {
            ann['category_id'] for ann in self.current_pred_anns
        }
        # 해당 클래스만 이름 순으로 정렬
        sorted_categories = sorted(
            ( (cid, info) for cid, info in self.categories.items()
              if cid in present_cats ),
            key=lambda item: item[1]['name']
        )

        for cat_id, cat_info in sorted_categories:
            class_name = cat_info['name']
            # — 클래스 전체 토글
            class_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                self.class_checkbox_frame,
                text=class_name,
                variable=class_var,
                command=self.on_visibility_change
            ).pack(anchor="w", fill="x")
            self.class_visibility[cat_id] = class_var

            # — GT 인스턴스 토글
            gt_list = []
            for idx, ann in enumerate(self.current_gt_anns):
                if ann['category_id'] != cat_id: continue
                key = f"gt_{idx}"
                num = self.instance_numbers.get(key, 0)
                gt_list.append((num, key))
            gt_list.sort(key=lambda x: x[0])
            
            for num, key in gt_list:
                label = f"GT_{class_name}_{num}"
                iv = tk.BooleanVar(value=True)
                ttk.Checkbutton(
                    self.class_checkbox_frame,
                    text=label,
                    variable=iv,
                    command=self.on_visibility_change
                ).pack(anchor="w", padx=20, fill="x")
                self.instance_visibility[key] = iv

            # — Prediction 인스턴스 토글
            pr_list = []
            for idx, ann in enumerate(self.current_pred_anns):
                if ann['category_id'] != cat_id: continue
                key = f"pred_{idx}"
                num = self.instance_numbers.get(key, 0)
                pr_list.append((num, key))
            pr_list.sort(key=lambda x: x[0])

            for num, key in pr_list:
                label = f"PR_{class_name}_{num}"
                iv = tk.BooleanVar(value=True)
                ttk.Checkbutton(
                    self.class_checkbox_frame,
                    text=label,
                    variable=iv,
                    command=self.on_visibility_change
                ).pack(anchor="w", padx=20, fill="x")
                self.instance_visibility[key] = iv

    def _update_pr_class_selector(self):
        self.pr_class_combobox.set('')
        if not self.categories:
            self.pr_class_combobox['values'] = []
            return

        class_names = ["Overall"] + sorted([cat['name'] for cat_id, cat in self.categories.items()])
        self.pr_class_combobox['values'] = class_names
        self.pr_class_combobox.set("Overall")
        self.selected_pr_class_id = "Overall"

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
                self._calculate_all_images_metadata()
                self._populate_image_treeview()

            if self.current_image_id:
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
        self.image_dir = dirpath
        messagebox.showinfo("Success", f"Image directory set to: {self.image_dir}")
        self.update_status(f"Image directory set: {self.image_dir}", 100)
        if self.current_image_id:
            self.load_image_and_annotations(self.current_image_id)
        self._update_ui_state()

    def on_image_select(self, event):
        selected_items = self.image_treeview.selection()
        if not selected_items:
            return
        selected_item = selected_items[0]

        try:
            new_image_id = int(selected_item)  # Treeview 아이템 ID가 이미지 ID라고 가정
        except ValueError:
            item_values = self.image_treeview.item(selected_item, 'values')
            if not item_values:
                return
            filename_to_find = item_values[0]
            found_id = None
            for img_id, img_info in self.gt_images.items():
                if img_info.get('file_name') == filename_to_find:
                    found_id = img_id
                    break
            if found_id is None:
                print(f"Error: Could not find image ID for filename {filename_to_find}")
                return
            new_image_id = found_id

        if new_image_id == self.current_image_id:
            return

        self.current_image_id = new_image_id
        print(f"Selected Image ID: {self.current_image_id}")
        if not self.image_dir:
            messagebox.showwarning("Warning", "Please select the image directory first.")
            if selected_item:
                self.image_treeview.selection_remove(selected_item)
            self.current_image_id = None
            self.update_status("Select image directory first.", 0)
            return

        self.update_status(f"Loading image ID: {self.current_image_id}...", 0)
        self.load_image_and_annotations(self.current_image_id)
        self.update_status(f"Image ID: {self.current_image_id} loaded.", 100)
        # 선택된 이미지에 등장하는 클래스만 필터링하여 체크박스 업데이트
        gt_ids   = {ann['category_id'] for ann in self.current_gt_anns}
        pred_ids = {ann['category_id'] for ann in self.current_pred_anns}
        image_class_ids = gt_ids.union(pred_ids)
        
        self._compute_instance_numbers()
        self._populate_visibility_checkboxes()
        self._update_pr_class_selector()
        self.update_visualization_and_map()
        self._update_ui_state()

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
        else:
            self.map_label.config(text=f"Current Image AP (IoU={iou_thresh_map:.2f}): N/A")
            self.class_ap_text.config(state=tk.NORMAL)
            self.class_ap_text.delete(1.0, tk.END)
            self.class_ap_text.insert(tk.END, "GT or Predictions missing for AP.")
            self.class_ap_text.config(state=tk.DISABLED)

        self.draw_pr_curve(iou_thresh_pr)

        self.update_status(f"Image {self.current_image_id} visualization updated.", 100)

    def on_threshold_change(self, value):
        try:
            conf_val = float(value) if self.conf_slider.cget("state") != tk.DISABLED else self.conf_slider.get()
            iou_val = float(value) if self.iou_slider.cget("state") != tk.DISABLED else self.iou_slider.get()

            self.conf_value_label.config(text=f"{self.conf_slider.get():.2f}")
            self.iou_value_label.config(text=f"{self.iou_slider.get():.2f}")

            if self.current_image_id:
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

        prec, rec, num_gt = map_calculator.get_pr_arrays(
            self.current_gt_anns,
            self.current_pred_anns,
            category_id=calc_cat_id,
            iou_threshold=iou_threshold
        )

        self.pr_ax.clear()
        if prec is not None and rec is not None:
            self.pr_ax.plot(rec, prec, marker='.', linestyle='-')

            if calc_cat_id is not None:
                ap = map_calculator.calculate_ap(rec, prec)
                title = f"PR Curve: {self.categories[calc_cat_id]['name']} (AP={ap:.3f})"
            else:
                title = f"PR Curve: Overall (IoU={iou_threshold:.2f})"

            self.pr_ax.set_title(title, fontsize=9)
        else:
            self.pr_ax.set_title(f"PR Curve: {self.pr_class_var.get()} (No Data)", fontsize=9)

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
        """모든 이미지에 대한 메타데이터를 계산하여 self.image_metadata에 저장합니다."""
        self.image_metadata.clear()
        if not self.gt_images or not self.pred_annotations_all or not self.categories:
            if self.gt_images:
                for img_id in self.gt_images.keys():
                    gt_anns_img = self.gt_annotations.get(img_id, [])
                    self.image_metadata[img_id] = {
                        "filename": self.gt_images[img_id].get('file_name', f'Image ID: {img_id}'),
                        "ap": "N/A",
                        "classes": len(set(ann['category_id'] for ann in gt_anns_img)),
                        "instances": len(gt_anns_img)
                    }
            return

        self.update_status("Calculating metadata for all images...", 0)
        total_images = len(self.gt_images)
        for i, img_id in enumerate(self.gt_images.keys()):
            metadata = self._calculate_image_metadata(img_id)
            if metadata:
                self.image_metadata[img_id] = metadata
            if (i + 1) % 10 == 0 or (i + 1) == total_images:
                self.update_status(f"Calculating metadata... {i+1}/{total_images}", int((i+1)/total_images*100))
        self.update_status("Metadata calculation complete.", 100)

    def _populate_image_treeview(self):
        """self.image_metadata를 사용하여 Treeview를 채웁니다."""
        for item in self.image_treeview.get_children():
            self.image_treeview.delete(item)

        if not self.image_metadata and self.gt_images:
            temp_metadata = {}
            for img_id, img_info in self.gt_images.items():
                gt_anns_img = self.gt_annotations.get(img_id, [])
                temp_metadata[img_id] = {
                    "filename": img_info.get('file_name', f'Image ID: {img_id}'),
                    "ap": "N/A",
                    "classes": len(set(ann['category_id'] for ann in gt_anns_img)),
                    "instances": len(gt_anns_img)
                }
            data_to_display = list(temp_metadata.items())
        elif not self.image_metadata:
            return
        else:
            data_to_display = list(self.image_metadata.items())

        key_map = {"File Name": "filename", "AP Score": "ap", "Classes": "classes", "Instances": "instances"}
        sort_key_actual = key_map.get(self.sort_column, "filename")

        def get_sort_value(item):
            value = item[1].get(sort_key_actual)
            if value == "N/A":
                return -float('inf') if self.sort_order_asc else float('inf')
            if isinstance(value, (int, float)):
                return value
            return str(value).lower()

        try:
            sorted_data = sorted(data_to_display, key=get_sort_value, reverse=not self.sort_order_asc)
        except TypeError as e:
            print(f"Sorting error: {e}. Data may contain mixed types for column {self.sort_column}.")
            sorted_data = data_to_display

        for img_id, meta in sorted_data:
            ap_display = f"{meta['ap']:.4f}" if isinstance(meta['ap'], float) else meta['ap']
            self.image_treeview.insert("", tk.END, id=str(img_id),
                                       values=(meta["filename"], ap_display, meta["classes"], meta["instances"]))

    def _sort_treeview_column(self, column_id):  # column_name 대신 column_id (Treeview 컬럼 식별자) 사용
        """Treeview 컬럼 헤더 클릭 시 정렬 수행"""
        if self.sort_column == column_id:
            self.sort_order_asc = not self.sort_order_asc
        else:
            self.sort_column = column_id
            self.sort_order_asc = True  # 기본 오름차순

        # 헤더에 정렬 방향 표시
        for col_id_iter in self.image_treeview["columns"]:
            current_text = self.image_treeview.heading(col_id_iter, "text")
            # 기존 화살표 제거 (정규식 사용이 더 안전할 수 있음)
            current_text = current_text.replace(' ▲', '').replace(' ▼', '')
            if col_id_iter == self.sort_column:
                arrow = ' ▲' if self.sort_order_asc else ' ▼'
                self.image_treeview.heading(col_id_iter, text=current_text + arrow)
            else:
                self.image_treeview.heading(col_id_iter, text=current_text)

        self._populate_image_treeview()

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