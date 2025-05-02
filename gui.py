# annotator_gui.py
import tkinter as tk
from tkinter import ttk # ttk 모듈 임포트
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import os
import copy # deepcopy 사용 위함

# 다른 모듈 임포트
import coco_loader
import map_calculator
# visualizer는 이제 InteractiveCanvas가 처리
from interactive_canvas import InteractiveCanvas

class AnnotatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Annotation Tool")
        master.geometry("1400x900") # 창 크기 조정
        master.tk.call('source', 'azure.tcl') # 테마 적용 (azure.tcl 파일 필요)
        master.tk.call("set_theme", "light") # 또는 "dark"

        # 데이터 변수
        self.gt_images = None
        self.gt_annotations = None
        self.categories = None
        self.pred_annotations_all = None # 모든 이미지에 대한 예측 {image_id: [preds]}
        self.current_image_id = None
        self.current_image_path = None
        self.current_gt_anns = []
        self.current_pred_anns = [] # 현재 이미지의 수정 가능한 예측 리스트
        self.image_dir = ""
        self.class_visibility = {} # 클래스별 가시성 상태 {cat_id: tk.BooleanVar}

        # UI 요소 생성
        self._create_widgets()

        # 초기 상태 설정
        self._update_ui_state()

    def _create_widgets(self):
        # 스타일 설정 (선택 사항)
        style = ttk.Style(self.master)

        # --- Top Frame: 파일 로드 버튼 및 상태 ---
        top_frame = ttk.Frame(self.master, padding="5 5 5 0")
        top_frame.pack(side=tk.TOP, fill=tk.X)

        btn_load_gt = ttk.Button(top_frame, text="Load GT Annotations", command=self.load_gt_data)
        btn_load_gt.pack(side=tk.LEFT, padx=5)
        btn_load_pred = ttk.Button(top_frame, text="Load Predictions", command=self.load_pred_data)
        btn_load_pred.pack(side=tk.LEFT, padx=5)
        btn_load_img_dir = ttk.Button(top_frame, text="Select Image Directory", command=self.select_image_dir)
        btn_load_img_dir.pack(side=tk.LEFT, padx=5)

        # --- Bottom Frame: 상태 표시줄 ---
        bottom_frame = ttk.Frame(self.master, padding="5 0 5 5")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(bottom_frame, text="Ready", anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.progress_bar = ttk.Progressbar(bottom_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)

        # --- Main Frame (Left, Center, Right 분할) ---
        main_frame = ttk.Frame(self.master, padding="5 5 5 5")
        main_frame.pack(fill="both", expand=True)

        # --- Left Frame: 이미지 목록 ---
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left_frame, text="Images:").pack(anchor="w")
        img_list_frame = ttk.Frame(left_frame) # 스크롤바를 위한 프레임
        img_list_frame.pack(fill="both", expand=True)
        self.img_listbox = tk.Listbox(img_list_frame, width=30, height=30, exportselection=False) # exportselection=False 추가
        self.img_listbox.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar = ttk.Scrollbar(img_list_frame, orient="vertical", command=self.img_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.img_listbox.config(yscrollcommand=scrollbar.set)
        self.img_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        # --- Right Frame: 컨트롤 및 정보 ---
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        ttk.Label(right_frame, text="Controls & Info", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 10))

        # Confidence Threshold Slider + Buttons
        self.conf_frame = ttk.LabelFrame(right_frame, text="Confidence Threshold", padding="5")
        self.conf_frame.pack(fill=tk.X, pady=5)
        self.conf_value_label = ttk.Label(self.conf_frame, text="0.50", width=5, anchor="e")
        self.conf_value_label.pack(side=tk.RIGHT, padx=(5, 0))
        btn_conf_minus = ttk.Button(self.conf_frame, text="-", width=2, command=lambda: self.adjust_slider(self.conf_slider, -0.01))
        btn_conf_minus.pack(side=tk.LEFT)
        self.conf_slider = ttk.Scale(self.conf_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                     command=self.on_threshold_change)
        self.conf_slider.set(0.5)
        self.conf_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        btn_conf_plus = ttk.Button(self.conf_frame, text="+", width=2, command=lambda: self.adjust_slider(self.conf_slider, 0.01))
        btn_conf_plus.pack(side=tk.LEFT)

        # IoU Threshold Slider (for mAP) + Buttons
        self.iou_frame = ttk.LabelFrame(right_frame, text="IoU Threshold (for mAP)", padding="5")
        self.iou_frame.pack(fill=tk.X, pady=5)
        self.iou_value_label = ttk.Label(self.iou_frame, text="0.50", width=5, anchor="e")
        self.iou_value_label.pack(side=tk.RIGHT, padx=(5, 0))
        btn_iou_minus = ttk.Button(self.iou_frame, text="-", width=2, command=lambda: self.adjust_slider(self.iou_slider, -0.05))
        btn_iou_minus.pack(side=tk.LEFT)
        self.iou_slider = ttk.Scale(self.iou_frame, from_=0.05, to=0.95, orient=tk.HORIZONTAL,
                                    command=self.on_threshold_change)
        self.iou_slider.set(0.5) # 초기값 설정
        self.iou_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        btn_iou_plus = ttk.Button(self.iou_frame, text="+", width=2, command=lambda: self.adjust_slider(self.iou_slider, 0.05))
        btn_iou_plus.pack(side=tk.LEFT)

        # mAP Display
        self.map_label = ttk.Label(right_frame, text="Current Image AP: N/A", font=("Arial", 10))
        self.map_label.pack(anchor="w", pady=5)

        # Class AP Display (Scrollable Text)
        class_ap_frame = ttk.LabelFrame(right_frame, text="Class APs", padding="5")
        class_ap_frame.pack(fill=tk.X, pady=5)
        self.class_ap_text = tk.Text(class_ap_frame, height=8, width=30, state=tk.DISABLED, relief=tk.FLAT) # 테두리 제거
        ap_scrollbar = ttk.Scrollbar(class_ap_frame, orient="vertical", command=self.class_ap_text.yview)
        self.class_ap_text.config(yscrollcommand=ap_scrollbar.set)
        ap_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.class_ap_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Class Visibility Checkboxes
        visibility_frame = ttk.LabelFrame(right_frame, text="Class Visibility", padding="5")
        visibility_frame.pack(fill="both", expand=True, pady=5)
        vis_canvas = tk.Canvas(visibility_frame, borderwidth=0, background="#ffffff")
        self.class_checkbox_frame = ttk.Frame(vis_canvas, padding="2") # 체크박스 들어갈 프레임
        vis_scrollbar = ttk.Scrollbar(visibility_frame, orient="vertical", command=vis_canvas.yview)
        vis_canvas.configure(yscrollcommand=vis_scrollbar.set)

        vis_scrollbar.pack(side="right", fill="y")
        vis_canvas.pack(side="left", fill="both", expand=True)
        vis_canvas.create_window((0,0), window=self.class_checkbox_frame, anchor="nw")

        self.class_checkbox_frame.bind("<Configure>", lambda e: vis_canvas.configure(scrollregion=vis_canvas.bbox("all")))

        # Edit Label Button
        self.edit_label_button = ttk.Button(right_frame, text="Edit Selected Label", command=self.edit_selected_label, state=tk.DISABLED)
        self.edit_label_button.pack(fill=tk.X, pady=5)

        # Save Annotations Button
        self.save_button = ttk.Button(right_frame, text="Save Modified Annotations", command=self.save_annotations, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, pady=5)

        # --- Center Frame: 이미지 및 Annotation 표시 ---
        center_frame = ttk.Frame(main_frame, padding="5")
        center_frame.pack(side=tk.LEFT, fill="both", expand=True)
        self.canvas = InteractiveCanvas(center_frame, bg="lightgrey")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.set_annotation_update_callback(self.on_annotation_update) # 콜백 설정

    def update_status(self, message, progress=None):
        """상태 레이블과 프로그레스 바를 업데이트합니다."""
        self.status_label.config(text=message)
        if progress is not None:
            self.progress_bar['value'] = progress
        self.master.update_idletasks() # UI 즉시 업데이트

    def adjust_slider(self, slider, delta):
        """슬라이더 값을 버튼으로 미세 조정합니다."""
        current_value = slider.get()
        new_value = round(current_value + delta, 2) # 소수점 2자리까지 반올림

        if slider == self.conf_slider:
            min_val, max_val = 0.0, 1.0
        elif slider == self.iou_slider:
            min_val, max_val = 0.05, 0.95
        else:
            return

        new_value = max(min_val, min(max_val, new_value)) # 범위 제한
        slider.set(new_value)

    def _update_ui_state(self):
        """데이터 로드 상태에 따라 UI 요소 활성화/비활성화"""
        gt_loaded = self.gt_images is not None and self.categories is not None
        pred_loaded = self.pred_annotations_all is not None
        img_dir_set = bool(self.image_dir)
        img_selected = self.current_image_id is not None

        self.img_listbox.config(state=tk.NORMAL if gt_loaded else tk.DISABLED)

        can_calculate_map = gt_loaded and pred_loaded and img_dir_set and img_selected
        self.conf_slider.config(state=tk.NORMAL if can_calculate_map else tk.DISABLED)
        self.iou_slider.config(state=tk.NORMAL if can_calculate_map else tk.DISABLED)

        for child in self.conf_frame.winfo_children():
            if isinstance(child, ttk.Button): child.config(state=tk.NORMAL if can_calculate_map else tk.DISABLED)
        for child in self.iou_frame.winfo_children():
             if isinstance(child, ttk.Button): child.config(state=tk.NORMAL if can_calculate_map else tk.DISABLED)

        self.edit_label_button.config(state=tk.NORMAL if img_selected and pred_loaded else tk.DISABLED)
        self.save_button.config(state=tk.NORMAL if pred_loaded else tk.DISABLED)

    def load_gt_data(self):
        filepath = filedialog.askopenfilename(
            title="Select Ground Truth COCO JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath: return

        self.update_status("Loading GT annotations...", 0)
        self.gt_images, self.gt_annotations, self.categories = coco_loader.load_coco_annotations(filepath)
        self.update_status("Processing GT data...", 50)

        if self.gt_images and self.categories:
            self.img_listbox.delete(0, tk.END)
            self.image_id_map = {} # Listbox index to image_id
            sorted_img_ids = sorted(self.gt_images.keys(), key=lambda img_id: self.gt_images[img_id].get('file_name', str(img_id)))

            for i, img_id in enumerate(sorted_img_ids):
                filename = self.gt_images[img_id].get('file_name', f'Image ID: {img_id}')
                self.img_listbox.insert(tk.END, filename)
                self.image_id_map[i] = img_id

            self._populate_class_checkboxes()

            messagebox.showinfo("Success", f"Loaded {len(self.gt_images)} images and {len(self.categories)} categories from GT.")
            self.update_status(f"GT Loaded: {len(self.gt_images)} images, {len(self.categories)} categories.", 100)
        else:
            messagebox.showerror("Error", "Failed to load GT annotations or categories.")
            self.update_status("Error loading GT.", 0)
        self._update_ui_state()

    def _populate_class_checkboxes(self):
        """카테고리 정보를 기반으로 클래스 가시성 체크박스를 생성합니다."""
        for widget in self.class_checkbox_frame.winfo_children():
            widget.destroy()
        self.class_visibility.clear()

        if not self.categories:
            return

        sorted_categories = sorted(self.categories.items(), key=lambda item: item[1]['name'])

        for cat_id, category in sorted_categories:
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(self.class_checkbox_frame, text=category['name'], variable=var,
                                 command=self.on_visibility_change)
            cb.pack(anchor="w", fill="x")
            self.class_visibility[cat_id] = var

    def on_visibility_change(self):
        """클래스 가시성 체크박스 상태 변경 시 호출됩니다."""
        self.update_visualization_and_map()

    def load_pred_data(self):
        filepath = filedialog.askopenfilename(
            title="Select Prediction COCO JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath: return

        self.update_status("Loading predictions...", 0)
        self.pred_annotations_all = coco_loader.load_predictions(filepath)
        self.update_status("Processing predictions...", 50)

        if self.pred_annotations_all is not None:
             messagebox.showinfo("Success", f"Loaded predictions for {len(self.pred_annotations_all)} images.")
             self.update_status(f"Predictions loaded for {len(self.pred_annotations_all)} images.", 100)
             if self.current_image_id:
                 self.load_annotations_for_current_image()
                 self.update_visualization_and_map()
        else:
            messagebox.showerror("Error", "Failed to load predictions.")
            self.update_status("Error loading predictions.", 0)
        self._update_ui_state()

    def select_image_dir(self):
        dirpath = filedialog.askdirectory(title="Select Image Directory")
        if not dirpath: return
        self.image_dir = dirpath
        messagebox.showinfo("Success", f"Image directory set to: {self.image_dir}")
        self.update_status(f"Image directory set: {self.image_dir}", 100)
        if self.current_image_id:
            self.load_image_and_annotations(self.current_image_id)
        self._update_ui_state()

    def on_image_select(self, event):
        selected_indices = self.img_listbox.curselection()
        if not selected_indices: return
        selected_index = selected_indices[0]

        if selected_index in self.image_id_map:
            new_image_id = self.image_id_map[selected_index]
            if new_image_id == self.current_image_id:
                return

            self.current_image_id = new_image_id
            print(f"Selected Image ID: {self.current_image_id}")
            if not self.image_dir:
                messagebox.showwarning("Warning", "Please select the image directory first.")
                self.img_listbox.selection_clear(0, tk.END)
                self.current_image_id = None
                self.update_status("Select image directory first.", 0)
                return

            self.update_status(f"Loading image ID: {self.current_image_id}...", 0)
            self.load_image_and_annotations(self.current_image_id)
            self.update_status(f"Image ID: {self.current_image_id} loaded.", 100)
        self._update_ui_state()

    def load_image_and_annotations(self, image_id):
        """선택된 이미지와 해당 annotation들을 로드하고 표시합니다."""
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
        """현재 이미지 ID에 해당하는 GT 및 예측 annotation을 로드합니다."""
        self.current_gt_anns = self.gt_annotations.get(self.current_image_id, []) if self.gt_annotations else []

        if self.pred_annotations_all:
            self.current_pred_anns = copy.deepcopy(self.pred_annotations_all.get(self.current_image_id, []))
        else:
            self.current_pred_anns = []

        print(f"Loaded {len(self.current_gt_anns)} GT annotations and {len(self.current_pred_anns)} predictions for image {self.current_image_id}")

    def update_visualization_and_map(self):
        """현재 설정을 기반으로 캔버스를 업데이트하고 mAP를 다시 계산합니다."""
        if not self.current_image_id or not self.categories:
            return

        conf_thresh = self.conf_slider.get()
        iou_thresh_map = self.iou_slider.get()

        visible_class_ids = {cat_id for cat_id, var in self.class_visibility.items() if var.get()}

        self.canvas.set_data(
            self.current_gt_anns,
            self.current_pred_anns,
            self.categories,
            conf_thresh,
            visible_class_ids
        )

        if self.current_gt_anns and self.current_pred_anns:
            filtered_preds = [p for p in self.current_pred_anns if p['score'] >= conf_thresh]

            mean_ap, class_aps = map_calculator.calculate_map(
                self.current_gt_anns, filtered_preds, self.categories, iou_thresh_map
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
            self.class_ap_text.insert(tk.END, "GT or Predictions missing.")
            self.class_ap_text.config(state=tk.DISABLED)
        self.update_status(f"Image {self.current_image_id} visualization updated.", 100)

    def on_threshold_change(self, value):
        """슬라이더 값 변경 시 호출됩니다."""
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
        """InteractiveCanvas에서 annotation(예측)이 수정되었을 때 호출되는 콜백"""
        print(f"GUI: Prediction {index} updated in canvas: {updated_annotation}")
        if 0 <= index < len(self.current_pred_anns):
            self.current_pred_anns[index] = updated_annotation

            self.update_visualization_and_map()
            self.update_status(f"Annotation {index} updated. Recalculating AP.", 50)
        else:
            print(f"Error: Invalid prediction index {index}")
            self.update_status(f"Error updating annotation {index}.", 0)

    def edit_selected_label(self):
        """선택된 annotation(예측)의 레이블을 수정합니다."""
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
        """수정된 예측 annotation들을 새 파일에 저장합니다."""
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
             self.pred_annotations_all[self.current_image_id] = self.current_pred_anns

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

if __name__ == '__main__':
    root = tk.Tk()
    app = AnnotatorGUI(root)
    root.mainloop()