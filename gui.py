# annotator_gui.py
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import os

# 다른 모듈 임포트
import coco_loader
import map_calculator
import visualizer # 시각화는 InteractiveCanvas가 직접 처리할 수도 있음
from interactive_canvas import InteractiveCanvas

class AnnotatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Annotation Tool")
        master.geometry("1200x800")

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

        # GUI 요소 생성
        self._create_widgets()

        # 초기 상태 설정
        self._update_ui_state()

    def _create_widgets(self):
        # 프레임 나누기
        top_frame = tk.Frame(self.master)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        left_frame = tk.Frame(self.master)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        center_frame = tk.Frame(self.master)
        center_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

        right_frame = tk.Frame(self.master)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # --- Top Frame: 파일 로드 버튼 ---
        btn_load_gt = tk.Button(top_frame, text="Load GT Annotations", command=self.load_gt_data)
        btn_load_gt.pack(side=tk.LEFT, padx=5)
        btn_load_pred = tk.Button(top_frame, text="Load Predictions", command=self.load_pred_data)
        btn_load_pred.pack(side=tk.LEFT, padx=5)
        btn_load_img_dir = tk.Button(top_frame, text="Select Image Directory", command=self.select_image_dir)
        btn_load_img_dir.pack(side=tk.LEFT, padx=5)

        # --- Left Frame: 이미지 목록 ---
        tk.Label(left_frame, text="Images:").pack(anchor="w")
        self.img_listbox = tk.Listbox(left_frame, width=30, height=30)
        self.img_listbox.pack(fill="both", expand=True)
        self.img_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        scrollbar = tk.Scrollbar(left_frame, orient="vertical", command=self.img_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.img_listbox.config(yscrollcommand=scrollbar.set)


        # --- Center Frame: 이미지 및 Annotation 표시 ---
        self.canvas = InteractiveCanvas(center_frame, bg="lightgrey")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.set_annotation_update_callback(self.on_annotation_update) # 콜백 설정

        # --- Right Frame: 컨트롤 및 정보 ---
        tk.Label(right_frame, text="Controls & Info").pack(anchor="w", pady=5)

        # Confidence Threshold Slider
        self.conf_frame = tk.Frame(right_frame)
        self.conf_frame.pack(fill=tk.X, pady=5)
        tk.Label(self.conf_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        self.conf_value_label = tk.Label(self.conf_frame, text="0.50")
        self.conf_value_label.pack(side=tk.RIGHT, padx=5)
        self.conf_slider = tk.Scale(self.conf_frame, from_=0.0, to=1.0, resolution=0.01,
                                    orient=tk.HORIZONTAL, command=self.on_threshold_change, showvalue=False)
        self.conf_slider.set(0.5)
        self.conf_slider.pack(fill=tk.X, expand=True)

        # IoU Threshold Slider (for mAP)
        self.iou_frame = tk.Frame(right_frame)
        self.iou_frame.pack(fill=tk.X, pady=5)
        tk.Label(self.iou_frame, text="IoU Threshold (for mAP):").pack(side=tk.LEFT)
        self.iou_value_label = tk.Label(self.iou_frame, text="0.50")
        self.iou_value_label.pack(side=tk.RIGHT, padx=5)
        self.iou_slider = tk.Scale(self.iou_frame, from_=0.05, to=0.95, resolution=0.05,
                                   orient=tk.HORIZONTAL, command=self.on_threshold_change, showvalue=False)
        self.iou_slider.set(0.5)
        self.iou_slider.pack(fill=tk.X, expand=True)

        # mAP Display
        self.map_label = tk.Label(right_frame, text="mAP: N/A", font=("Arial", 12))
        self.map_label.pack(anchor="w", pady=10)

        # Class AP Display (Scrollable Text)
        tk.Label(right_frame, text="Class APs:").pack(anchor="w")
        self.class_ap_text = tk.Text(right_frame, height=10, width=30, state=tk.DISABLED)
        self.class_ap_text.pack(fill=tk.X, pady=5)

        # Edit Label Button
        self.edit_label_button = tk.Button(right_frame, text="Edit Selected Label", command=self.edit_selected_label, state=tk.DISABLED)
        self.edit_label_button.pack(fill=tk.X, pady=5)

        # Save Annotations Button
        self.save_button = tk.Button(right_frame, text="Save Modified Annotations", command=self.save_annotations, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, pady=5)


    def _update_ui_state(self):
        """데이터 로드 상태에 따라 UI 요소 활성화/비활성화"""
        gt_loaded = self.gt_images is not None
        pred_loaded = self.pred_annotations_all is not None
        img_dir_set = bool(self.image_dir)
        img_selected = self.current_image_id is not None

        # 이미지 목록은 GT 로드 시 채워짐
        self.img_listbox.config(state=tk.NORMAL if gt_loaded else tk.DISABLED)

        # 슬라이더 및 mAP 계산은 GT, 예측, 이미지 디렉토리 모두 필요
        can_calculate_map = gt_loaded and pred_loaded and img_dir_set and img_selected
        self.conf_slider.config(state=tk.NORMAL if can_calculate_map else tk.DISABLED)
        self.iou_slider.config(state=tk.NORMAL if can_calculate_map else tk.DISABLED)

        # 레이블 수정 버튼은 annotation 선택 시 활성화 (InteractiveCanvas에서 관리 필요)
        # 여기서는 일단 이미지 선택 시 활성화
        self.edit_label_button.config(state=tk.NORMAL if img_selected and pred_loaded else tk.DISABLED)

        # 저장 버튼은 예측 로드 시 활성화
        self.save_button.config(state=tk.NORMAL if pred_loaded else tk.DISABLED)


    def load_gt_data(self):
        filepath = filedialog.askopenfilename(
            title="Select Ground Truth COCO JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath: return

        self.gt_images, self.gt_annotations, self.categories = coco_loader.load_coco_annotations(filepath)

        if self.gt_images:
            self.img_listbox.delete(0, tk.END)
            self.image_id_map = {} # Listbox index to image_id
            for i, img_id in enumerate(self.gt_images.keys()):
                filename = self.gt_images[img_id].get('file_name', f'Image ID: {img_id}')
                self.img_listbox.insert(tk.END, filename)
                self.image_id_map[i] = img_id
            messagebox.showinfo("Success", f"Loaded {len(self.gt_images)} images from GT annotations.")
        else:
            messagebox.showerror("Error", "Failed to load GT annotations.")
        self._update_ui_state()

    def load_pred_data(self):
        filepath = filedialog.askopenfilename(
            title="Select Prediction COCO JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath: return

        self.pred_annotations_all = coco_loader.load_predictions(filepath)
        if self.pred_annotations_all is not None:
             messagebox.showinfo("Success", f"Loaded predictions for {len(self.pred_annotations_all)} images.")
             # 현재 선택된 이미지가 있다면 예측 업데이트
             if self.current_image_id:
                 self.load_annotations_for_current_image()
                 self.update_visualization_and_map()
        else:
            messagebox.showerror("Error", "Failed to load predictions.")
        self._update_ui_state()


    def select_image_dir(self):
        dirpath = filedialog.askdirectory(title="Select Image Directory")
        if not dirpath: return
        self.image_dir = dirpath
        messagebox.showinfo("Success", f"Image directory set to: {self.image_dir}")
        # 현재 선택된 이미지가 있다면 경로 업데이트 및 다시 로드
        if self.current_image_id:
            self.load_image_and_annotations(self.current_image_id)
        self._update_ui_state()

    def on_image_select(self, event):
        selected_indices = self.img_listbox.curselection()
        if not selected_indices: return
        selected_index = selected_indices[0]

        if selected_index in self.image_id_map:
            self.current_image_id = self.image_id_map[selected_index]
            print(f"Selected Image ID: {self.current_image_id}")
            if not self.image_dir:
                messagebox.showwarning("Warning", "Please select the image directory first.")
                self.img_listbox.selection_clear(0, tk.END)
                self.current_image_id = None
                return
            self.load_image_and_annotations(self.current_image_id)
        self._update_ui_state()


    def load_image_and_annotations(self, image_id):
        """선택된 이미지와 해당 annotation들을 로드하고 표시합니다."""
        if not self.gt_images or image_id not in self.gt_images:
            print(f"오류: 이미지 ID {image_id}에 대한 정보를 찾을 수 없습니다.")
            return

        image_info = self.gt_images[image_id]
        self.current_image_path = coco_loader.get_image_path(image_info, self.image_dir)

        if not self.current_image_path or not os.path.exists(self.current_image_path):
            messagebox.showerror("Error", f"Image file not found: {self.current_image_path}")
            self.canvas.delete("all") # 캔버스 클리어
            self.current_image_id = None # 선택 해제
            self.img_listbox.selection_clear(0, tk.END)
            return

        # 캔버스에 이미지 로드
        self.canvas.load_image(self.current_image_path)

        # Annotation 로드
        self.load_annotations_for_current_image()

        # 시각화 및 mAP 업데이트
        self.update_visualization_and_map()


    def load_annotations_for_current_image(self):
        """현재 이미지 ID에 해당하는 GT 및 예측 annotation을 로드합니다."""
        self.current_gt_anns = self.gt_annotations.get(self.current_image_id, []) if self.gt_annotations else []

        # 예측 annotation 로드 (수정을 위해 깊은 복사 고려)
        if self.pred_annotations_all:
            # 중요: 원본 pred_annotations_all을 수정하지 않도록 복사본 사용
            import copy
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

        # InteractiveCanvas 업데이트 (예측만 전달)
        self.canvas.set_data(self.current_pred_anns, self.categories, conf_thresh)

        # mAP 계산 (현재 이미지에 대해서만)
        # 주의: COCO mAP는 전체 데이터셋에 대해 계산하는 것이 표준임
        # 여기서는 현재 이미지의 AP를 계산하여 보여주는 방식으로 단순화
        if self.current_gt_anns and self.current_pred_anns:
             # Confidence threshold로 예측 필터링
            filtered_preds = [p for p in self.current_pred_anns if p['score'] >= conf_thresh]

            mean_ap, class_aps = map_calculator.calculate_map(
                self.current_gt_anns, filtered_preds, self.categories, iou_thresh_map
            )
            self.map_label.config(text=f"Current Image AP (IoU={iou_thresh_map:.2f}): {mean_ap:.4f}")

            # 클래스별 AP 표시
            self.class_ap_text.config(state=tk.NORMAL)
            self.class_ap_text.delete(1.0, tk.END)
            if class_aps:
                for cat_id, ap in sorted(class_aps.items()):
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


    def on_threshold_change(self, value):
        """슬라이더 값 변경 시 호출됩니다."""
        # 값 레이블 업데이트
        self.conf_value_label.config(text=f"{self.conf_slider.get():.2f}")
        self.iou_value_label.config(text=f"{self.iou_slider.get():.2f}")
        # 시각화 및 mAP 업데이트
        self.update_visualization_and_map()

    def on_annotation_update(self, index, updated_annotation):
        """InteractiveCanvas에서 annotation이 수정되었을 때 호출되는 콜백"""
        print(f"GUI: Annotation {index} updated in canvas: {updated_annotation}")
        # self.current_pred_anns 리스트의 해당 인덱스 업데이트
        if 0 <= index < len(self.current_pred_anns):
            # 원본 ID 등 유지해야 할 정보가 있다면 병합 필요
            original_id = self.current_pred_anns[index].get('id')
            self.current_pred_anns[index] = updated_annotation
            if original_id is not None:
                 self.current_pred_anns[index]['id'] = original_id # ID 유지

            # 변경 후 mAP 재계산 및 표시 업데이트
            self.update_visualization_and_map()
        else:
            print(f"오류: 잘못된 annotation 인덱스 {index}")


    def edit_selected_label(self):
        """선택된 annotation의 레이블을 수정합니다."""
        selected_ann_idx = self.canvas._selected_ann_idx # InteractiveCanvas의 내부 변수 접근 (더 나은 방법 고려)

        if selected_ann_idx == -1 or not (0 <= selected_ann_idx < len(self.current_pred_anns)):
            messagebox.showwarning("Warning", "Please select an annotation on the canvas first.")
            return

        current_ann = self.current_pred_anns[selected_ann_idx]
        current_cat_id = current_ann['category_id']
        current_label = self.categories.get(current_cat_id, {}).get('name', f'ID:{current_cat_id}')

        # 카테고리 목록 생성 (이름: ID)
        category_choices = {cat['name']: cat_id for cat_id, cat in self.categories.items()}
        category_names = list(category_choices.keys())

        # 새 레이블 입력 받기 (ComboBox 또는 simpledialog 사용)
        # 여기서는 simpledialog 사용 예시
        new_label_name = simpledialog.askstring("Edit Label", f"Enter new label for '{current_label}':",
                                                parent=self.master)

        if new_label_name:
            # 입력된 이름으로 카테고리 ID 찾기
            new_cat_id = None
            if new_label_name in category_choices:
                new_cat_id = category_choices[new_label_name]
            else:
                # 새 카테리 이름 처리 (새 ID 할당 또는 오류) - 여기서는 기존 목록에서만 선택 가능하다고 가정
                messagebox.showerror("Error", f"Unknown category name: '{new_label_name}'. Please choose from existing categories.")
                return

            if new_cat_id is not None and new_cat_id != current_cat_id:
                print(f"Updating label for annotation {selected_ann_idx}: {current_label} -> {new_label_name} (ID: {new_cat_id})")
                self.current_pred_anns[selected_ann_idx]['category_id'] = new_cat_id
                # 시각화 및 mAP 업데이트
                self.update_visualization_and_map()


    def save_annotations(self):
        """수정된 예측 annotation들을 새 파일에 저장합니다."""
        if not self.pred_annotations_all:
            messagebox.showerror("Error", "No predictions loaded to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Modified Predictions",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not save_path: return

        # 현재 이미지의 수정 사항을 전체 예측 데이터(self.pred_annotations_all)에 반영
        if self.current_image_id and self.current_pred_anns:
             self.pred_annotations_all[self.current_image_id] = self.current_pred_anns

        # 전체 예측 데이터를 COCO 제출 형식 (리스트)으로 변환
        output_predictions = []
        for img_id, preds in self.pred_annotations_all.items():
            output_predictions.extend(preds)

        try:
            import json
            with open(save_path, 'w') as f:
                json.dump(output_predictions, f, indent=4)
            messagebox.showinfo("Success", f"Modified predictions saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = AnnotatorGUI(root)
    root.mainloop()