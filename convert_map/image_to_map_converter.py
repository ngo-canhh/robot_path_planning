import cv2
import numpy as np
import os
import yaml
import math
import matplotlib.pyplot as plt

class ImageToMapConverter:
    def __init__(self,
                 min_area_threshold=20,
                 rectangle_area_threshold=100,
                 circle_area_threshold=150,
                 circle_vertex_threshold=8,
                 epsilon_factor=0.009,
                 rectangularity_threshold=0.92): # <-- NEW: Threshold for 'boxiness'
        """
        Khởi tạo converter với các ngưỡng

        Args:
            min_area_threshold: Ngưỡng diện tích tối thiểu để loại bỏ nhiễu
            rectangle_area_threshold: Nếu diện tích < ngưỡng này và có 4 đỉnh thì chuyển thành hình chữ nhật
            circle_area_threshold: Nếu diện tích < ngưỡng này và có nhiều đỉnh > circle_vertex_threshold thì chuyển thành hình tròn
            circle_vertex_threshold: Số đỉnh tối thiểu để cân nhắc chuyển thành hình tròn
            epsilon_factor: Hệ số để xấp xỉ đa giác
            rectangularity_threshold: Ngưỡng tỉ lệ diện tích contour / diện tích minAreaRect
                                      để coi một đa giác phức tạp là gần đúng hình chữ nhật.
        """
        self.min_area_threshold = min_area_threshold
        self.rectangle_area_threshold = rectangle_area_threshold
        self.circle_area_threshold = circle_area_threshold
        self.circle_vertex_threshold = circle_vertex_threshold
        self.epsilon_factor = epsilon_factor
        self.rectangularity_threshold = rectangularity_threshold # Store the new threshold

    def classify_shape(self, contour, approx_polygon):
        """Phân loại hình dạng dựa trên số đỉnh và các đặc điểm hình học (CHỈ DÙNG CHO GHI NHÃN)"""
        num_vertices = len(approx_polygon)
        shape_name = "Không xác định"
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if num_vertices == 3:
            shape_name = "Tam giác"
        elif num_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx_polygon)
            aspect_ratio = float(w) / h if h != 0 else 0
            # Note: This classification is primarily for visualization text now.
            # The actual shape determination for output happens later.
            shape_name = "Hình chữ nhật/Vuông (4 đỉnh)"
        elif num_vertices == 5:
             shape_name = "Ngũ giác"
        elif num_vertices > 5:
            if perimeter == 0:
                 circularity = 0
            else:
                circularity = 4 * math.pi * (area / (perimeter * perimeter))
            if circularity > 0.85: # High circularity suggests circle
                shape_name = "Hình tròn (ước lượng)"
            else:
                 # Check rectangularity using minAreaRect for labeling purposes if desired
                try:
                    rect = cv2.minAreaRect(contour)
                    (w_rect, h_rect) = rect[1]
                    rect_area = w_rect * h_rect
                    if rect_area > 0 and (area / rect_area) > self.rectangularity_threshold:
                         shape_name = f"Đa giác (gần chữ nhật, {num_vertices} đỉnh)"
                    else:
                         shape_name = f"Đa giác ({num_vertices} đỉnh)"
                except: # Handle potential errors in minAreaRect
                     shape_name = f"Đa giác ({num_vertices} đỉnh)"

        else: # < 3 vertices
            shape_name = "Đường thẳng/Nhiễu"

        # Add area info for context
        # shape_name += f" (A:{int(area)})" # Optional: add area to label
        return shape_name

    def process_image(self, img_path, output_path=None, visualize=False, flip_image=True, resize_img_size=None):
        """
        Xử lý hình ảnh và tạo ra file config

        Args:
            img_path: Đường dẫn đến ảnh cần xử lý
            output_path: Đường dẫn để lưu file config YAML
            visualize: Có hiển thị kết quả trực quan không
            flip_image: Có lật ảnh theo chiều dọc không

        Returns:
            config: Dictionary chứa thông tin config
            img_result: Ảnh kết quả đã được vẽ lên
            object_data: Thông tin về các đối tượng được phát hiện
        """
        # --- 1. Đọc ảnh ---
        img_original = cv2.imread(img_path)
        if img_original is None:
            print(f"Lỗi: Không thể đọc ảnh từ {img_path}")
            return None, None, None

        if flip_image:
            img_original = cv2.flip(img_original, 0)

        if resize_img_size:
            img_original = cv2.resize(img_original, resize_img_size, interpolation=cv2.INTER_AREA)

        img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

        # --- 2. Ngưỡng hóa (Nghịch đảo) ---
        ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # --- 3. Xử lý Viền ---
        h, w = thresh.shape[:2]
        border_thickness = 3
        thresh[0:border_thickness, :] = 0
        thresh[h-border_thickness:h, :] = 0
        thresh[:, 0:border_thickness] = 0
        thresh[:, w-border_thickness:w] = 0

        # --- 4. Tìm Contours ---
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Tìm thấy {len(contours)} đối tượng tiềm năng.")

        img_result = img_original.copy()
        object_data = []
        obstacles_data = []

        # --- 5. Lặp qua từng Contour ---
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.min_area_threshold:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            perimeter = cv2.arcLength(contour, True)
            epsilon = self.epsilon_factor * perimeter
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            num_vertices = len(approx_polygon)

            obs = {'x': cX, 'y': cY, 'dynamic_flag': 0, 'vel_x': 0.0, 'vel_y': 0.0}
            shape_name_for_label = "Không xác định" # For visualization text
            draw_contour = approx_polygon # Default contour to draw

            # --- Shape Determination Logic ---

            # A. Small shapes check (Rectangle or Circle conversion)
            is_small_special_case = False
            if num_vertices == 4 and area < self.rectangle_area_threshold:
                try:
                    rect = cv2.minAreaRect(contour) # Use minAreaRect for small quads too
                    (cx_rect, cy_rect), (w_rect, h_rect), angle_rect = rect
                    if w_rect < h_rect: # Ensure width >= height
                        w_rect, h_rect = h_rect, w_rect
                        angle_rect += 90
                    angle_rect = angle_rect % 180 # Normalize angle
                    # convert angle to radians
                    angle_rect = math.radians(angle_rect)
                    obs['shape_type'] = 1 # Rectangle
                    obs['shape_params'] = [int(w_rect), int(h_rect), float(angle_rect)]
                    shape_name_for_label = f"HCN nhỏ (cđ, {int(w_rect)}x{int(h_rect)})"
                    box = cv2.boxPoints(rect)
                    draw_contour = np.intp(box) # Use intp or int0
                    is_small_special_case = True
                except Exception as e:
                    print(f"Lỗi xử lý HCN nhỏ (contour {i}): {e}. Dùng đa giác.")
                    # Fallback handled below

            elif num_vertices > self.circle_vertex_threshold and area < self.circle_area_threshold:
                try:
                    radius = np.sqrt(area / np.pi)
                    obs['shape_type'] = 0 # Circle
                    obs['shape_params'] = [int(radius)]
                    shape_name_for_label = f"Tròn nhỏ (cđ, r={int(radius)})"
                    # For drawing, we just use the center and radius later
                    draw_contour = None # Signal not to draw polygon
                    cv2.circle(img_result, (cX, cY), int(radius), (0, 255, 255), 2) # Cyan circle
                    is_small_special_case = True
                except Exception as e:
                    print(f"Lỗi xử lý Tròn nhỏ (contour {i}): {e}. Dùng đa giác.")
                    # Fallback handled below

            # B. If not a small special case, check for large "nearly rectangular" shapes
            if not is_small_special_case:
                is_nearly_rectangular = False
                try:
                    # Calculate minAreaRect for the original contour
                    rect = cv2.minAreaRect(contour)
                    (cx_rect, cy_rect), (w_rect, h_rect), angle_rect = rect
                    rect_area = w_rect * h_rect

                    if rect_area > 0:
                        fill_ratio = area / rect_area
                        # Check if it fills the bounding box well enough
                        if fill_ratio > self.rectangularity_threshold:
                            if w_rect < h_rect: # Ensure width >= height
                                w_rect, h_rect = h_rect, w_rect
                                angle_rect += 90
                            angle_rect = angle_rect % 180 # Normalize angle
                            # Convert angle to radians
                            angle_rect = math.radians(angle_rect)

                            obs['shape_type'] = 1 # Rectangle
                            obs['shape_params'] = [int(w_rect), int(h_rect), float(angle_rect)]
                            shape_name_for_label = f"HCN lớn (gđ, {int(w_rect)}x{int(h_rect)})"
                            box = cv2.boxPoints(rect)
                            draw_contour = np.intp(box) # Draw the bounding box
                            is_nearly_rectangular = True
                except Exception as e:
                    print(f"Lỗi khi kiểm tra tính chữ nhật (contour {i}): {e}. Dùng đa giác.")
                    # Fallback handled below

                # C. If not converted to rectangle above, treat as Polygon
                if not is_nearly_rectangular:
                    try:
                        obs['shape_type'] = 3 # Polygon
                        # Use relative coordinates from approxPolyDP result
                        relative_shape = approx_polygon - np.array([[cX, cY]])
                        obs['shape_params'] = relative_shape.flatten().tolist()
                        # Get a descriptive name using the original classify_shape
                        shape_name_for_label = self.classify_shape(contour, approx_polygon)
                        draw_contour = approx_polygon # Draw the approximated polygon
                    except Exception as e:
                         print(f"Lỗi khi xử lý đa giác (contour {i}): {e}")
                         obs['shape_type'] = 3
                         obs['shape_params'] = [] # Safe fallback
                         shape_name_for_label = "Đa giác (lỗi)"
                         draw_contour = approx_polygon # Attempt to draw original approx
            obs['bounding_box'] = None # Using for DynamicObstacle, not here
            
            # --- Visualization ---
            if draw_contour is not None and len(draw_contour) > 0:
                 # Draw the determined shape boundary (blue for polygons, yellow for rectangles)
                 color = (0, 255, 255) if obs.get('shape_type') == 1 else (255, 0, 0)
                 cv2.drawContours(img_result, [draw_contour], 0, color, 2)

            # Draw center and label
            cv2.circle(img_result, (cX, cY), 3, (0, 0, 255), -1) # Red center
            cv2.putText(img_result, shape_name_for_label, (cX - 30, cY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1) # Red text

            obstacles_data.append(obs)
            object_data.append({'id': i, 'center': (cX, cY), 'shape': shape_name_for_label, 'area': area, 'type': obs.get('shape_type', -1)})


        # --- Final Output ---
        config = {
            'environment': {
                'width': img_original.shape[1],
                'height': img_original.shape[0],
            },
            'obstacles': obstacles_data,
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as file:
                yaml.dump(config, file, sort_keys=False) # Ensure params on one line
            print(f"Đã lưu config vào {output_path}")

        if visualize:
            plt.figure(figsize=(10, 10)) # Adjusted size for single plot
            plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
            plt.title('Kết quả: Tâm và Hình dạng (Ước lượng HCN)')
            plt.tight_layout()
            plt.show()

        return config, img_result, object_data

    def batch_process(self, input_dir, output_dir, visualize=False, flip_image=True, resize_img_size=None):
        """
        Xử lý hàng loạt ảnh trong thư mục (Code không đổi)
        """
        os.makedirs(output_dir, exist_ok=True)
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]

        if not image_files:
            print(f"Không tìm thấy file ảnh nào trong {input_dir}")
            return
        print(f"Tìm thấy {len(image_files)} ảnh cần xử lý.")

        for img_file in image_files:
            img_path = os.path.join(input_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            output_path = os.path.join(output_dir, f"{base_name}.yaml")

            print(f"Đang xử lý: {img_file}")
            try:
                self.process_image(img_path, output_path, visualize=False, flip_image=flip_image, resize_img_size=resize_img_size) # Visualize one by one if needed
            except Exception as e:
                 print(f"LỖI NGHIÊM TRỌNG khi xử lý {img_file}: {e}") # Catch errors per image

        print(f"Đã hoàn thành xử lý {len(image_files)} ảnh.")
        # If visualize=True was passed to batch_process, show results *after* processing all
        if visualize:
             print("Visualizing results is better done per image or by checking output files.")


# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    converter = ImageToMapConverter(
        min_area_threshold=5,
        rectangle_area_threshold=60,  # Ngưỡng cho HCN/Tròn nhỏ
        circle_area_threshold=60,
        circle_vertex_threshold=3,
        epsilon_factor=0.009,         # Giảm nhẹ epsilon có thể giữ lại nhiều chi tiết hơn
        rectangularity_threshold=0.9 # <<-- TUNE THIS VALUE (0.9 to 0.98 typical)
                                      #      Higher = stricter rectangle requirement
    )

    # # Đường dẫn đến ảnh của bạn
    # img_path = 'convert_map/maps/Boston_2_512.png' # <<-- THAY ĐỔI ĐƯỜNG DẪN NÀY
    # # Kiểm tra xem file có tồn tại không
    # if not os.path.exists(img_path):
    #      print(f"Lỗi: Không tìm thấy file ảnh tại: {img_path}")
    #      # Hoặc đặt một đường dẫn mặc định nếu muốn
    #      # img_path = 'convert_map/maps/Boston_2_512.png' # Ví dụ fallback
    #      # if not os.path.exists(img_path):
    #      #     print("Lỗi: Không tìm thấy cả file ảnh của bạn và file mặc định.")
    #      #     exit() # Thoát nếu không có ảnh nào

    # else:
    #     base_name = os.path.splitext(os.path.basename(img_path))[0]
    #     # Tạo thư mục output nếu chưa có
    #     output_dir = 'convert_map/config_map' # Thư mục lưu YAML
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_path = os.path.join(output_dir, f'{base_name}.yaml')

    #     try:
    #         print(f"Đang xử lý ảnh: {img_path}")
    #         print(f"Sẽ lưu config vào: {output_path}")
    #         config, img_result, object_data = converter.process_image(
    #             img_path=img_path,
    #             output_path=output_path,
    #             visualize=True,        # Hiển thị kết quả
    #             flip_image=False       # <<-- Đặt True nếu ảnh của bạn cần lật dọc
    #         )
    #         # Tùy chọn: In thông tin đối tượng ra console
    #         # print("\n--- Detected Objects ---")
    #         # for obj in object_data:
    #         #     print(f"ID: {obj['id']}, Center: {obj['center']}, Type: {obj['type']}, Label: {obj['shape']}, Area: {int(obj['area'])}")

    #     except FileNotFoundError:
    #         print(f"Lỗi: File ảnh không tồn tại tại '{img_path}'")
    #     except Exception as e:
    #         print(f"Xảy ra lỗi không mong muốn khi xử lý ảnh: {e}")
    #         import traceback
    #         traceback.print_exc() # In chi tiết lỗi để debug


    # Hoặc xử lý hàng loạt
    print("\n--- BATCH PROCESSING ---")
    converter.batch_process(
        input_dir='convert_map/maps',    # <<-- THAY ĐỔI THƯ MỤC NÀY
        output_dir='convert_map/config_map',        # <<-- Thư mục lưu kết quả
        visualize=False,                   # Không hiển thị từng ảnh khi chạy batch
        flip_image=True,                   # <<-- Đặt True nếu cần lật ảnh
        resize_img_size=(100, 100)         # Kích thước ảnh đầu ra (nếu cần)
    )