#!/usr/bin/env python3
"""
Script để load và lưu hình ảnh của tất cả các map được sử dụng khi chọn scenario 'all'.
Hình ảnh sẽ được lưu với tên có ý nghĩa và tổ chức theo folder.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Thêm path của thư mục algorithms vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_env.indoor_robot_env import IndoorRobotEnv
from components.maps import maps
from components.maps import get_empty_map

def save_map_images():
    """Lưu hình ảnh của tất cả các map trong scenario 'all'"""
    
    # Tạo folder chính để lưu hình ảnh
    output_dir = "map_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Các loại map types khi chọn scenario 'all'
    map_types = ["maze", "dense", "room", "trap", "real"]
    
    # Tham số chung
    start_pos = (64, 500)  # Vị trí bắt đầu như trong main.py
    goal_pos = (470, 180)  # Vị trí đích như trong main.py
    
    print("=== BẮT ĐẦU LƯU HÌNH ẢNH CÁC MAP ===")
    
    # Lặp qua từng loại map
    for map_type in map_types:
        print(f"\n--- Xử lý map type: {map_type} ---")
        
        # Tạo folder cho từng loại map
        map_type_dir = os.path.join(output_dir, map_type)
        os.makedirs(map_type_dir, exist_ok=True)
        
        # Tìm tất cả map của loại này
        map_names = [key for key in maps.keys() if key.startswith(map_type)]
        
        # Sắp xếp theo số thứ tự
        map_names.sort(key=lambda x: int(x[len(map_type):]) if x[len(map_type):].isdigit() else 0)
        
        print(f"Tìm thấy {len(map_names)} map: {map_names}")
        
        # Xử lý từng map
        for map_name in map_names:
            print(f"  Đang xử lý map: {map_name}")
            
            try:
                # Tạo environment với human mode để có thể hiển thị
                env = IndoorRobotEnv(
                    robot_radius=3,
                    max_steps=500,
                    sensor_range=50,
                    render_mode='human',  # Dùng human mode để tạo figure
                    config_path=get_empty_map(530),
                    metrics_callback=None,
                )
                
                # Thêm obstacles từ map
                env.add_obstacles(maps[map_name])
                
                # Set vị trí start và goal
                env.start = start_pos
                env.goal = goal_pos
                
                # Reset environment để initialize
                observation, info = env.reset(seed=88)  # Dùng seed cố định để nhất quán
                
                # Render để tạo visualization
                env.render()
                
                # Lưu figure hiện tại
                if env.fig is not None:
                    # Đường dẫn lưu file
                    image_path = os.path.join(map_type_dir, f"{map_name}.png")
                    
                    # Lưu với DPI cao
                    env.fig.savefig(image_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
                    
                    print(f"    ✓ Đã lưu: {image_path}")
                else:
                    print(f"    ✗ Figure is None cho map: {map_name}")
                
                # Đóng environment
                env.close()
                
            except Exception as e:
                print(f"    ✗ Lỗi khi xử lý map {map_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n=== HOÀN THÀNH ===")
    print(f"Tất cả hình ảnh đã được lưu trong folder: {output_dir}")
    print("Cấu trúc folder:")
    for map_type in map_types:
        map_type_dir = os.path.join(output_dir, map_type)
        if os.path.exists(map_type_dir):
            files = [f for f in os.listdir(map_type_dir) if f.endswith('.png')]
            print(f"  {map_type}/: {len(files)} hình ảnh")


def save_single_map_image(map_name):
    """Test function để lưu hình ảnh một map"""
    print(f"=== TEST: Lưu hình ảnh map {map_name} ===")
    
    try:
        # Tạo environment với human mode để có thể hiển thị
        env = IndoorRobotEnv(
            robot_radius=3,
            max_steps=500,
            sensor_range=50,
            render_mode='human',
            config_path=get_empty_map(530),
            metrics_callback=None,
        )
        
        # Thêm obstacles từ map
        env.add_obstacles(maps[map_name])
        
        # Set vị trí start và goal
        env.start = (64, 500)
        env.goal = (470, 180)
        
        # Reset environment
        observation, info = env.reset(seed=88)
        
        # Render để tạo visualization
        env.render()
        
        # Lưu figure hiện tại
        if env.fig is not None:
            output_path = f"test_{map_name}.png"
            env.fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Đã lưu test image: {output_path}")
        else:
            print("✗ Figure is None")
        
        # Đóng environment
        env.close()
        
    except Exception as e:
        print(f"✗ Lỗi: {e}")
        import traceback
        traceback.print_exc()


def create_summary_grid():
    """Tạo grid tổng hợp tất cả các map trong một hình ảnh"""
    print("\n=== TẠO GRID TỔNG HỢP ===")
    
    output_dir = "map_images"
    map_types = ["maze", "dense", "room", "trap", "real"]
    
    # Tính số hàng và cột cho grid
    total_maps = sum(len([key for key in maps.keys() if key.startswith(map_type)]) for map_type in map_types)
    cols = min(6, total_maps)  # Tối đa 6 cột
    rows = (total_maps + cols - 1) // cols  # Tính số hàng cần thiết
    
    print(f"Tạo grid {rows}x{cols} cho {total_maps} maps")
    
    # Tạo figure lớn
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
    fig.suptitle('Tổng hợp tất cả Maps trong Scenario "All"', fontsize=20, fontweight='bold')
    
    # Flatten axes nếu cần
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Duyệt qua từng map type và map
    for map_type in map_types:
        map_names = [key for key in maps.keys() if key.startswith(map_type)]
        map_names.sort(key=lambda x: int(x[len(map_type):]) if x[len(map_type):].isdigit() else 0)
        
        for map_name in map_names:
            if plot_idx >= len(axes):
                break
                
            image_path = os.path.join(output_dir, map_type, f"{map_name}.png")
            
            try:
                if os.path.exists(image_path):
                    # Load và hiển thị hình ảnh
                    img = plt.imread(image_path)
                    axes[plot_idx].imshow(img)
                    axes[plot_idx].set_title(f'{map_name}', fontsize=12, fontweight='bold')
                    axes[plot_idx].axis('off')
                else:
                    # Nếu không có hình ảnh, tạo placeholder
                    axes[plot_idx].text(0.5, 0.5, f'No image\n{map_name}', 
                                      ha='center', va='center', transform=axes[plot_idx].transAxes)
                    axes[plot_idx].set_title(f'{map_name}', fontsize=12)
                    axes[plot_idx].axis('off')
                    
            except Exception as e:
                print(f"Lỗi khi load hình ảnh {map_name}: {e}")
                axes[plot_idx].text(0.5, 0.5, f'Error\n{map_name}', 
                                  ha='center', va='center', transform=axes[plot_idx].transAxes)
                axes[plot_idx].set_title(f'{map_name}', fontsize=12)
                axes[plot_idx].axis('off')
            
            plot_idx += 1
    
    # Ẩn các subplot không sử dụng
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    # Lưu grid tổng hợp
    grid_path = os.path.join(output_dir, "all_maps_grid.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Đã tạo grid tổng hợp: {grid_path}")


if __name__ == "__main__":
    # Test với một map trước
    print("=== TEST với một map trước ===")
    save_single_map_image("maze1")
    
    # Nếu test thành công thì chạy full
    input("\nNhấn Enter để tiếp tục với tất cả maps...")
    
    # Lưu từng hình ảnh map riêng lẻ
    save_map_images()
    
    # Tạo grid tổng hợp
    create_summary_grid()
    
    print("\n🎉 Hoàn thành việc lưu hình ảnh tất cả các map!") 