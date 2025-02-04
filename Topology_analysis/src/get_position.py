"""
This script provides an interactive tool for marking points on an image and saving their relative coordinates.
The script allows users to:
1. Click left mouse button to add points
2. Click left mouse button multiple times quickly in same position to skip that many numbers
3. Click right mouse button to undo the last point
4. Drag existing points to adjust their positions
5. Close the window to save the points and view their relative positions
"""

import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from typing import List, Tuple, Dict, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Text
import time
import os

# Configure matplotlib settings
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class PointMarker:
    def __init__(self, image_path: str):
        """
        Initialize the PointMarker with the given image path.
        
        Args:
            image_path (str): Path to the image file
        """
        self.image_path = image_path
        self.img = mpimg.imread(image_path)
        self.img_height, self.img_width = self.img.shape[0], self.img.shape[1]
        
        self.clicked_points: Dict[int, Tuple[float, float]] = {}  # 使用字典存储点的编号和坐标
        self.point_plots: Dict[int, Line2D] = {}  # 使用字典存储点的编号和图形对象
        self.text_labels: Dict[int, Text] = {}  # 使用字典存储点的编号和文本对象
        
        self.current_number = 1  # 当前编号
        self.last_click_time = 0  # 用于检测双击
        self.last_click_pos = (None, None)  # 用于存储上次点击位置
        self.double_click_threshold = 0.6  # 双击时间阈值（秒）
        
        # 拖动相关变量
        self.dragging = False
        self.drag_point_number: Optional[int] = None
        self.drag_threshold = 5  # 拖动判定阈值（像素）
        
        self.fig, self.ax = plt.subplots()
        self.setup_plot()
        
    def setup_plot(self) -> None:
        """Set up the initial plot configuration."""
        self.ax.imshow(self.img)
        self.ax.set_title('请用左键标点，右键撤销上一次标点，双击跳过编号，拖动可调整点位 (关闭窗口结束)')
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)
        
    def find_nearest_point(self, x: float, y: float) -> Optional[int]:
        """Find the number of the nearest point within threshold distance."""
        min_dist = float('inf')
        nearest_num = None
        
        for num, (px, py) in self.clicked_points.items():
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < min_dist and dist < self.drag_threshold:
                min_dist = dist
                nearest_num = num
                
        return nearest_num
        
    def onclick(self, event) -> None:
        """
        Handle mouse click events.
        
        Args:
            event: MouseEvent object containing click information
        """
        if not event.inaxes:
            return
            
        if event.button == 1:
            current_time = time.time()
            current_pos = (event.xdata, event.ydata)
            
            # 检查是否是快速连续点击
            is_consecutive_click = (current_time - self.last_click_time < self.double_click_threshold and
                             self.last_click_pos[0] is not None and
                             abs(current_pos[0] - self.last_click_pos[0]) < 5 and
                             abs(current_pos[1] - self.last_click_pos[1]) < 5)
            
            if is_consecutive_click:
                self.current_number += 1
                print(f"跳过编号 {self.current_number - 1}")
                self.last_click_time = current_time
                self.last_click_pos = current_pos
                return
                
            # 检查是否点击了已存在的点
            nearest_num = self.find_nearest_point(event.xdata, event.ydata)
            if nearest_num is not None:
                # 开始拖动已存在的点
                self.dragging = True
                self.drag_point_number = nearest_num
                return
                
            self._add_point(event.xdata, event.ydata)
            self.last_click_time = current_time
            self.last_click_pos = current_pos
            
        elif event.button == 3:
            self._remove_last_point()
            
    def onrelease(self, event) -> None:
        """Handle mouse button release events."""
        if event.button == 1:
            self.dragging = False
            self.drag_point_number = None
            
    def onmotion(self, event) -> None:
        """Handle mouse motion events for dragging points."""
        if not self.dragging or self.drag_point_number is None or not event.inaxes:
            return
            
        # 更新点的位置
        self.clicked_points[self.drag_point_number] = (event.xdata, event.ydata)
        
        # 更新点的显示位置
        self.point_plots[self.drag_point_number].set_data([event.xdata], [event.ydata])
        
        # 更新标签位置
        self.text_labels[self.drag_point_number].set_position((event.xdata + 5, event.ydata))
        
        plt.draw()
        
    def _add_point(self, x: float, y: float) -> None:
        """Add a new point to the plot."""
        self.clicked_points[self.current_number] = (x, y)
        point = self.ax.plot(x, y, 'ro', markersize=5)[0]
        self.point_plots[self.current_number] = point
        
        text = self.ax.text(x+5, y, str(self.current_number), color='white', 
                          fontsize=8, backgroundcolor='black')
        self.text_labels[self.current_number] = text
        
        self.current_number += 1
        plt.draw()
        
    def _remove_last_point(self) -> None:
        """Remove the last added point."""
        if not self.clicked_points:
            return
            
        last_number = max(self.clicked_points.keys())
        
        # 移除点和标签
        self.point_plots[last_number].remove()
        self.text_labels[last_number].remove()
        
        # 从字典中删除
        del self.clicked_points[last_number]
        del self.point_plots[last_number]
        del self.text_labels[last_number]
        
        # 更新当前编号
        self.current_number = last_number
        
        plt.draw()
        
    def get_relative_coordinates(self) -> np.ndarray:
        """
        Convert clicked points to relative coordinates.
        
        Returns:
            np.ndarray: Array of relative coordinates with their numbers.
            Maintains original numbers (including gaps from skipped numbers).
        """
        if not self.clicked_points:
            return np.array([])
            
        # 将字典转换为有序列表，保持原始编号
        sorted_points = sorted(self.clicked_points.items())
        numbers = np.array([num for num, _ in sorted_points])
        coordinates = np.array([(x / self.img_width, y / self.img_height) 
                              for _, (x, y) in sorted_points])
        
        # 组合原始编号和相对坐标
        return np.column_stack((numbers, coordinates))
        
    def save_coordinates(self, output_file: str) -> None:
        """
        Save relative coordinates to a CSV file.
        Maintains original numbers (including gaps from skipped numbers).
        
        Args:
            output_file (str): Path to save the coordinates
        """
        relative_points = self.get_relative_coordinates()
        if len(relative_points) > 0:
            np.savetxt(output_file, relative_points, delimiter=',',
                      header='number,relative_x,relative_y', comments='')
            print(f"已保存标记点相对坐标至 {output_file}（共 {len(relative_points)} 个点，编号范围 {int(relative_points[0,0])} - {int(relative_points[-1,0])}）")
            self._plot_relative_coordinates(relative_points)
        else:
            print("未标记任何点，不生成输出文件及相对坐标系散点图。")
            
    def _plot_relative_coordinates(self, relative_points: np.ndarray) -> None:
        """
        Plot the relative coordinates in a scatter plot and save it.
        Shows the original numbers (maintaining gaps from skipped numbers).
        
        Args:
            relative_points (np.ndarray): Array of relative coordinates to plot
        """
        # 创建保存图像的目录
        graph_dir = os.path.join(os.path.dirname(os.path.dirname(self.image_path)), 'graph')
        os.makedirs(graph_dir, exist_ok=True)
        
        # 获取原始图像文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        fig2, ax2 = plt.subplots()
        ax2.scatter(relative_points[:,1], relative_points[:,2], c='r', s=20)
        ax2.set_title('标记点在相对坐标系中的分布')
        ax2.set_xlabel('relative x')
        ax2.set_ylabel('relative y')
        ax2.set_xlim([0,1])
        ax2.set_ylim([1,0])
        ax2.grid(True)
        
        for num, rx, ry in relative_points:
            ax2.text(rx+0.01, ry, str(int(num)), color='red', fontsize=8)
            
        # 保存图像
        save_path = os.path.join(graph_dir, f'{base_name}_coordinates.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存坐标分布图至 {save_path}")
        
        plt.show(block=True)

def main():
    """Main function to run the point marking tool."""
    image_path = '../datasets/Day9_Max.png'
    output_file = '../datasets/Day9_Max_position.csv'
    
    marker = PointMarker(image_path)
    plt.show(block=True)
    marker.save_coordinates(output_file)

if __name__ == '__main__':
    main()