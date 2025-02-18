import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from PIL import Image
import imageio

def html_to_gif(html_path, frames=50, frame_duration=0.2, capture_interval=0.1):
    """
    将HTML文件转换为GIF动画
    
    参数:
    html_path (str): HTML文件路径
    frames (int): 要捕获的帧数，默认50帧
    frame_duration (float): GIF中每帧显示的时间（秒），默认0.2秒
    capture_interval (float): 捕获帧之间的时间间隔（秒），默认0.1秒
    """
    # 设置Chrome选项
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # 无界面模式
    chrome_options.add_argument('--window-size=1920,1080')  # 设置窗口大小
    
    # 初始化WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    
    # 加载HTML文件
    driver.get(f"file:///{os.path.abspath(html_path)}")
    time.sleep(2)  # 等待页面加载
    
    # 创建临时文件夹存储截图
    temp_dir = "temp_screenshots"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 捕获多个截图
    screenshots = []
    for i in range(frames):
        screenshot_path = os.path.join(temp_dir, f"screenshot_{i}.png")
        driver.save_screenshot(screenshot_path)
        screenshots.append(screenshot_path)
        time.sleep(capture_interval)
    
    # 关闭浏览器
    driver.quit()
    
    # 将截图转换为GIF
    images = []
    for screenshot in screenshots:
        images.append(Image.open(screenshot))
    
    # 生成GIF文件路径（与HTML文件同目录）
    gif_path = os.path.splitext(html_path)[0] + '.gif'
    
    # 保存GIF
    imageio.mimsave(gif_path, images, duration=frame_duration)
    
    # 清理临时文件
    for screenshot in screenshots:
        os.remove(screenshot)
    os.rmdir(temp_dir)
    
    return gif_path

if __name__ == "__main__":
    html_path = "../graph/Day9_pos_topology.html"
    # 生成一个50帧的GIF，每帧显示0.2秒，捕获间隔0.1秒
    gif_path = html_to_gif(html_path, frames=50, frame_duration=0.2, capture_interval=0.1)
    print(f"GIF已保存至: {gif_path}")
