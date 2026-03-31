import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_texture_improved(image, window_width=20, window_height=5, threshold_ratio=0.75):
    """
    改进的纹理提取算法
    
    改进点：
    1. 使用二维窗口（高度>1）减少噪声影响
    2. 使用局部对比度判断，而非单一像素
    3. 四方向扫描后进行形态学处理连接纹理
    
    Parameters:
    - image: 输入图像
    - window_width: 窗口宽度
    - window_height: 窗口高度
    - threshold_ratio: 阈值比例
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64)
    h, w = gray.shape
    
    def extract_horizontal(direction='left_to_right'):
        texture_map = np.zeros((h, w), dtype=np.float64)
        half_h = window_height // 2
        
        for row in range(half_h, h - half_h):
            if direction == 'left_to_right':
                col = 0
                while col < w - window_width:
                    window = gray[row - half_h:row + half_h + 1, col:col + window_width]
                    avg = np.mean(window)
                    next_col = col + window_width
                    
                    if next_col < w:
                        center_val = gray[row, next_col]
                        if center_val < avg * threshold_ratio:
                            for dr in range(-half_h, half_h + 1):
                                if 0 <= row + dr < h and next_col < w:
                                    texture_map[row + dr, next_col] += 1
                            start = max(0, next_col - window_width // 2)
                            end = min(w, next_col + window_width // 2 + 1)
                            col = next_col + 1
                            continue
                    col += 1
            else:
                col = w - 1
                while col >= window_width:
                    window = gray[row - half_h:row + half_h + 1, col - window_width + 1:col + 1]
                    avg = np.mean(window)
                    prev_col = col - window_width
                    
                    if prev_col >= 0:
                        center_val = gray[row, prev_col]
                        if center_val < avg * threshold_ratio:
                            for dr in range(-half_h, half_h + 1):
                                if 0 <= row + dr < h and prev_col >= 0:
                                    texture_map[row + dr, prev_col] += 1
                            col = prev_col - 1
                            continue
                    col -= 1
        
        return texture_map
    
    def extract_vertical(direction='top_to_bottom'):
        texture_map = np.zeros((h, w), dtype=np.float64)
        half_w = window_width // 2
        
        for col in range(half_w, w - half_w):
            if direction == 'top_to_bottom':
                row = 0
                while row < h - window_width:
                    window = gray[row:row + window_width, col - half_w:col + half_w + 1]
                    avg = np.mean(window)
                    next_row = row + window_width
                    
                    if next_row < h:
                        center_val = gray[next_row, col]
                        if center_val < avg * threshold_ratio:
                            for dc in range(-half_w, half_w + 1):
                                if 0 <= col + dc < w and next_row < h:
                                    texture_map[next_row, col + dc] += 1
                            row = next_row + 1
                            continue
                    row += 1
            else:
                row = h - 1
                while row >= window_width:
                    window = gray[row - window_width + 1:row + 1, col - half_w:col + half_w + 1]
                    avg = np.mean(window)
                    prev_row = row - window_width
                    
                    if prev_row >= 0:
                        center_val = gray[prev_row, col]
                        if center_val < avg * threshold_ratio:
                            for dc in range(-half_w, half_w + 1):
                                if 0 <= col + dc < w and prev_row >= 0:
                                    texture_map[prev_row, col + dc] += 1
                            row = prev_row - 1
                            continue
                    row -= 1
        
        return texture_map
    
    print("正在进行四方向纹理提取...")
    
    print("  处理方向: 从左到右")
    result_lr = extract_horizontal('left_to_right')
    print("  处理方向: 从右到左")
    result_rl = extract_horizontal('right_to_left')
    print("  处理方向: 从上到下")
    result_tb = extract_vertical('top_to_bottom')
    print("  处理方向: 从下到上")
    result_bt = extract_vertical('bottom_to_top')
    
    combined = result_lr + result_rl + result_tb + result_bt
    
    if combined.max() > 0:
        combined = combined / combined.max() * 255
    combined = combined.astype(np.uint8)
    
    return combined, {
        'left_to_right': (result_lr / result_lr.max() * 255).astype(np.uint8) if result_lr.max() > 0 else result_lr.astype(np.uint8),
        'right_to_left': (result_rl / result_rl.max() * 255).astype(np.uint8) if result_rl.max() > 0 else result_rl.astype(np.uint8),
        'top_to_bottom': (result_tb / result_tb.max() * 255).astype(np.uint8) if result_tb.max() > 0 else result_tb.astype(np.uint8),
        'bottom_to_top': (result_bt / result_bt.max() * 255).astype(np.uint8) if result_bt.max() > 0 else result_bt.astype(np.uint8)
    }


def extract_texture_local_contrast(image, block_size=15, k=0.5):
    """
    基于局部对比度的纹理提取
    
    原理：纹理区域通常具有较低的像素值和较高的局部对比度
    使用局部均值和局部标准差来检测纹理
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64)
    
    print("计算局部均值...")
    local_mean = cv2.blur(gray, (block_size, block_size))
    
    print("计算局部标准差...")
    local_mean_sq = cv2.blur(gray ** 2, (block_size, block_size))
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean ** 2, 0))
    
    print("检测纹理区域...")
    threshold = local_mean - k * local_std
    texture_map = (gray < threshold).astype(np.uint8) * 255
    
    return texture_map, local_mean, local_std


def extract_texture_morphology(image, kernel_size=5, iterations=2):
    """
    使用形态学方法提取纹理
    
    通过顶帽变换提取局部较暗的区域
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    print("执行形态学顶帽变换...")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
    
    texture_map = cv2.add(tophat, blackhat)
    
    _, binary = cv2.threshold(texture_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary, tophat, blackhat


def extract_texture_combined(image):
    """
    综合方法：结合多种技术提取纹理
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    print("方法1: 局部对比度法...")
    texture1, _, _ = extract_texture_local_contrast(gray, block_size=15, k=0.4)
    
    print("方法2: 形态学方法...")
    texture2, _, _ = extract_texture_morphology(gray, kernel_size=7, iterations=1)
    
    print("方法3: 改进的滑动窗口法...")
    texture3, _ = extract_texture_improved(gray, window_width=15, window_height=3, threshold_ratio=0.8)
    
    print("合并结果...")
    combined = cv2.addWeighted(texture1, 0.4, texture2, 0.3, 0)
    combined = cv2.addWeighted(combined, 0.7, texture3, 0.3, 0)
    
    print("后处理：形态学闭运算连接断裂纹理...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    print("后处理：去除小噪点...")
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return combined, texture1, texture2, texture3


def visualize_results(original, final_result, intermediate_results):
    """
    可视化结果
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    titles = ['局部对比度法', '形态学方法', '改进滑动窗口法']
    for i, (result, title) in enumerate(zip(intermediate_results, titles), start=2):
        plt.subplot(2, 3, i)
        plt.imshow(result, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(final_result, cmap='gray')
    plt.title('综合结果')
    plt.axis('off')
    
    if len(original.shape) == 3:
        overlay = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).copy()
    else:
        overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB).copy()
    
    overlay[final_result > 50] = [255, 0, 0]
    
    plt.subplot(2, 3, 6)
    plt.imshow(overlay)
    plt.title('纹理叠加显示 (红色)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('texture_extraction_improved.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n结果已保存到 texture_extraction_improved.png")


if __name__ == '__main__':
    image_path = 'aa.jpg'
    print(f"读取图像: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        exit(1)
    
    print(f"图像尺寸: {image.shape}")
    print("=" * 50)
    
    final_result, texture1, texture2, texture3 = extract_texture_combined(image)
    
    cv2.imwrite('texture_final_improved.png', final_result)
    print("\n最终纹理结果已保存到 texture_final_improved.png")
    
    cv2.imwrite('texture_local_contrast.png', texture1)
    cv2.imwrite('texture_morphology.png', texture2)
    cv2.imwrite('texture_sliding_window.png', texture3)
    print("各方法纹理结果已保存")
    
    visualize_results(image, final_result, (texture1, texture2, texture3))
