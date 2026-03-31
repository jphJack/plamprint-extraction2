# 掌纹纹理提取 (Palmprint Texture Extraction)

基于多种图像处理技术的掌纹纹理提取项目，综合使用局部对比度法、形态学方法和改进滑动窗口法进行纹理提取。

## 项目特点

- **多种提取方法**：综合三种不同的纹理提取算法
- **四方向扫描**：滑动窗口法支持从左到右、从右到左、从上到下、从下到上四个方向
- **结果可视化**：自动生成对比图和叠加显示图

## 输入图像

![输入图像](aa.jpg)

## 输出结果

### 综合结果

![综合结果](texture_final_improved.png)

### 各方法对比

![方法对比](texture_extraction_improved.png)

### 单独方法结果

| 局部对比度法 | 形态学方法 | 改进滑动窗口法 |
|:---:|:---:|:---:|
| ![局部对比度](texture_local_contrast.png) | ![形态学](texture_morphology.png) | ![滑动窗口](texture_sliding_window.png) |

### 四方向扫描结果

| 从左到右 | 从右到左 |
|:---:|:---:|
| ![left_to_right](texture_left_to_right.png) | ![right_to_left](texture_right_to_left.png) |

| 从上到下 | 从下到上 |
|:---:|:---:|
| ![top_to_bottom](texture_top_to_bottom.png) | ![bottom_to_top](texture_bottom_to_top.png) |

## 技术原理

### 1. 局部对比度法
纹理区域通常具有较低的像素值和较高的局部对比度。该方法通过计算局部均值和局部标准差来检测纹理区域。

### 2. 形态学方法
通过顶帽变换（Top-hat）和黑帽变换（Black-hat）提取局部较暗的区域，适用于提取细小的纹理特征。

### 3. 改进滑动窗口法
使用二维窗口进行四方向扫描，通过比较窗口均值与中心像素值来判断纹理位置，并进行形态学后处理连接断裂纹理。

## 使用方法

### 环境要求

```bash
pip install opencv-python numpy matplotlib
```

### 运行

```bash
python texture_extraction.py
```

修改代码中的 `image_path` 变量以处理不同的图像：

```python
image_path = 'aa.jpg'  # 修改为你的图像路径
```

## 输出文件

运行后会生成以下文件：

- `texture_final_improved.png` - 综合纹理提取结果
- `texture_extraction_improved.png` - 方法对比可视化图
- `texture_local_contrast.png` - 局部对比度法结果
- `texture_morphology.png` - 形态学方法结果
- `texture_sliding_window.png` - 滑动窗口法结果
- `texture_left_to_right.png` - 从左到右扫描结果
- `texture_right_to_left.png` - 从右到左扫描结果
- `texture_top_to_bottom.png` - 从上到下扫描结果
- `texture_bottom_to_top.png` - 从下到上扫描结果

## 许可证

MIT License
