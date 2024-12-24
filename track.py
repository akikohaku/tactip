import cv2
import numpy as np
import math

# 打开相机
cap = cv2.VideoCapture(0)



# 六边形参数
hex_radius = 20  # 六边形外接圆半径
hex_height = math.sqrt(3) * hex_radius  # 六边形的高度
hex_width = math.sqrt(3) * hex_radius  # 六边形的宽度
offset_x = hex_width   # X方向六边形的间隔
offset_y = hex_height  # Y方向六边形的间隔
rotation_angle = 30  # 每个六边形的旋转角度

# 生成六边形的顶点
def generate_hexagon(center, radius):
    return np.array([
        (center[0] + radius * math.cos(math.radians(angle)),
         center[1] + radius * math.sin(math.radians(angle)))
        for angle in range(0, 360, 60)
    ], dtype=np.float32)

# 旋转六边形顶点
def rotate_hexagon(hexagon, center, angle):
    angle_rad = math.radians(angle)
    cos_theta, sin_theta = math.cos(angle_rad), math.sin(angle_rad)
    rotated_hexagon = []
    for x, y in hexagon:
        # 平移到原点
        x -= center[0]
        y -= center[1]
        # 旋转
        x_rot = x * cos_theta - y * sin_theta
        y_rot = x * sin_theta + y * cos_theta
        # 平移回原位置
        x_rot += center[0]
        y_rot += center[1]
        rotated_hexagon.append((x_rot, y_rot))
    return np.array(rotated_hexagon, dtype=np.int32)

if not cap.isOpened():
    print("无法打开相机")
    exit()

# 绘制六边形网格
def draw_hex_grid_left(img, center_1, radius, layers, rotation_angle,initial_positions,current_positions):
    count=0
    for i in range(7):  # 每层六边形环
        for j in range(layers+i):
                # 计算当前六边形的中心
            cx = int(center_1[0] + j  * offset_x- radius*math.sin( math.radians(120))*i)
            cy = int(center_1[1] + i * math.sin( math.radians(120)) * offset_y )
            hexagon = generate_hexagon((cx, cy), radius)
            rotated_hexagon = rotate_hexagon(hexagon, (cx, cy), rotation_angle)
            x_initial, y_initial = initial_positions[count]
            x_current, y_current = current_positions[count]
            cv2.fillPoly(img, [rotated_hexagon], color=(0, 0, min(255,int(((x_initial-x_current)**2+(y_initial-y_current)**2)))))  # 绿色填充
            cv2.polylines(img, [rotated_hexagon], isClosed=True, color=(0, 0, 0), thickness=2)
            count+=1
    for i in range(6):  # 每层六边形环
        for j in range(layers*2-2-i):
                # 计算当前六边形的中心
            cx = int(center_1[0] + j  * offset_x+ radius*math.sin( math.radians(120))*(i-5))
            cy = int(center_1[1]+radius*10+math.sin( math.radians(30))*radius + i * math.sin( math.radians(120)) * offset_y )
            hexagon = generate_hexagon((cx, cy), radius)
            rotated_hexagon = rotate_hexagon(hexagon, (cx, cy), rotation_angle)
            x_initial, y_initial = initial_positions[count]
            x_current, y_current = current_positions[count]
            cv2.fillPoly(img, [rotated_hexagon], color=(0, 0, min(255,int(((x_initial-x_current)**2+(y_initial-y_current)**2)))))  # 绿色填充
            cv2.polylines(img, [rotated_hexagon], isClosed=True, color=(0, 0, 0), thickness=2)
            count+=1


def sort_hexagonal_grid(positions, epsilon=1e-1):
    """
    对六边形点阵的点进行按行排序。
    
    参数：
    - positions: numpy数组，形状为(N, 2)，每行是一个点的坐标 [center_x, center_y]。
    - epsilon: float，y坐标分组的阈值，决定分组时y坐标的误差范围。
    
    返回：
    - sorted_positions: numpy数组，按行排序后的点阵。
    """
    # 按y坐标排序，方便后续分组
    sorted_by_y = positions[np.argsort(positions[:, 1])]
    
    # 分组：将y坐标相近的点分为一组
    rows = []
    current_row = [sorted_by_y[0]]
    
    for i in range(1, len(sorted_by_y)):
        if abs(sorted_by_y[i, 1] - current_row[-1][1]) < epsilon:
            current_row.append(sorted_by_y[i])
        else:
            rows.append(current_row)
            current_row = [sorted_by_y[i]]
    rows.append(current_row)  # 添加最后一行
    
    # 对每一行按x坐标排序
    sorted_positions = []
    for row in rows:
        row_sorted = sorted(row, key=lambda p: p[0])  # 按x坐标排序
        sorted_positions.extend(row_sorted)
    
    return np.array(sorted_positions)


# 用于保存标记点的初始和当前状态
initial_positions = None
current_positions = None

# 光流参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 保存前一帧的灰度图像
prev_gray = None

while True:
    # 读取帧
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧")
        break

    # 转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 初始化标记点
    if initial_positions is None:
        # 转换为二值图像
        _, binary_frame = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY)

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_frame, connectivity=8)

        # 记录初始标记点
        initial_positions = []
        for i in range(1, num_labels):  # 跳过背景（标签 0）
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 5 and area < 100:
                center_x, center_y = int(centroids[i][0]), int(centroids[i][1])
                initial_positions.append([center_x, center_y])
        
        # 转换为 NumPy 数组
        initial_positions = np.array(initial_positions, dtype=np.float32)
        initial_positions = sort_hexagonal_grid(initial_positions, epsilon=5)
        current_positions = initial_positions.copy()
        prev_gray = gray_frame.copy()
        continue

    # 使用光流法计算当前标记点的位置
    next_positions, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, current_positions, None, **lk_params)

    # 更新上一帧图像
    prev_gray = gray_frame.copy()

    # 创建白色背景图像
    height, width = frame.shape[:2]
    marked_points_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 绘制初始位置和当前跟踪位置的箭头
    for i, (x_next, y_next) in enumerate(next_positions):
        if status[i]:  # 确保跟踪有效
            x_initial, y_initial = initial_positions[i]
            x_current, y_current = current_positions[i]

            # 绘制箭头
            cv2.arrowedLine(
                marked_points_image,
                (int(x_initial), int(y_initial)),
                (int(x_next), int(y_next)),
                (255, 0, 0),  # 蓝色箭头
                2,
                tipLength=0.3
            )
            # 绘制当前点为红色
            cv2.circle(marked_points_image, (int(x_next), int(y_next)), 3, (0, 0, 255), -1)
            label = f"{i + 1}"  # 编号从 1 开始
            cv2.putText(
            marked_points_image,
            label,
            (int(x_next) - 10, int(y_next) + 10),  # 调整文本位置
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # 字体大小
            (0, 0, 0),  # 黑色文本
            1,  # 文本粗细
            lineType=cv2.LINE_AA
        )

    # 更新当前标记点位置
    current_positions = next_positions.copy()

    # 显示结果
    # 绘制网格，中心加1层
    img = np.ones((450, 600, 3), dtype=np.uint8) * 125
    draw_hex_grid_left(img, (125, 60), hex_radius, 7, rotation_angle,initial_positions,current_positions)

# 显示结果
    cv2.imshow("Rotated Hexagonal Grid", img)
    cv2.imshow('Original', frame)
    cv2.imshow('Tracked Changes', marked_points_image)

    # 按下 'r' 键重置初始状态
    if cv2.waitKey(1) == ord('r'):
        initial_positions = None
        current_positions = None

    # 按下 'q' 键退出
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
