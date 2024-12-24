import cv2
import numpy as np

# 打开相机（默认相机索引为 0）
cap = cv2.VideoCapture(0)

initial_positions = None  # 初始标记点位置
tracking_enabled = False  # 标记是否已经初始化

# 检查相机是否成功打开
if not cap.isOpened():
    print("无法打开相机")
    exit()

while True:
    # 读取帧
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        print("无法接收帧 (可能是流的结束?)")
        break

    #height, width, _ = frame.shape

    # 定义裁剪区域 (y1:y2, x1:x2)
    #x2, y2 = 510, 350  # 右下角 (x, y)

    # 确保裁剪区域在帧范围内
   # x1, x2 = max(0, x1), min(width, x2)
    #y1, y2 = max(0, y1), min(height, y2)

    # 裁剪图像
    #frame = frame[y1:y2, x1:x2]


    alpha = 1.5  # 对比度值（1.0-3.0之间调整）
    beta = 50    # 亮度值（-100到100之间调整）
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold_value = 200  # 阈值
    _, frame = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)  # 开运算去噪点
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)  # 闭运算填补小孔
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame, connectivity=8)

    height, width = frame.shape
    marked_points_image = np.ones((height, width, 3), dtype=np.uint8)*255
    # 创建一个标记的彩色图像副本
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    current_positions = []
    # 遍历所有的连通域
    for i in range(1, num_labels):  # 跳过背景（标签 0）
        area = stats[i, cv2.CC_STAT_AREA]  # 获取连通域的面积
        if area > 5 and area < 50:  # 仅处理面积大于 9 的连通域
            center_x, center_y = int(centroids[i][0]), int(centroids[i][1])  # 获取中心坐标
            # 在标记图像上绘制中心点
            current_positions.append((center_x, center_y))

    # 初始化初始位置
    if initial_positions is None and len(current_positions) > 0:
        initial_positions = current_positions.copy()
        tracking_enabled = True
    
    if tracking_enabled:
        for i, (x_current, y_current) in enumerate(current_positions):
            # 如果初始点存在，计算偏移并绘制箭头
            if i < len(initial_positions):
                x_initial, y_initial = initial_positions[i]
                # 绘制箭头
                cv2.arrowedLine(
                    marked_points_image,
                    (x_initial, y_initial),
                    (x_current, y_current),
                    (255, 0, 0),  # 蓝色箭头
                    2,
                    tipLength=0.3
                )
                # 绘制当前点为红色
                cv2.circle(marked_points_image, (x_current, y_current), 3, (0, 0, 255), -1)


    # 显示帧
    cv2.imshow('Camera', frame)
    cv2.imshow('Marked Points', marked_points_image)
    if cv2.waitKey(1) == ord('r'):
        initial_positions = None
        tracking_enabled = False
    # 按下 'q' 键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放相机资源
cap.release()
cv2.destroyAllWindows()