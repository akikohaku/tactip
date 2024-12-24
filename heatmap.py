import cv2
import numpy as np
import math

# 创建画布
width, height = 1000, 900
img = np.ones((height, width, 3), dtype=np.uint8) * 255

# 六边形参数
hex_radius = 40  # 六边形外接圆半径
hex_height = math.sqrt(3) * hex_radius  # 六边形的高度
hex_width = math.sqrt(3) * hex_radius  # 六边形的宽度
offset_x = hex_width   # X方向六边形的间隔
offset_y = hex_height  # Y方向六边形的间隔
center_x, center_y = width // 2, height // 2  # 网格中心
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

# 绘制六边形网格
def draw_hex_grid(img, center, radius, layers, rotation_angle):
    hex_count = 0  # 计数器
    for layer in range(layers):
        for i in range(6):  # 每层六边形环
            angle = math.radians(60 * i)
            for j in range(layer + 1):
                # 计算当前六边形的中心
                cx = int(center[0] + (layer * math.cos(angle)) * offset_x + j * math.cos(angle + math.radians(120)) * offset_x)
                cy = int(center[1] + (layer * math.sin(angle)) * offset_y + j * math.sin(angle + math.radians(120)) * offset_y)
                hexagon = generate_hexagon((cx, cy), radius)
                rotated_hexagon = rotate_hexagon(hexagon, (cx, cy), rotation_angle)

                hex_count += 1
                if hex_count == 10:
                    # 给第12个六边形上色
                    cv2.fillPoly(img, [rotated_hexagon], color=(0, 255, 0))  # 绿色填充
                else:
                    cv2.polylines(img, [rotated_hexagon], isClosed=True, color=(0, 0, 0), thickness=2)


# 绘制六边形网格
def draw_hex_grid_left(img, center_1, radius, layers, rotation_angle):
    hex_count = 0  # 计数器
    for i in range(7):  # 每层六边形环
        for j in range(layers+i):
                # 计算当前六边形的中心
            cx = int(center_1[0] + j  * offset_x- radius*math.sin( math.radians(120))*i)
            cy = int(center_1[1] + i * math.sin( math.radians(120)) * offset_y )
            hexagon = generate_hexagon((cx, cy), radius)
            rotated_hexagon = rotate_hexagon(hexagon, (cx, cy), rotation_angle)

            hex_count += 1
            if hex_count == 2:
                    # 给第12个六边形上色
                cv2.fillPoly(img, [rotated_hexagon], color=(0, 255, 0))  # 绿色填充
                cv2.polylines(img, [rotated_hexagon], isClosed=True, color=(0, 0, 0), thickness=2)
            else:
                cv2.polylines(img, [rotated_hexagon], isClosed=True, color=(0, 0, 0), thickness=2)
    
    for i in range(6):  # 每层六边形环
        for j in range(layers*2-2-i):
                # 计算当前六边形的中心
            cx = int(center_1[0] + j  * offset_x+ radius*math.sin( math.radians(120))*(i-5))
            cy = int(center_1[1]+400+math.sin( math.radians(30))*radius + i * math.sin( math.radians(120)) * offset_y )
            hexagon = generate_hexagon((cx, cy), radius)
            rotated_hexagon = rotate_hexagon(hexagon, (cx, cy), rotation_angle)

            hex_count += 1
            if hex_count == 2:
                    # 给第12个六边形上色
                cv2.fillPoly(img, [rotated_hexagon], color=(0, 255, 0))  # 绿色填充
                cv2.polylines(img, [rotated_hexagon], isClosed=True, color=(0, 0, 0), thickness=2)
            else:
                cv2.polylines(img, [rotated_hexagon], isClosed=True, color=(0, 0, 0), thickness=2)



# 绘制网格，中心加1层
draw_hex_grid_left(img, (250, 60), hex_radius, 7, rotation_angle)

# 显示结果
cv2.imshow("Rotated Hexagonal Grid", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
