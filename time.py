import time

import torch

from ultralytics import YOLO

# 创建新模型（随机初始化权重）
model = YOLO("ultralytics/cfg/models/11/yolo11n.yaml")  # 使用你的模型配置文件

# 生成随机测试图像（640×640）
dummy_img = torch.rand(1, 3, 640, 640)  # BCHW格式

# 预热（GPU首次运行会有额外开销）
for _ in range(5):
    model.predict(dummy_img, device=0)

# 正式计时（测试100次取平均）
total_time = 0
for _ in range(100):
    start = time.time()
    model.predict(dummy_img, device=0)
    end = time.time()
    total_time += end - start

avg_time = total_time / 100 * 1000  # 转换为毫秒
print(f"平均推理时间：{avg_time:.2f}ms")
