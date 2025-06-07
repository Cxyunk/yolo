from ultralytics import YOLO
import keyboard
# 加载预训练的 YOLOv11n 模型
model = YOLO('yolo11n.pt')
source = 'E:\\ultralytics-main\\csreen.png' #更改为自己的图片路径
# 运行推理，并附加参数
model.predict(source, save=True,show=True)
while True:
    if keyboard.is_pressed('p'):
        break