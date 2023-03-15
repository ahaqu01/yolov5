## YOLOV5 API

## 一、接口说明

该接口为基于YOLOV5的图像检测接口

由官方代码重构而得，本仓库只有推理代码，

yolov5的官方代码见：https://github.com/ultralytics/yolov5.git

本API提供检测类型80种，对应索引和类型见下图

![img](https://img-blog.csdnimg.cn/20210601214628463.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNjY3MDUyOQ==,size_16,color_FFFFFF,t_70)

## 二、环境说明

见requirements.txt

## 三、参数说明

##### **yolov5.src.detecter.Detecter**

类构建参数：

1）model_cfg: str, 模型结构参数路径，可在“./src/config”中选择不同模型结构参数

2）model_weights: str, 模型权重路径，可在44服务器，路径“./yolov5/src/weights”中获取参数1）对应模型结构参数的权重

3）input_size: int, 输入到模型的图像最长边 

4）device: torch.device object, 推理使用的device 

5）half: bool, 是否使用半精度推理

6）conf_thres: float, 无效置信阈值 

7）iou_thres: float, nms时的iou阈值

8）select_classes: list, 选择的检测类型，对照接口说明中的索引图，如果要检测人和自行车输入select_classes=[0,1] 

9）agnostic_nms: bool, 多个类一起应用nms还是执行按照不同的类分别应用nms 

10）multi_label: bool, 单个目标是否输出多个标签

11）max_det: int, 最多检测目标数量

###### yolov5.src.detecter.Detecter.single_inference

说明: 单张图片yolov5目标检测方法

img: ndarray, shape: (H, W, 3), 通道顺序: RGB

output: 长度为0的list, 设为pred, 

其中pred[0]为tensor, pred[0]的shape为(N, 6), 

其中N代表检测出的目标数, 

对于每一个目标, (:, :4)分别为bbox左上角右下角xy坐标, (:, 4)代表置信度, (:, 5)代表目标类别。



## 四、使用样例

在自己的项目目录下，git submodule add  https://github.com/ahaqu01/yolov5.git

便会在项目目录下下载到yolov5相关代码

下载完成后，便可在自己项目中使用yolov5API，**使用样例**如下：

```python
from yolov5.src.detecter import Detecter
from yolov5.src.utils.torch_utils import select_device

# 设置推理device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# step1: 创建Detecter类
dt = Detecter(model_cfg='/workspace/Yolov5_DeepSort_Pytorch/yolov5/src/config/yolov5s.yaml',
              model_weights='/workspace/Yolov5_DeepSort_Pytorch/yolov5/src/weights/yolov5s_resave.pt',
              device=device,
              select_classes=[2])

# step2: 读取图片以及single inference
img_path = "/workspace/Yolov5_DeepSort_Pytorch/test_data/01.jpeg"
im = Image.open(img_path)
img = np.array(im)
pred = dt.single_inference(img)
print(pred)
```

 

