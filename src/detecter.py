from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from .models.yolo import Model
from .models.sub_modules import Detections, AutoShape
from .utils.augmentations import letterbox
from .utils.general import make_divisible, non_max_suppression, scale_coords


class Detecter(nn.Module):
    # YOLOv5 python inference on pytorch
    def __init__(self,
                 model_cfg='yolov5s.yaml',
                 model_weights='./yolov5s.pt',
                 input_size=640,
                 device=None,
                 half=False,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 select_classes=None,
                 agnostic_nms=False,
                 multi_label=False,
                 max_det=1000):
        # class-agnostic NMS
        # NMS multiple labels per box
        # maximum number of detections per image
        super().__init__()

        self.device = device
        self.half = half
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.select_classes = select_classes
        self.agnostic_nms = agnostic_nms
        self.multi_label = multi_label
        self.max_det = max_det

        # create model
        self.model = Model(model_cfg)

        # load weights
        weights_state_dict = torch.load(model_weights, map_location=device)
        self.model.load_state_dict(weights_state_dict, strict=False)  # load weights
        self.stride = int(self.model.stride.max())# model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

        # use half precision
        if self.half:
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()

    def pre_process(self, img_paths):
        # img_paths: img path list,[path1, path2, path3, ...]
        img_num = len(img_paths)
        imgs, shape0, shape1, files = [], [], [], []  # image and inference shapes, filenames
        for i, path in enumerate(img_paths):
            files.append(Path(path).with_suffix('.jpg').name)
            img = np.asarray(Image.open(path))
            s = img.shape[:2]  # HWC
            shape0.append(s)  # image shape (H,W)
            g = (self.input_size / max(s))  # gain
            shape1.append([y * g for y in s])  # resized shape (H, W)
            imgs.append(img if img.data.contiguous else np.ascontiguousarray(img))

        shape1 = [make_divisible(x, self.stride) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if img_num > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        if self.half:
            x = torch.from_numpy(x).type(torch.HalfTensor) / 255  # uint8 to fp16
        else:
            x = torch.from_numpy(x).type(torch.FloatTensor) / 255  # uint8 to fp32
        x = x.to(self.device)
        return x, imgs, shape0, shape1, files

    def single_pre_process(self, img):
        # img: ndarray, (H, W, 3), RGB
        # Padded resize
        src_img = img.copy()
        x = letterbox(img, self.input_size, stride=self.stride)[0]
        x = x.transpose((2, 0, 1))  # HWC to CHW,
        x = np.ascontiguousarray(x)
        if self.half:
            x = torch.from_numpy(x).type(torch.HalfTensor) / 255  # uint8 to fp16
        else:
            x = torch.from_numpy(x).type(torch.FloatTensor) / 255  # uint8 to fp32
        x = x.to(self.device)
        x = x.unsqueeze(0)
        return x, [src_img], [src_img.shape[:2]], x.shape[-2:]

    def post_process(self, pred, shape0, shape1):
        # Post-process
        pred = non_max_suppression(pred,
                                   conf_thres=self.conf_thres,
                                   iou_thres=self.iou_thres,
                                   classes=self.select_classes,
                                   agnostic=self.agnostic_nms,
                                   multi_label=self.multi_label,
                                   max_det=self.max_det)
        batch_size = len(pred)
        for i in range(batch_size):
            scale_coords(shape1, pred[i][:, :4], shape0[i])
        return pred

    def single_post_process(self, pred, shape0, shape1):
        # Post-process
        pred = non_max_suppression(pred,
                                   conf_thres=self.conf_thres,
                                   iou_thres=self.iou_thres,
                                   classes=self.select_classes,
                                   agnostic=self.agnostic_nms,
                                   multi_label=self.multi_label,
                                   max_det=self.max_det)
        scale_coords(shape1, pred[0][:, :4], shape0[0])
        return pred

    @torch.no_grad()
    def batch_inference(self, img_paths, With_Detections=False):  # batch-inference
        t = [time_sync()]

        # do imgs pre-process
        x, imgs, shape0, shape1, files = self.pre_process(img_paths)
        t.append(time_sync())

        with amp.autocast(enabled=self.device.type != 'cpu'):
            # do inference
            pred = self.model(x, augment=False, profile=False)[0]  # forward
            t.append(time_sync())

            # do pred post-process
            pred = self.post_process(pred, shape0=shape0, shape1=shape1)
            t.append(time_sync())

        if With_Detections:
            res = Detections(imgs, pred, files, t, self.names, x.shape)
            return res
        else:
            return pred

    @torch.no_grad()
    def single_inference(self, img):  # single-inference
        # img: ndarray, (H, W, 3), RGB
        # do img pre-process
        x, img, shape0, shape1 = self.single_pre_process(img)

        with amp.autocast(enabled=self.device.type != 'cpu'):
            # do inference
            pred = self.model(x, augment=False, profile=False)[0]  # forward

            # do pred post-process
            pred = self.single_post_process(pred, shape0, shape1)

            return pred


if __name__ == "__main__":
    import os
    import glob
    import time
    from .utils.torch_utils import select_device, time_sync

    device = "1"
    device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else device)
    dt = Detecter(model_cfg='/workspace/huangniu_demo/yolov5/src/config/yolov5m6.yaml',
                  model_weights='/workspace/huangniu_demo/yolov5/src/weights/yolov5m6_resave.pt',
                  input_size=1280,
                  device=device,
                  select_classes=[0],
                  conf_thres=0.5,
                  iou_thres=0.45
                  )

    # get paths
    # img_paths = []
    # test_data_root = "/workspace/huangniu_demo/test_data/yolov5_testdata"
    # test_data_result_root = "/workspace/huangniu_demo/test_data/yolov5_testdata_results"
    # for i, (root, dirs, files) in enumerate(os.walk(test_data_root)):
    #     for file in files:
    #         if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
    #             img_paths.append(os.path.join(root, file))

    # # batch inference
    # res = dt.batch_inference(img_paths)
    # res.print()
    # res.save(save_dir=test_data_result_root)

    # single inference
    # img_path = "/workspace/huangniu_demo/test_data/yolov5_testdata/祥移677汽车城东门路面_20211207101123至20211207102309/mp4-00004.jpeg"
    # im = Image.open(img_path)
    # img = np.array(im)
    # pred = dt.single_inference(img)
    # print(pred)

    # speed test
    img_path = "/workspace/huangniu_demo/test_data/yolov5_testdata/祥移677汽车城东门路面_20211207101123至20211207102309/mp4-00004.jpeg"
    im = Image.open(img_path)
    img = np.array(im)
    s_t = time.time()
    for i in range(100):
        pred = dt.single_inference(img)
        print(i, pred[0].shape)
    e_t = time.time()
    print(1000*(e_t-s_t)/100.)

