import glob
import numpy as np
import torch
import os
import cv2
from model.model import Unet

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = Unet(in_channels=3, n_classes=1,phi=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('../val/image/*.tif')
    # 遍历素有图片
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[2].split('/')[3] + '_res.tif'
        # 读取图片

        image = cv2.imread(test_path,1)
        image = image / 255
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image).type(torch.FloatTensor)
        image = image.to(device=device, dtype=torch.float)
        # 转为灰度图
        #print(image)
        # 预测
        pred = net(image)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite('/content/unet/unet_lib/datapredict/'+save_res_path, pred)
