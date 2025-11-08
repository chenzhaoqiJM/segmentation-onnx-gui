from PyQt5.QtCore import QThread ,QMutex,pyqtSignal
from PIL import Image
import os
import cv2
import numpy as np
import time

class Info():
    def __init__(self):
        self.name = 'x'
        # self.shape = [1,3,500,1000]
        self.shape = [1,3,1024,1024]

def softmax(x):
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

#线程1 继承QThread
class Thread_1(QThread):
    _signal = pyqtSignal(list)  #线程要发出的信号格式
    def __init__(self, img, net):
        super().__init__() 
        self.img = img
        # self.parameters = parameters
        self.session = net
        
    def run(self):
        try:
            # image_src = cv2.imread(self.path)  # 读取图
            image_src = self.img
            
            inputs_info = self.session.get_inputs()
            in_h, in_w = inputs_info[0].shape[2], inputs_info[0].shape[3]
            print("input info: ", inputs_info[0].shape)
            if in_w != image_src.shape[1] or in_h != image_src.shape[0]:
                image_src = cv2.resize(image_src, (in_w, in_h))
            else:
                pass

            if inputs_info[0].shape[1] == 3:
                # img_in = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
                img_in = image_src
            else:
                # 单通道语义分割

                img_in = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
                img_in = img_in[..., np.newaxis]
                
            img_in = img_in.astype(np.float32)/255.0
            img_in = (img_in- img_in.min())/(img_in.max()-img_in.min())
           
            img_in = np.transpose(img_in, (2, 0, 1))
            img_in = np.expand_dims(img_in, axis=0)
            input_name = inputs_info[0].name
            t1 = time.time()
            outputs = self.session.run(None, {input_name: img_in})
            print("输出的形状：", np.shape(outputs))
            # self.session._reset_session(self.session.get_providers(), None)
            print("运行的时间：", time.time()-t1)
            logmax = softmax(outputs[0])
            print("softmax后形状: ", np.shape(logmax))
            print("head: ", logmax[:,:,0,0])
            # pred2 = logmax[0,1,:,:]
            pred2 = logmax[0,1:,:,:] # [class, H, W] 0是背景，不要
        except Exception as e:
            pred2 = np.ones(shape=(1024,1024), dtype=np.float32)
            print("线程中遇到错误: ", e)
        
        res = pred2
        # 输出shape: [w,h]
        self._signal.emit(["off", res])
        
