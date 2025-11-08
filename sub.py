from PyQt5.QtWidgets import  QMainWindow
from PyQt5.QtWidgets import QDialog, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QIcon, QPalette, QColor, QImage, QResizeEvent
from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtCore
import numpy as np
import os
from PIL import Image, ImageQt, ImageEnhance
import qtawesome as qta
import cv2
from cmythread import Thread_1
from ui_files.auto import Ui_MainWindow
import onnxruntime
from natsort import natsorted

class SubPage(QMainWindow):
    def __init__(self, parent=None):
        super(SubPage, self).__init__(parent)
        self.ui = Ui_MainWindow()  # 初始化窗口对象
        self.ui.setupUi(self)  # 初始化控件
        self.__set_ui()
        
        self.__set_global_variable()  # 初始化必要变量
        self.__set_signal_and_slot()  # 初始化信号和槽的连接
        self.__load_default_img()
        
        try:
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = 4 # 设置每个操作的线程数，以控制并行执行的程度。
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL 	# ORT_SEQUENTIAL ORT_PARALLEL 并行执行模式。在这种模式下，计算图中的操作可以并行执行
            sess_options.inter_op_num_threads = 4 # 设置跨操作的线程数，以控制这些操作之间的并行执行程度。
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = False
            # 指定CUDA设备
            onnx_path_demo = "./models/PP-Matting-1024.onnx"
            if self.ui.useGPURB.isChecked():
                self.net = onnxruntime.InferenceSession(onnx_path_demo, providers=['CUDAExecutionProvider'],sess_options=sess_options)
            else:
                self.net = onnxruntime.InferenceSession(onnx_path_demo, providers=['CPUExecutionProvider'], sess_options=sess_options)   
        except:
            try:
                self.net = onnxruntime.InferenceSession(onnx_path_demo, providers=['CPUExecutionProvider'], sess_options=sess_options)
            except :
                self.net = None
        self.modelNameList = []
            
    # &&&定义信号与槽的连接&&&
    def __set_signal_and_slot(self):
        # # 每次切换图片时是否执行自动检测
        self.ui.autoDetectRB.clicked.connect(self._slot_auto_detect_slot)
        
        self.ui.useGPURB.clicked.connect(self._slot_update_model)
        
        # 检测图片
        self.ui.checkThePicture.clicked.connect(self._slot_check_one_img)

        # # 文件选择的信号与槽
        self.ui.selectFileBtn.clicked.connect(self._slot_open_image)  # 选择图片
        self.ui.previousImgBtn.clicked.connect(self._slot_previous_img)  # 上一张图片
        self.ui.nextImgBtn.clicked.connect(self._slot_next_img)  # 下一张图片

        # # 图片选择信号与槽
        self.ui.selectedFileNameCB.activated[str].connect(self._slot_switch_image_by_cb)   # 用户下拉框改变
        
        # 分割阈值信号与槽
        self.ui.threshold_hS.sliderMoved.connect(self._update_dst_img_slider)

        # 选择模型
        self.ui.selectModelBtn.clicked.connect(self._slot_open_model)
        self.ui.selectModelCB.activated[str].connect(self._slot_switch_model_by_cb)
        
        # 保存标签
        self.ui.pushButton_Save.clicked.connect(self._slot_save_mask)
        
        #
        
        self.ui.BrightnessHSlider.sliderMoved.connect(self._slot_distort_sliders)
        self.ui.ContrastHSlider.sliderMoved.connect(self._slot_distort_sliders)
        self.ui.saturationHSlider.sliderMoved.connect(self._slot_distort_sliders)
        self.ui.HueHSlider.sliderMoved.connect(self._slot_distort_sliders)
        self.ui.sharpnessHSlider.sliderMoved.connect(self._slot_distort_sliders)
        
    # --------------------------------------槽函数定义------------------------------------------------------------------------------------
    # ---滑动条，亮度对比度等
    def _slot_distort_sliders(self):
        # {"brightness":1, "contrast":1, "saturation":1, "hue":0, "sharpness":0}
        sender = self.sender()
        if(sender == self.ui.BrightnessHSlider):
            brightnessDelta = float(self.ui.BrightnessHSlider.sliderPosition())/float(self.ui.BrightnessHSlider.maximum()-self.ui.BrightnessHSlider.minimum())
            brightnessDelta = brightnessDelta * 2
            self.distortMap["brightness"] = brightnessDelta
            self.ui.label_brightness_delta.setText(str(brightnessDelta)[0:6])
            print("brightness!!!!!!!!!!!")
            
        if(sender == self.ui.ContrastHSlider):
            contrastDelta = float(self.ui.ContrastHSlider.sliderPosition())/float(self.ui.ContrastHSlider.maximum()-self.ui.ContrastHSlider.minimum())
            contrastDelta = contrastDelta * 2
            self.distortMap["contrast"] = contrastDelta
            self.ui.label_contrast_delta.setText(str(contrastDelta)[0:6])
            print("ContrastHSlider!!!!!!!!!!!")
            
        if(sender == self.ui.saturationHSlider):
            saturationDelta = float(self.ui.saturationHSlider.sliderPosition())/float(self.ui.saturationHSlider.maximum()-self.ui.saturationHSlider.minimum())
            saturationDelta = saturationDelta * 2
            self.distortMap["saturation"] = saturationDelta
            self.ui.label_saturation_delta.setText(str(saturationDelta)[0:6])
            print("saturationHSlider!!!!!!!!!!!")
            
        if(sender == self.ui.HueHSlider):
            hueDelta = float(self.ui.HueHSlider.sliderPosition())/float(self.ui.HueHSlider.maximum()-self.ui.HueHSlider.minimum())
            hueDelta =  (hueDelta - 0.5) * 240 # [-120, 120]
            self.distortMap["hue"] = hueDelta
            self.ui.label_hue_delta.setText(str(hueDelta)[0:6])
            print("HueHSlider!!!!!!!!!!!")
            
        if(sender == self.ui.sharpnessHSlider):
            sharpnessDelta = float(self.ui.sharpnessHSlider.sliderPosition())/float(self.ui.sharpnessHSlider.maximum()-self.ui.sharpnessHSlider.minimum())
            sharpnessDelta =  (sharpnessDelta - 0.5) * 100 # [-50, 50]
            self.distortMap["sharpness"] = sharpnessDelta
            self.ui.label_sharpness_delta.setText(str(sharpnessDelta)[0:6])
            print("sharpnessHSlider!!!!!!!!!!!")
            
        self._update_source_image_win_byTransform()
    
    # ---保存标签
    def _slot_save_mask(self):
        # 目标二值图
        self.ui.pushButton_Save.setEnabled(False)
        color_map = get_color_map_list(256)
        save_mask_path = self.imgPath
        mask_name = self.currentImgName.split('.')[0] + '.png'
        care = self.result_probMap>=self.ratio
        result = np.zeros(care.shape[1:], dtype=np.uint8) # [h, w]
        for i in range(0, len(care)):
            result[care[i]] = i+1
        lbl_pil = Image.fromarray(result)
        lbl_pil.putpalette(color_map)
        lbl_pil.save(os.path.join(save_mask_path, mask_name))
        self.ui.pushButton_Save.setEnabled(True)
    
    # ----------槽函数，键盘事件,重写父类的方法--------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._slot_previous_img()  # 上一张图片
        if event.key() == Qt.Key_Right:
            self._slot_next_img()

    def resizeEvent(self, a0: QResizeEvent):
        # pass
        self.resizeWindow()
        
    def resizeWindow(self):
        try:
            if self.currentSourceImg.shape[0]>=128:
                w, h = self.ui.srcImgWin.width(), self.ui.srcImgWin.height() 
                # w, h = self.ui.srcImgWin.geometry()[2], self.ui.srcImgWin.geometry()[3]    
                # print(w, h)           
                result =  cv2.resize(self.currentSourceImg, (w, h))
                disp_frame = CommonHelper.numpy_to_QImage(result, format=QImage.Format_RGB888)
                disp_frame = QPixmap.fromImage(disp_frame)
                # disp_frame = disp_frame.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self.ui.srcImgWin.setPixmap(disp_frame)
                
        except:
            pass
        try:
            if self.currentDstImg.shape[0]>=128:
                w, h = self.ui.dstImgWin.width(), self.ui.dstImgWin.height()
                # w, h = self.ui.dstImgWin.geometry()[2], self.ui.dstImgWin.geometry()[3]
                result =  cv2.resize(self.currentDstImg, (w, h))
                disp_frame = CommonHelper.numpy_to_QImage(result, format=QImage.Format_RGB888)
                disp_frame = QPixmap.fromImage(disp_frame)
                self.ui.dstImgWin.setPixmap(disp_frame)
        except:
            pass
        try:
            if self.currentAddImg.shape[0]>=128:
                w, h = self.ui.dstImgWin_2.width(), self.ui.dstImgWin_2.height()
                result =  cv2.resize(self.currentAddImg, (w, h))
                disp_frame = CommonHelper.numpy_to_QImage(result, format=QImage.Format_RGB888)
                disp_frame = QPixmap.fromImage(disp_frame)
                self.ui.dstImgWin_2.setPixmap(disp_frame)
        except:
            pass
    
    # ----------槽函数，勾选自动检测触发--------------------
    def _slot_auto_detect_slot(self, val):
        #  print(self.ui.autoDetectRB.isChecked())
        if self.ui.autoDetectRB.isChecked():
            self._slot_check_one_img()
    
     # 按下检测按钮后进行检测，新开一个线程
    def _slot_check_one_img(self):
        self.ui.checkThePicture.setEnabled(False)  # 避免多次点击
        try:
            path = os.path.join(self.imgPath, self.currentImgName)
            if not os.path.exists(path):
                self.ui.checkThePicture.setEnabled(True)  # 解除检测按钮封印
                return
            if self.net == None:
                print("模型为空！！，请检查")
                self.ui.checkThePicture.setEnabled(True)  # 解除检测按钮封印
                return
            # print("传入线程的Path ", path)
            thread_1 = Thread_1( self.currentTransformImg, self.net)  # 创建线程1
            self.tasksList.append(thread_1)
            thread_1._signal.connect(self._th_slot_destroy_thread_1)         # 线程发出信号连接到槽函数
            thread_1.start()  # 开启线程
        except Exception as e:
            print("开启检测线程失败: ", e)
        
    #  线程正常结束时发出一个信号，销毁检测线程，并取出返回结果。注：仅当子线程正常执行结束才会进入该函数，异常中断不会。
    def _th_slot_destroy_thread_1(self, rev):
        sender = self.sender()
        if len(rev) > 1:
            # print("suceed!!!")
            res = rev[1]
            # print(res.shape)
            self.result_probMap = res # 返回的是numpy数组，掩膜，shape(class,h,w)
            self._update_dst_img(res, 0.5)
            
        if rev[0] == "off":
            sender = self.sender()
            if sender in self.tasksList:
                try:
                    if sender.isRunning():
                        sender.quit()
                        sender.wait()
                    self.tasksList.remove(sender)
                except Exception as e:
                    print(e)
        # print("线程结束")
        self.ui.checkThePicture.setEnabled(True)  # 解除检测按钮封印
    
    # =====更新结果图=======
    def _update_dst_img(self, probMap, mask_ratio=0.5):
       
        self.ui.threshold_hS.setSliderPosition((self.ui.threshold_hS.minimum()+self.ui.threshold_hS.maximum())//2) # 滑动条重置
        self.ui.label_prob_th.setText(str(0.5))
        
        self._update_binary_picture(probMap, mask_ratio) # 显示二值图
        self._update_add_picture(probMap, mask_ratio) # 显示叠加图
        
            
    def _update_dst_img_slider(self):
        w, h = self.ui.dstImgWin.width(), self.ui.dstImgWin.height()
        new_th = self.ui.threshold_hS.sliderPosition()
        hs_min = self.ui.threshold_hS.minimum()
        hs_max = self.ui.threshold_hS.maximum()
        ratio = float(new_th)/(float(hs_max-hs_min))
        self.ui.label_prob_th.setText(str(ratio)) # 设置显示框里面的置信度阈值
        
        self.ratio = ratio
        probMap = self.result_probMap
        self._update_binary_picture(probMap, ratio) # 显示二值图
        self._update_add_picture(probMap, ratio) # 显示叠加图
    
    # 设置语义分割二值图（虽然是彩图，为了显示方便）
    def _update_binary_picture(self, probMap, mask_ratio=0.5):
        care = probMap>=mask_ratio # 0.5作为分割阈值
        h,w = np.shape(probMap)[1:]
        result = np.ones(shape=(h, w, 3), dtype=np.uint8)*255 # 纯白色图片，绘制语义分割纯粹结果，RGB
        len_cmap=len(self.colorMap)
        for i in range(0, len(care)):
            result[care[i]] = self.colorMap[i%len_cmap] # 每一类的语义分割结果赋予不同颜色
        
        self.currentDstImg = result
        w, h = self.ui.dstImgWin.width(), self.ui.dstImgWin.height()
        result =  cv2.resize(result, (w, h))
        disp_frame = CommonHelper.numpy_to_QImage(result, format=QImage.Format_RGB888)
        disp_frame = QPixmap.fromImage(disp_frame)
        try:
            if not disp_frame.isNull():     # 不是空QPixmap才进行设置
                self.ui.dstImgWin.setPixmap(disp_frame)                
        except Exception as e:
            print("设置语义分割二值处理结果出错！:", e)
    
    # 设置叠加图       
    def _update_add_picture(self, probMap, mask_ratio=0.5):
        w, h = self.ui.dstImgWin_2.width(), self.ui.dstImgWin_2.height()
       
        img = self.currentSourceImg.copy()
        img = img.astype(np.float32)/255.0
        
        care = probMap>=mask_ratio # 取掩膜[class, h, w]
        fall_ratio = float(np.count_nonzero(care))/float(care.shape[1]*care.shape[2]) # 更新脱落率
        self.ui.label_fall_ratio.setText(str(fall_ratio)[0:7])
        # 图片和掩膜尺寸必须一致
        if img.shape[0] == care.shape[1] and img.shape[1] == care.shape[2]:
            pass
        else:
            img = cv2.resize(img, (care.shape[2], care.shape[1]))
        
        len_cmap=len(self.colorMap)
        for i in range(0, len(care)):
            img[care[i]] = img[care[i]] + np.array(self.colorMap[i%len_cmap], dtype=np.float32)/255.0
        #img[:,:,1][care] =  img[:,:,1][care]+0.5
        
        img = (img - img.min())/(img.max()-img.min())
        img = (img*255).astype(np.uint8)
        self.currentAddImg = img
        result =  cv2.resize(img, (w, h))
        # result = img
        disp_frame = CommonHelper.numpy_to_QImage(result, format=QImage.Format_RGB888)
        disp_frame = QPixmap.fromImage(disp_frame)
        
        try:
            if not disp_frame.isNull():     # 不是空QPixmap才进行设置
                self.ui.dstImgWin_2.setPixmap(disp_frame)
        except Exception as e:
            print("设置叠加图处理结果出错！:", e)
        

    # ----------槽函数，切换图片,用户界面操作时才会进入该函数----------------------------
    def _slot_switch_image_by_cb(self, text):
        if text == self.currentImgName or text == "":
            return
        else:
            self.currentImgName = text
            self._update_source_image_win()  # 更新窗口显示以及文件名显示
            if self.ui.autoDetectRB.isChecked():
                self._slot_check_one_img()
                
    def _slot_switch_model_by_cb(self, text):
            if text == self.currentModelName or text == "":
                return
            else:
                self.currentModelName = text
                self._slot_update_model()  # 更新模型
                if self.ui.autoDetectRB.isChecked():
                    self._slot_check_one_img()
                    
    #  ------------ 槽函数，选择图片路径或者单个图片，切换图片---------------------
    def _slot_open_image(self):
        # 设置文件扩展名过滤,注意用双分号间隔
        img_type: str
        img_name, img_type = QFileDialog.getOpenFileName(self, "打开图片", "",
                                                       " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        if img_name == "":
            return
        self.imgPath = os.path.split(img_name)[0]  # 待检测图片文件所在的路径
        self.currentImgName = os.path.split(img_name)[1]  # 被选中的图片文件的名字
        NameList = CommonHelper.get_file_list(self.imgPath)  # 获取所有图片名
        self.fileNameList = natsorted(NameList)
        self._update_source_image_win()     # 更新窗口显示的图片
        self._update_images_name_list_cb()  # 更新下拉框里面的文件名列表
        
    def _slot_open_model(self):
        img_type: str
        model_name, img_type = QFileDialog.getOpenFileName(self, "选择模型", "",
                                                       " *.onnx")
        if model_name == "":
            return
        self.modelPath = os.path.split(model_name)[0]  # 模型文件所在的路径
        self.currentModelName = os.path.split(model_name)[1]  # 被选中的模型文件的名字
        self.modelNameList = []
        for item in os.listdir(self.modelPath):
            if item.split('.')[-1] == 'onnx':
                self.modelNameList.append(item)
        self._update_model_name_list_cb()
        self._slot_update_model()
        
    def _slot_update_model(self):
        modelName = self.currentModelName
        path = os.path.join(self.modelPath, modelName)
        try:
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = 4 # 设置每个操作的线程数，以控制并行执行的程度。
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL 	# ORT_SEQUENTIAL ORT_PARALLEL 并行执行模式。在这种模式下，计算图中的操作可以并行执行
            sess_options.inter_op_num_threads = 4 # 设置跨操作的线程数，以控制这些操作之间的并行执行程度。
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = False
            if self.ui.useGPURB.isChecked():
                # sess_options.cuda_device_id = 0
                self.net = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider'],sess_options=sess_options)
            else:
                self.net = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=sess_options)
        except Exception as e:
            print("加载模型失败，尝试重新加载, 异常信息为：\n", str(e))
            try:
                # sess_options.use_cuda = False
                self.net = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=sess_options)
            except Exception as e2:
                # print('error GPU')
                print("加载模型完全失败，模型初始化为空，异常信息为：\n", str(e))
                self.net = None

    # ----------上一张图片----------------  s
    def _slot_previous_img(self):
        self._switch_adjacent_image(-1)
        if self.ui.autoDetectRB.isChecked():
            self._slot_check_one_img()

    # --------- 下一张图片----------------
    def _slot_next_img(self):
        self._switch_adjacent_image(1)
        if self.ui.autoDetectRB.isChecked():
            self._slot_check_one_img()
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&初始化函数&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # &定义必要的全局变量&
    def set_global_variable_dynamic(self):
        self.currentSourceImg = np.ones(shape=(1024,1024), dtype=np.uint8)*255 # 原图
        self.currentTransformImg = np.ones(shape=(1024,1024), dtype=np.uint8)*255 # 变换后的图像
        self.result_probMap = np.ones(shape=(1, 1024,1024), dtype=np.uint8)*255 # 概率图
        
        self.currentDstImg = np.ones(shape=(1024,1024, 3), dtype=np.uint8)*255  # 目标二值图
        self.currentAddImg = np.ones(shape=(1024,1024, 3), dtype=np.uint8)*255  # 目标叠加图
        
    def __set_global_variable(self):
        self.colorMap=[(255,50,70),(0,255,30),(0,56,230),
                    (254,67,101), (252,157,154), (131,175,155),(244,208,0),(220,87,18),
                    (64,116,52),(161,47,47)]
        # self.imgPath = r'C:\Users\hp\Desktop\test\1'  # 图片所在的路径，选择图片之后才设置
        self.imgPath = r'C:\Users\hp\Desktop\tiredata\images'  # 图片所在的路径，选择图片之后才设置
        
        self.currentImgName = "2.jpg"  # 当前图片的名称，和self.imgPath合起来就是图片绝对路径   
        
        self.set_global_variable_dynamic()
        self.fileNameList = []  # 定义图片文件名储存的列表
        self.results_dict = {} # 储存已经检测的结果、相关阈值，每张图片用列表储存
        self.tasksList = []
        
        self.ui.stackedWidget.setCurrentIndex(1)
        self.distortMap = {"brightness":1, "contrast":1, "saturation":1, "hue":0, "sharpness":0}
        self.ratio = 0.5
        self._reset_sliders_labels() # 重置诸多滑动条
    
    # &加载显示默认路径下的图片文件，并设置标签
    def __load_default_img(self):
        # 如果提供的默认路径存在，则打开并显示图片
        if os.path.exists(self.imgPath):
            self._update_source_image_win()
            self.fileNameList = CommonHelper.get_file_list(self.imgPath)    # 刷新图片名列表
            self._update_images_name_list_cb()  # 更新图片名选择框

    # ===============================辅助函数，更新模型、更新图片、更新预训结果显示==========================================================
    
    def _reset_sliders_labels(self):
        self.ui.threshold_hS.setValue((self.ui.threshold_hS.maximum()-self.ui.threshold_hS.minimum())//2) # 设置滑动条到中点
        self.ui.label_prob_th.setText(str(0.5)) # 概率阈值标签
        
        self.ui.BrightnessHSlider.setValue((self.ui.BrightnessHSlider.maximum()-self.ui.BrightnessHSlider.minimum())//2)
        self.ui.label_brightness_delta.setText(str(1))
        
        self.ui.ContrastHSlider.setValue((self.ui.ContrastHSlider.maximum()-self.ui.ContrastHSlider.minimum())//2)
        self.ui.label_contrast_delta.setText(str(1))
        
        self.ui.saturationHSlider.setValue((self.ui.saturationHSlider.maximum()-self.ui.saturationHSlider.minimum())//2)
        self.ui.label_saturation_delta.setText(str(1))
        
        self.ui.HueHSlider.setValue((self.ui.HueHSlider.maximum()-self.ui.HueHSlider.minimum())//2)
        self.ui.label_hue_delta.setText(str(0))
        
        self.ui.sharpnessHSlider.setValue((self.ui.sharpnessHSlider.maximum()-self.ui.sharpnessHSlider.minimum())//2)
        self.ui.label_sharpness_delta.setText(str(0))
        
        self.distortMap = {"brightness":1, "contrast":1, "saturation":1, "hue":0, "sharpness":0} # 重置参数字典
        

    # ***********************更新图片选择框里面的名字**********************************
    def _update_images_name_list_cb(self):
        self.ui.selectedFileNameCB.clear()
        self.ui.selectedFileNameCB.addItems(self.fileNameList)
        self.ui.selectedFileNameCB.setCurrentText(self.currentImgName)  # 设置文件名选择框里面的名字
        
    def _update_model_name_list_cb(self):
        self.ui.selectModelCB.clear()
        self.ui.selectModelCB.addItems(self.modelNameList)
        self.ui.selectModelCB.setCurrentText(self.currentModelName)

    # ************************更新窗口显示的图片*******************************************
    def _update_source_image_win(self):
        self.set_global_variable_dynamic() # 更新临时的全局变量
        
        #  利用QLabel显示图片
        w, h = self.ui.srcImgWin.width(), self.ui.srcImgWin.height() 
        abs_path = os.path.join(self.imgPath, self.currentImgName)
        
        # img = np.array(Image.open(abs_path))
        img = np.ones(shape=(1024,1024), dtype=np.uint8)*255
        try:
            img=cv2.imdecode(np.fromfile(abs_path,dtype=np.uint8),-1)
        except:
            pass
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.currentSourceImg = img
        self.currentTransformImg = self.currentSourceImg.copy()
        result =  cv2.resize(img, (w, h))
        # result  = img
        disp_frame = CommonHelper.numpy_to_QImage(result, format=QImage.Format_RGB888)
        disp_frame = QPixmap.fromImage(disp_frame)
        if not disp_frame.isNull():
              # 缩放图片到显示框一样大小
            self.ui.srcImgWin.clear()
            self.ui.srcImgWin.setPixmap(disp_frame)
            # 设置其它结果窗口空白
            self.ui.dstImgWin.clear()
            self.ui.dstImgWin_2.clear()
            
    def _update_source_image_win_byTransform(self):
        #  利用QLabel显示图片
        w, h = self.ui.srcImgWin.width(), self.ui.srcImgWin.height()         

        self.currentTransformImg = self._transformImg(self.currentSourceImg)
        result =  cv2.resize(self.currentTransformImg, (w, h))
        # result  = img
        disp_frame = CommonHelper.numpy_to_QImage(result, format=QImage.Format_RGB888)
        disp_frame = QPixmap.fromImage(disp_frame)
        if not disp_frame.isNull():
              # 缩放图片到显示框一样大小
            self.ui.srcImgWin.clear()
            self.ui.srcImgWin.setPixmap(disp_frame)
            # 设置其它结果窗口空白
            self.ui.dstImgWin.clear()
            self.ui.dstImgWin_2.clear()
            
    def _transformImg(self, srcImg):
        pil_img = Image.fromarray(srcImg) # 转到PIL，RGB格式
        # self.distortMap = {"brightness":1, "contrast":1, "saturation":1, "hue":0, "sharpness":0}
        if(abs(self.distortMap["brightness"]-1.0)>=0.001):
            pil_img = brightness(pil_img, self.distortMap["brightness"])
            
        if(abs(self.distortMap["contrast"]-1.0)>=0.001):
            pil_img = contrast(pil_img, self.distortMap["contrast"])
            
        if(abs(self.distortMap["saturation"]-1.0)>=0.001):
            pil_img = saturation(pil_img, self.distortMap["saturation"])
            
        if(abs(self.distortMap["hue"]-1.0)>=0.001):
            pil_img = hue(pil_img, self.distortMap["hue"])
            
        if(abs(self.distortMap["sharpness"]-1.0)>=0.001):
            pil_img = sharpness(pil_img, self.distortMap["sharpness"])
            
        return np.array(pil_img)

    # ************************往前或后切换一张图片*******************************************
    def _switch_adjacent_image(self, mode):
        num = len(self.fileNameList)
        # 没有图片或只有一张图片就直接跳过
        if num == 0 or num == 1:
            return
        i = self.fileNameList.index(self.currentImgName)  # 获取现在显示的图片在图片列表中的索引
        if i == 0 and mode == -1:
            return
        if i == len(self.fileNameList) - 1 and mode == 1:
            return
        else:
            self.currentImgName = self.fileNameList[i + mode]  # 先改变当前文件名
            self._update_source_image_win()  # 更新窗口显示以及文件名显示
            self._reset_sliders_labels() # 重置诸多滑动条
            self.ui.selectedFileNameCB.setCurrentText(self.currentImgName)  # 设置文件名选择框里面的名字


        
    # ********************设置ui的函数，内容比较多，就放后面了***************

    def __set_ui(self):
        # 本窗口设置
        base_path = r'E:\code\git\PaddleSeg\qt_dev\v1'
        icon = QIcon();
        icon_img = QPixmap(os.path.join(base_path, "source/1.png"))
        self.setIconSize(QSize(60, 60))
        if not icon_img.isNull():
            icon.addPixmap(icon_img)
        self.setWindowIcon(icon)
        self.setWindowTitle("Segmentation ONNX GUI")
        # self.setWindowOpacity(1)  # 设置窗口透明度
        pe = QPalette()
        pe.setColor(QPalette.Window, QColor(238, 237, 239))  # 设置背景色,rgb
        self.setPalette(pe)

        # 设置样式表
        #  stylefile = os.path.join(basePath, 'style/style.qss')
        #  self.qssstyle = CommonHelper.readQSS(stylefile)
        #  self.setStyleSheet(self.qssstyle)

        # ui控件设置
        # 检测按钮
        self.ui.checkThePicture.setStyleSheet(
            "QPushButton{color:rgb(101,153,26)}"  # 按键前景色，字体颜色
            "QPushButton{background-color:rgb(198,224,205)}"  # 按键背景色,填充颜色
            "QPushButton:hover{color:red}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )
        self.ui.checkThePicture.setIconSize(QSize(30, 30))
        self.ui.checkThePicture.setIcon(
            qta.icon('fa5s.magic',
                     active='fa5s.balance-scale',
                     color='blue',
                     color_active='orange')
        )

        # 上一张按钮
        #  self.ui.previousImgBtn.setLayoutDirection(1)
        self.ui.previousImgBtn.setStyleSheet(
            "QPushButton{color:rgb(255,255,255)}"  # 文字颜色
            #  "font:bold 14px" #  字体
            "QPushButton{background-color:rgb(128,128,255)}"  # 按键背景色
            "QPushButton{border-radius:6px}"  # 圆角半径
        )
        self.ui.previousImgBtn.setIconSize(QSize(30, 30))
        self.ui.previousImgBtn.setIcon(
            qta.icon('fa5s.arrow-circle-left',
                     active='fa5s.balance-scale',
                     color='blue',
                     color_active='orange'))

        # 下一张按钮
        self.ui.nextImgBtn.setLayoutDirection(1)
        self.ui.nextImgBtn.setStyleSheet(
            "QPushButton{color:rgb(255,255,255)}"  # /*文字颜色*/
            "QPushButton{background-color:rgb(128,128,255)}"  # 按键背景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            # "QPushButton{padding-right:40px}"
        )
        self.ui.nextImgBtn.setIconSize(QSize(30, 30))
        self.ui.nextImgBtn.setIcon(
            qta.icon('fa5s.arrow-circle-right', ctive='fa5s.balance-scale',
                     color='blue',
                     color_active='orange'))

       

        # 选择文件按钮
        self.ui.selectFileBtn.setStyleSheet(
            "QPushButton{color:rgb(101,153,26)}"  # 按键前景色，字体颜色
            "QPushButton{background-color:rgb(198,224,205)}"  # 按键背景色,填充颜色
            "QPushButton{border-radius:8px}"  # 圆角半径
        )
        self.ui.selectFileBtn.setIconSize(QSize(30, 30))
        self.ui.selectFileBtn.setIcon(  # 文件选择按钮
            qta.icon('fa5s.folder-open')
        )

     

# # 定义一些辅助函数
class CommonHelper():
    @staticmethod
    def readQSS(style):
        with open(style, "r") as f:
            return f.read()

    @staticmethod
    def get_file_list(dir):
        name_list = os.listdir(dir)
        file_name_list = []
        for item in name_list:
            try:
                if item.split('.')[-1] in ['jpg', 'bmp', 'png', 'jpeg']:
                    file_name_list.append(item)
            except:
                pass
        return file_name_list
    
    @staticmethod
    def numpy_to_QImage(result, format=QImage.Format_RGB888):
        if len(result.shape) == 3:
            h, w, ch = result.shape
            bytesPerLine = ch * w
            disp_frame = QImage(result.data, w, h, bytesPerLine, format)
        else:
            h, w, ch = result.shape[0], result.shape[1], 1
            bytesPerLine = ch * w
            disp_frame = QImage(result.data, w, h, bytesPerLine, QImage.Format_Indexed8)
        return disp_frame
    
def get_color_map_list(num_classes):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.
    Args:
        num_classes (int): Number of classes.
    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map


def brightness(im, brightness_delta=1):
    im = ImageEnhance.Brightness(im).enhance(brightness_delta)
    return im

def contrast(im, contrast_delta=1):
    im = ImageEnhance.Contrast(im).enhance(contrast_delta)
    return im


def saturation(im, saturation_delta=1):
    im = ImageEnhance.Color(im).enhance(saturation_delta)
    return im


def hue(im, hue_delta):
    im = np.array(im.convert('HSV'))
    im[:, :, 0] = im[:, :, 0] + hue_delta
    im = Image.fromarray(im, mode='HSV').convert('RGB')
    return im


def sharpness(im, sharpness_delta):
    im = ImageEnhance.Sharpness(im).enhance(sharpness_delta)
    return im

    
    


  