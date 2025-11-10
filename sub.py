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
import colorsys
import random

class SubPage(QMainWindow):
    def __init__(self, parent=None):
        super(SubPage, self).__init__(parent)
        self.ui = Ui_MainWindow()   # Initialize the window object
        self.ui.setupUi(self)       # Initialize the widgets
        self.__set_ui()
        
        self.__set_global_variable()  # Initialize the necessary variables
        self.__set_signal_and_slot()  # Initialize the connections between signals and slots
        self.__load_default_img()
        
        try:
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = 4 
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL 	
            sess_options.inter_op_num_threads = 4 
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = False
            
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
            
    # &&&Define the connections between signals and slots&&&
    def __set_signal_and_slot(self):
        ## Whether to perform automatic detection each time the image is switched
        self.ui.autoDetectRB.clicked.connect(self._slot_auto_detect_slot)
        
        self.ui.useGPURB.clicked.connect(self._slot_update_model)
        
        ## Seg the image
        self.ui.checkThePicture.clicked.connect(self._slot_check_one_img)

        ## Signal and slot for file selection
        self.ui.selectFileBtn.clicked.connect(self._slot_open_image)    # Select an image
        self.ui.previousImgBtn.clicked.connect(self._slot_previous_img) # Previous image
        self.ui.nextImgBtn.clicked.connect(self._slot_next_img)         # Next image

        ## Signal and slot for image selection
        self.ui.selectedFileNameCB.activated[str].connect(self._slot_switch_image_by_cb)   # User dropdown change
        
        ## Signal and slot for segmentation threshold
        self.ui.threshold_hS.sliderMoved.connect(self._update_dst_img_slider)

        ## Select model
        self.ui.selectModelBtn.clicked.connect(self._slot_open_model)
        self.ui.selectModelCB.activated[str].connect(self._slot_switch_model_by_cb)
        
        ## Save Mask
        self.ui.pushButton_Save.clicked.connect(self._slot_save_mask)
        
        self.ui.BrightnessHSlider.sliderMoved.connect(self._slot_distort_sliders)
        self.ui.ContrastHSlider.sliderMoved.connect(self._slot_distort_sliders)
        self.ui.saturationHSlider.sliderMoved.connect(self._slot_distort_sliders)
        self.ui.HueHSlider.sliderMoved.connect(self._slot_distort_sliders)
        self.ui.sharpnessHSlider.sliderMoved.connect(self._slot_distort_sliders)
        
    # --------------------------------------Definition of slot functions------------------------------------------------------------------------------------
    # ---Sliders for brightness, contrast, etc.
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
    
    # ---Save labels
    def _slot_save_mask(self):
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
    
    # ----------Slot functions, keyboard events, and overriding parent class methods--------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._slot_previous_img() 
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
    
    # ----------Slot function triggered by checking automatic detection--------------------
    def _slot_auto_detect_slot(self, val):
        #  print(self.ui.autoDetectRB.isChecked())
        if self.ui.autoDetectRB.isChecked():
            self._slot_check_one_img()
    
    # Perform detection in a new thread after pressing the detect button
    def _slot_check_one_img(self):
        self.ui.checkThePicture.setEnabled(False)  # Prevent multiple clicks
        try:
            path = os.path.join(self.imgPath, self.currentImgName)
            if not os.path.exists(path):
                self.ui.checkThePicture.setEnabled(True)  # Re-enable the detect button
                return
            if self.net == None:
                print("Model is empty!! Please check.")
                self.ui.checkThePicture.setEnabled(True)  # Re-enable the detect button
                return
            
            thread_1 = Thread_1( self.currentTransformImg, self.net) # Create the inference thread
            self.tasksList.append(thread_1)
            thread_1._signal.connect(self._th_slot_destroy_thread_1) # Connect the signal emitted by the thread to the slot function
            thread_1.start()  # 开启线程
        except Exception as e:
            print("Failed to start the detection thread: ", e)
        
    # When the thread finishes normally, emit a signal to destroy the detection thread and retrieve the results. 
    # Note: this function is only entered if the child thread completes successfully; it will not be triggered by an abnormal termination.
    def _th_slot_destroy_thread_1(self, rev):
        sender = self.sender()
        if len(rev) > 1:
            # print("suceed!!!")
            res = rev[1]
            # print(res.shape)
            self.result_probMap = res # The returned data is a NumPy array representing the mask, with shape `(class, h, w)`.
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
        
        self.ui.checkThePicture.setEnabled(True)  # Unlock the detect button
    
    # =====Update the result image=======
    def _update_dst_img(self, probMap, mask_ratio=0.5):
       
        self.ui.threshold_hS.setSliderPosition((self.ui.threshold_hS.minimum()+self.ui.threshold_hS.maximum())//2) # Reset the sliders
        self.ui.label_prob_th.setText(str(0.5))
        
        self._update_binary_picture(probMap, mask_ratio)    # Display the binary image
        self._update_add_picture(probMap, mask_ratio)       # Display the overlay image
        
            
    def _update_dst_img_slider(self):
        w, h = self.ui.dstImgWin.width(), self.ui.dstImgWin.height()
        new_th = self.ui.threshold_hS.sliderPosition()
        hs_min = self.ui.threshold_hS.minimum()
        hs_max = self.ui.threshold_hS.maximum()
        ratio = float(new_th)/(float(hs_max-hs_min))
        self.ui.label_prob_th.setText(str(ratio)) # Set the confidence threshold in the display box
        
        self.ratio = ratio
        probMap = self.result_probMap
        self._update_binary_picture(probMap, ratio) # Display the binary map
        self._update_add_picture(probMap, ratio)    # Display the overlay image
    
    # Set the semantic segmentation binary map (though it's a color image, for easier visualization)
    def _update_binary_picture(self, probMap, mask_ratio=0.5):
        care = probMap>=mask_ratio # Use 0.5 as the segmentation threshold
        h,w = np.shape(probMap)[1:]
        result = np.ones(shape=(h, w, 3), dtype=np.uint8)*255 # Pure white image for drawing the raw semantic segmentation result in RGB
        len_cmap=len(self.colorMap)
        for i in range(0, len(care)):
            result[care[i]] = self.colorMap[i%len_cmap] # Assign a different color to the semantic segmentation result of each class
        
        self.currentDstImg = result
        w, h = self.ui.dstImgWin.width(), self.ui.dstImgWin.height()
        result =  cv2.resize(result, (w, h))
        disp_frame = CommonHelper.numpy_to_QImage(result, format=QImage.Format_RGB888)
        disp_frame = QPixmap.fromImage(disp_frame)
        try:
            if not disp_frame.isNull(): 
                self.ui.dstImgWin.setPixmap(disp_frame)                
        except Exception as e:
            print("Failed to set the binary result of semantic segmentation!:", e)
    
    # Set the overlay image       
    def _update_add_picture(self, probMap, mask_ratio=0.5):
        w, h = self.ui.dstImgWin_2.width(), self.ui.dstImgWin_2.height()
       
        img = self.currentSourceImg.copy()
        img = img.astype(np.float32)/255.0
        
        care = probMap>=mask_ratio # Retrieve the mask with shape [class, h, w]
        fall_ratio = float(np.count_nonzero(care))/float(care.shape[1]*care.shape[2])
        self.ui.label_fall_ratio.setText(str(fall_ratio)[0:7])
        # The image and mask dimensions must match
        if img.shape[0] == care.shape[1] and img.shape[1] == care.shape[2]:
            pass
        else:
            img = cv2.resize(img, (care.shape[2], care.shape[1]))
        
        len_cmap=len(self.colorMap)
        for i in range(0, len(care)):
            img[care[i]] = img[care[i]] + np.array(self.colorMap[i%len_cmap], dtype=np.float32)/255.0
        
        img = (img - img.min())/(img.max()-img.min())
        img = (img*255).astype(np.uint8)
        self.currentAddImg = img
        result =  cv2.resize(img, (w, h))
        # result = img
        disp_frame = CommonHelper.numpy_to_QImage(result, format=QImage.Format_RGB888)
        disp_frame = QPixmap.fromImage(disp_frame)
        
        try:
            if not disp_frame.isNull(): 
                self.ui.dstImgWin_2.setPixmap(disp_frame)
        except Exception as e:
            print("Failed to set the overlay image result!:", e)
        

    # ----------Slot function for switching images, entered only during user interface interactions.----------------------------
    def _slot_switch_image_by_cb(self, text):
        if text == self.currentImgName or text == "":
            return
        else:
            self.currentImgName = text
            self._update_source_image_win()  # Update the window display and the filename display
            if self.ui.autoDetectRB.isChecked():
                self._slot_check_one_img()
                
    def _slot_switch_model_by_cb(self, text):
            if text == self.currentModelName or text == "":
                return
            else:
                self.currentModelName = text
                self._slot_update_model() 
                if self.ui.autoDetectRB.isChecked():
                    self._slot_check_one_img()
                    
    #  ------------ Slot function for selecting an image path or a single image and switching the image.---------------------
    def _slot_open_image(self):
        # Set file extension filters; note to separate them using double semicolons.
        img_type: str
        img_name, img_type = QFileDialog.getOpenFileName(self, "Open image", "",
                                                       " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        if img_name == "":
            return
        self.imgPath = os.path.split(img_name)[0]            # Path of the images to be detected
        self.currentImgName = os.path.split(img_name)[1]     # Name of the selected image file
        NameList = CommonHelper.get_file_list(self.imgPath)  # Retrieve all image filenames
        self.fileNameList = natsorted(NameList)
        self._update_source_image_win()     # Update the image displayed in the window
        self._update_images_name_list_cb()  # Update the list of filenames in the dropdown box
        
    def _slot_open_model(self):
        img_type: str
        model_name, img_type = QFileDialog.getOpenFileName(self, "Select a model", "",
                                                       " *.onnx")
        if model_name == "":
            return
        self.modelPath = os.path.split(model_name)[0]         # Path to the model file
        self.currentModelName = os.path.split(model_name)[1]  # Name of the selected model file
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
            sess_options.intra_op_num_threads = 4
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            sess_options.inter_op_num_threads = 4
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = False
            if self.ui.useGPURB.isChecked():
                # sess_options.cuda_device_id = 0
                self.net = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider'],sess_options=sess_options)
            else:
                self.net = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=sess_options)
        except Exception as e:
            print("Failed to load the model, attempting to reload. Exception message:\n", str(e))
            try:
                # sess_options.use_cuda = False
                self.net = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=sess_options)
            except Exception as e2:
                # print('error GPU')
                print("Failed to fully load the model; model initialization is empty. Exception message:\n", str(e))
                self.net = None

    # ----------Previous image----------------  s
    def _slot_previous_img(self):
        self._switch_adjacent_image(-1)
        if self.ui.autoDetectRB.isChecked():
            self._slot_check_one_img()

    # --------- Next image----------------
    def _slot_next_img(self):
        self._switch_adjacent_image(1)
        if self.ui.autoDetectRB.isChecked():
            self._slot_check_one_img()

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&Initialization function&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # &Define the necessary global variables&
    def set_global_variable_dynamic(self):
        self.currentSourceImg = np.ones(shape=(1024,1024), dtype=np.uint8)*255 # src img
        self.currentTransformImg = np.ones(shape=(1024,1024), dtype=np.uint8)*255 # Transformed image
        self.result_probMap = np.ones(shape=(1, 1024,1024), dtype=np.uint8)*255 # Probability map
        
        self.currentDstImg = np.ones(shape=(1024,1024, 3), dtype=np.uint8)*255  # Target binary map
        self.currentAddImg = np.ones(shape=(1024,1024, 3), dtype=np.uint8)*255  # Target overlay image
        
    def __set_global_variable(self):
        self.colorMap=[(255,50,70),(0,255,30),(0,56,230),
                    (254,67,101), (252,157,154), (131,175,155),(244,208,0),(220,87,18),
                    (64,116,52),(161,47,47)]
        
        # Generate the remaining 245 colors (HSV evenly distributed)
        def generate_colors(n):
            colors = []
            for i in range(n):
                h = i / n
                s = 0.9
                v = 0.9
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                colors.append((int(r * 255), int(g * 255), int(b * 255)))
            return colors

        # 扩展到255个颜色
        self.colorMap.extend(generate_colors(255 - len(self.colorMap)))
        
        self.imgPath = r'C:\Users\hp\Desktop\tiredata\images'  # Path of the image, set only after selecting an image
        
        self.currentImgName = "2.jpg"  #Current image name; combined with self.imgPath, it forms the absolute image path.   
        
        self.set_global_variable_dynamic()
        self.fileNameList = []  # Define the list for storing image file names
        self.results_dict = {}  # Store the detected results and related thresholds. Each image should be stored in a list
        self.tasksList = []
        
        self.ui.stackedWidget.setCurrentIndex(1)
        self.distortMap = {"brightness":1, "contrast":1, "saturation":1, "hue":0, "sharpness":0}
        self.ratio = 0.5
        self._reset_sliders_labels() # Reset many sliders
    
    # &Load the image file displayed in the default path and set the label
    def __load_default_img(self):
        # If the default path provided exists, open and display the image
        if os.path.exists(self.imgPath):
            self._update_source_image_win()
            self.fileNameList = CommonHelper.get_file_list(self.imgPath) # Refresh the list of picture rankings
            self._update_images_name_list_cb()  # Update the picture name selection box

    # ===============================Auxiliary functions: update the model, update the image, and update the pre-training result display==========================================================
    
    def _reset_sliders_labels(self):
        self.ui.threshold_hS.setValue((self.ui.threshold_hS.maximum()-self.ui.threshold_hS.minimum())//2) # Set the slider to the midpoint
        self.ui.label_prob_th.setText(str(0.5)) # Probability threshold label
        
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
        
        self.distortMap = {"brightness":1, "contrast":1, "saturation":1, "hue":0, "sharpness":0} 
        

    # ***********************Update the name in the picture selection box**********************************
    def _update_images_name_list_cb(self):
        self.ui.selectedFileNameCB.clear()
        self.ui.selectedFileNameCB.addItems(self.fileNameList)
        self.ui.selectedFileNameCB.setCurrentText(self.currentImgName)  # Set the name in the file name selection box
        
    def _update_model_name_list_cb(self):
        self.ui.selectModelCB.clear()
        self.ui.selectModelCB.addItems(self.modelNameList)
        self.ui.selectModelCB.setCurrentText(self.currentModelName)

    # ************************Update the picture displayed in the window*******************************************
    def _update_source_image_win(self):
        self.set_global_variable_dynamic() # Update temporary global variables
        
        #  Display images using QLabel
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
            # Scale the picture to the same size as the display box
            self.ui.srcImgWin.clear()
            self.ui.srcImgWin.setPixmap(disp_frame)
            # Set other result Windows blank
            self.ui.dstImgWin.clear()
            self.ui.dstImgWin_2.clear()
            
    def _update_source_image_win_byTransform(self):
        w, h = self.ui.srcImgWin.width(), self.ui.srcImgWin.height()         

        self.currentTransformImg = self._transformImg(self.currentSourceImg)
        result =  cv2.resize(self.currentTransformImg, (w, h))
        # result  = img
        disp_frame = CommonHelper.numpy_to_QImage(result, format=QImage.Format_RGB888)
        disp_frame = QPixmap.fromImage(disp_frame)
        if not disp_frame.isNull():
            self.ui.srcImgWin.clear()
            self.ui.srcImgWin.setPixmap(disp_frame)
            self.ui.dstImgWin.clear()
            self.ui.dstImgWin_2.clear()
            
    def _transformImg(self, srcImg):
        pil_img = Image.fromarray(srcImg) # to PIL，RGB format
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

    # ************************Switch a picture forward or backward*******************************************
    def _switch_adjacent_image(self, mode):
        num = len(self.fileNameList)
        # Skip it directly if there is no picture or only one picture
        if num == 0 or num == 1:
            return
        i = self.fileNameList.index(self.currentImgName)  # Get the index of the currently displayed image in the image list
        if i == 0 and mode == -1:
            return
        if i == len(self.fileNameList) - 1 and mode == 1:
            return
        else:
            self.currentImgName = self.fileNameList[i + mode]  # Change the current file name first
            self._update_source_image_win()  # Update the window display and file name display
            self._reset_sliders_labels() # Reset many sliders
            self.ui.selectedFileNameCB.setCurrentText(self.currentImgName)  # Set the name in the file name selection box


        
    # ********************The function for setting the ui has quite a lot of content, so it's placed later***************

    def __set_ui(self):
        base_path = r'E:\code\git\PaddleSeg\qt_dev\v1'
        icon = QIcon();
        icon_img = QPixmap(os.path.join(base_path, "source/1.png"))
        self.setIconSize(QSize(60, 60))
        if not icon_img.isNull():
            icon.addPixmap(icon_img)
        self.setWindowIcon(icon)
        self.setWindowTitle("Segmentation ONNX GUI")
        pe = QPalette()
        pe.setColor(QPalette.Window, QColor(238, 237, 239))  # Set the background color,rgb
        self.setPalette(pe)


        # "Detection button"
        self.ui.checkThePicture.setStyleSheet(
            "QPushButton{color:rgb(101,153,26)}"  # The view in front of the keys and the font color
            "QPushButton{background-color:rgb(198,224,205)}"  # Key background color, fill color
            "QPushButton:hover{color:red}"  # The foreground color after the cursor is moved above
            "QPushButton{border-radius:6px}"  # Fillet radius
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # The style when pressed
        )
        self.ui.checkThePicture.setIconSize(QSize(30, 30))
        self.ui.checkThePicture.setIcon(
            qta.icon('fa5s.magic',
                     active='fa5s.balance-scale',
                     color='blue',
                     color_active='orange')
        )

        # The previous button
        #  self.ui.previousImgBtn.setLayoutDirection(1)
        self.ui.previousImgBtn.setStyleSheet(
            "QPushButton{color:rgb(255,255,255)}"  # Text color
            "QPushButton{background-color:rgb(128,128,255)}"  # Key background color
            "QPushButton{border-radius:6px}"  # Fillet radius
        )
        self.ui.previousImgBtn.setIconSize(QSize(30, 30))
        self.ui.previousImgBtn.setIcon(
            qta.icon('fa5s.arrow-circle-left',
                     active='fa5s.balance-scale',
                     color='blue',
                     color_active='orange'))

        # Next button
        self.ui.nextImgBtn.setLayoutDirection(1)
        self.ui.nextImgBtn.setStyleSheet(
            "QPushButton{color:rgb(255,255,255)}"  
            "QPushButton{background-color:rgb(128,128,255)}"  
            "QPushButton{border-radius:6px}"  
            # "QPushButton{padding-right:40px}"
        )
        self.ui.nextImgBtn.setIconSize(QSize(30, 30))
        self.ui.nextImgBtn.setIcon(
            qta.icon('fa5s.arrow-circle-right', ctive='fa5s.balance-scale',
                     color='blue',
                     color_active='orange'))

       

        # Select File button
        self.ui.selectFileBtn.setStyleSheet(
            "QPushButton{color:rgb(101,153,26)}"  
            "QPushButton{background-color:rgb(198,224,205)}" 
            "QPushButton{border-radius:8px}" 
        )
        self.ui.selectFileBtn.setIconSize(QSize(30, 30))
        self.ui.selectFileBtn.setIcon(  
            qta.icon('fa5s.folder-open')
        )

     

# # Define some auxiliary functions
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

    
    


  