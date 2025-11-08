
# Semantic Segmentation ONNX Inference GUI

A lightweight and user-friendly GUI tool for performing **semantic segmentation** inference using **ONNX models**.  
This application supports models exported from **[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)** as well as models from other deep learning frameworks that are compatible with the ONNX format.

---

## ✨ Features

- 🧠 **ONNX model inference** powered by [ONNX Runtime](https://onnxruntime.ai/)  
- 🖼️ **Interactive GUI** built with [PyQt5](https://pypi.org/project/PyQt5/)  
- ⚙️ **Model-agnostic**: Works with PaddleSeg-exported models and standard ONNX segmentation models  
- 🎨 **Image enhancement controls**:
  - Brightness, contrast, hue, sharpness, and saturation adjustment  
- 🧩 **Flexible preprocessing**:
  - Automatic input shape detection  
  - Dynamic dtype handling (`float32`, `float16`, etc.)  
- 🔍 **Overlay visualization**:
  - Segmentation mask overlay on original image  
  - Adjustable transparency and color mapping  

---

## 🖥️ Demo

![input](assets/example1.png)

---

## 📦 Installation

### 1. Clone the repository
```bash
git https://github.com/chenzhaoqiJM/segmentation-onnx-gui.git
cd segmentation-onnx-gui
```

### 2. Create a Python environment (recommended)

```bash
conda create -n onnx_infer python=3.12
conda activate onnx_infer
```

### 3. Install dependencies

```bash
pip install -r requirements312.txt
```

Please select the requirements file according to your python version

---

## 🚀 Usage

### 1. Launch the GUI

```bash
python main.py
```

### 2. Load your ONNX model

* Click **“Select Model”**
* Select your `.onnx` file (e.g., PaddleSeg exported model)

### 3. Open an image

* Click **“Select Picture”** to choose an input image
* Adjust image enhancements if needed

### 4. Run inference

* Click **“Infer”** to generate segmentation results
* Adjust overlay opacity to compare the result with the original image

---

## 🧩 Model Requirements

* **Input format**: `NCHW` (e.g., `[1, 3, 1024, 1024]`)
* **Data type**: `float32` or `float16`
* **Output**: segmentation mask (class indices or probabilities)

If your model input type differs, the tool automatically adapts preprocessing based on the model’s `input_type`.

---

## 📁 Project Structure

```
.
├── main.py                # Main program entry (GUI main window)
├── ui_files/              # UI-related resources (icons, .ui files, etc.)
├── models/                # Default ONNX models (optional)
├── images/                # Example images
├── assets/                # Resource files (sample images or color maps)
├── cmythread.py           # Preprocessing, inference, and postprocessing code
├── sub.py                 # GUI interaction logic
└── requirementsxxx.txt    # Requirement files for different Python versions
```

---

## ⚙️ Advanced Notes

* Supports **float16** inference when ONNX model expects it
* Automatically handles **channel normalization** and **resizing**
* Compatible with **ONNX models exported from PyTorch**, **TensorFlow**, and **PaddlePaddle**

---

## 🧠 Example: Exporting from PaddleSeg

```bash
python export.py \
    --config configs/deeplabv3p_resnet50_os8_ade20k.yml \
    --model_path output/deeplabv3p.onnx \
    --save_dir export_model
```

Then simply load `export_model/model.onnx` in the GUI.

---

## 📜 License

This project is released under the **MIT License**.
See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a pull request or report a bug.

---

## 📬 Contact

Author: chenzhaoqi
Email: 869948402@qq.com

---


