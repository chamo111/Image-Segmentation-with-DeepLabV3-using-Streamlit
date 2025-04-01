# Image-Segmentation-with-DeepLabV3-using-Streamlit

This project is an interactive web app for performing semantic image segmentation using a pre-trained DeepLabV3 model with a ResNet-50 backbone, powered by PyTorch and Streamlit. It detects and segments objects in an uploaded image, highlighting each one against a clean white background for clear visualization.

---

##  Features

-  Upload any image and perform object segmentation instantly  
-  DeepLabV3 model trained on Pascal VOC dataset  
-  Visualizes each detected object separately  
-  Clean and responsive web interface with Streamlit  

---

##  Model Details

The app uses `torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)`, which is trained on the **Pascal VOC** dataset. The model supports 20 object categories:

- `aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`,  
- `dining table`, `dog`, `horse`, `motorbike`, `person`, `potted plant`, `sheep`, `sofa`, `train`, `tv`.

---

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/chamo111/Image-Segmentation-with-DeepLabV3-using-Streamlit
   cd image-segmentation-streamlit

   
