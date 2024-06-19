# Neural Style Transfer Project

## Project Overview
This project aims to transform ordinary photographs into artistic masterpieces by applying the styles of renowned painters like Van Gogh, Picasso, or Monet. This is achieved through a technique called Neural Style Transfer, which uses deep learning to combine the content of one image with the style of another, creating visually appealing and artistically coherent images.

## Repository Structure
```
Neural_Style_Transfer_Project/
│
├── code/
│   ├── Neural_Style_Transfer.py
│   ├── utils.py
│   └── ...
│
├── data/
│   ├── content_images/
│   ├── style_images/
│   └── ...
│
├── results/
│   ├── generated_images/
│   └── ...
│
├── docs/
│   ├── Project_Report.pdf
│   └── ...
│
├── README.md
└── requirements.txt
```

## Installation Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-username/Neural_Style_Transfer_Project.git
    cd Neural_Style_Transfer_Project
    ```

2. **Create a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download VGG19 weights**
    - Download the VGG19 weights file from [here](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5).
    - Place the weights file in the `code` directory.

## Usage

1. **Run the Neural Style Transfer**
    - Open a terminal and navigate to the `code` directory.
    - Run the Streamlit app to start the interactive interface:
    ```bash
    streamlit run Neural_Style_Transfer.py
    ```

2. **Upload Images**
    - Open the Streamlit app in your browser.
    - Upload a content image and a style image.
    - Click the "Generate" button to create the stylized image.

## Dependencies
- Python 3.8+
- TensorFlow
- Keras
- Streamlit
- Pillow
- NumPy

## Challenges Faced

### Model Selection
Initially, different architectures like ResNet50 and InceptionV3 were explored, but they didn't perform as well as VGG19 for style transfer. VGG19's ability to capture detailed hierarchical features at multiple levels proved to be more effective.

### Loss Function Tuning
Finding the right balance between content and style losses was challenging. High weights on style loss resulted in abstract images with less content detail, while high weights on content loss led to insufficient style transfer. Iterative experimentation was required to find optimal weights.

### Computational Intensity
The optimization process is computationally intensive, requiring significant time and resources. This made it challenging to achieve real-time performance. Future work will focus on exploring faster approximation methods.

### Noise Reduction
Achieving a smooth texture in the generated images required careful tuning of the total variation loss. This was essential to reduce noise without compromising the artistic style.

### Online Hosting
Hosting the project online was difficult due to the resource constraint present on platforms like streamlit community servers , also the lack of GPU units makes the generation process very slow .

## Example Usage

```python
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Initialize Streamlit session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

img_size = 400
vgg = tf.keras.applications.VGG19(include_top=False, input_shape=(img_size, img_size, 3), weights='vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
vgg.trainable = False

def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

STYLE_LAYERS = [('block1_conv1', 0.2), ('block2_conv1', 0.2), ('block3_conv1', 0.2), ('block4_conv1', 0.2), ('block5_conv1', 0.2)]
content_layer = [('block5_conv4', 1)]
vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Neural Style Transfer</h1>", unsafe_allow_html=True)
st.divider()

# Implementation details continue...
```

## Acknowledgements
- This project utilizes the VGG19 model pre-trained on the ImageNet dataset.
- Special thanks to the creators of TensorFlow, Keras, and Streamlit for providing the tools necessary to build this project.
- Based on the research paper by Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *Image style transfer using convolutional neural networks*.
