import streamlit as st
from PIL import Image
import os
import sys
import scipy.io
import scipy.misc
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

img_size = 400
vgg = tf.keras.applications.VGG19(include_top=False,input_shape=(img_size, img_size, 3),weights='vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
vgg.trainable = False
def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model
        
STYLE_LAYERS = [('block1_conv1', 0.2),('block2_conv1', 0.2),('block3_conv1', 0.2),('block4_conv1', 0.2),('block5_conv1', 0.2)]
content_layer = [('block5_conv4', 1)]
vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Neural Style Transfer</h1>", unsafe_allow_html=True)
st.divider()

co1, co2, co3 , co4 = st.columns(4)
with co2:
    epochs=st.number_input("Input number of epochs",min_value=200,max_value=20000,step=50)
with co3:
    st.write(" ")
    st.write(" ")
    st.button('Generate Art', on_click=click_button, type="primary",use_container_width=True)  

col1, col2, col3 = st.columns(3)

with col1:
    content_img = st.file_uploader("Input Content Image")
    if content_img is not None:
        content_image = np.array( Image.open(content_img).resize((img_size, img_size)))
        generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
        noise = tf.random.uniform(tf.shape(generated_image), 0, 0.5)
        generated_image = tf.add(generated_image, noise)
        generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
        content_target = vgg_model_outputs(content_image) 
        preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
        a_C = vgg_model_outputs(preprocessed_content)
        a_G = vgg_model_outputs(generated_image)
        st.image(content_img, caption="CONTENT IMAGE", use_column_width=True)
        
with col2:
    style_img = st.file_uploader("Input Style Image")
    if style_img is not None:
        style_image = np.array( Image.open(style_img).resize((img_size, img_size)))
        style_targets = vgg_model_outputs(style_image)
        preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
        a_S = vgg_model_outputs(preprocessed_style)
        st.image(style_img, caption="STYLE IMAGE", use_column_width=True)
        
def compute_content_cost(content_output, generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.transpose(tf.reshape(a_C, shape=[m, -1, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, shape=[m, -1, n_C]))
    J_content =  (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    return J_content

def gram_matrix(A):
    GA = tf.matmul(A, A, transpose_b=True)
    return GA

def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = (1 / (4 * n_C **2 * (n_H * n_W) **2)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    return J_style_layer

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    J_style = 0
    a_S = style_image_output[:-1]
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weight[1] * J_style_layer
    return J_style

@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha * J_content + beta * J_style
    return J



def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        a_G = vgg_model_outputs(generated_image)
        J_style = compute_style_cost(a_S, a_G)
        J_content = compute_content_cost(a_C, a_G)
        J = total_cost(J_content, J_style)
    grad = tape.gradient(J, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    return J

with col3:
    st.write("Generated Image")
    if st.session_state.clicked:
        generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
        for i in range(epochs):
            train_step(generated_image)
            if i % 50 == 0:
                image = tensor_to_image(generated_image)
                st.image(image)
                st.write(f"Epoch {i}")
