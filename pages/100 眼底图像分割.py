import streamlit as st
import pages.funds.stream_test as segment_seg
from ultralytics import YOLO
import time
import numpy as np
import os
import cv2 as cv
from PIL import Image, ImageDraw


st.set_page_config(page_title="眼底病变分割", page_icon="📈")


st.header("眼底病变分割")
st.write(
    """输入一幅眼底图像，输出眼底病变情况"""
)


img_file_buffer_segment = st.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=2)


# DEMO_IMAGE = "pages/Results/benign (2).png"
# if img_file_buffer_segment is not None:
#     img = cv.imdecode(np.fromstring(img_file_buffer_segment.read(), np.uint8), 1)
#     image = np.array(Image.open(img_file_buffer_segment))
#     file_name = img_file_buffer_segment.name
# else:
#     img = cv.imread(DEMO_IMAGE)
#     image = np.array(Image.open(DEMO_IMAGE))



if img_file_buffer_segment is not None:
    img = cv.imdecode(np.fromstring(img_file_buffer_segment.read(), np.uint8), 1)
    image = np.array(Image.open(img_file_buffer_segment))
    file_name = img_file_buffer_segment.name
else:
    st.stop()

st.text("Original Image")

cols = st.columns(2)
cols[0].image(image, clamp=True, channels='GRAY', use_container_width=True, caption="眼底图像")
#cols[1].image(img,channels='RBG', use_container_width=True)

# predict
# segment_seg.predict(file_name, st)


stream_mask = segment_seg.predict(file_name, st)


st.subheader('Output Image')
row1_cols = st.columns(4)
row1_cols[0].image(stream_mask[0], clamp=True, channels='GRAY', use_container_width=True, caption="EX病变")
row1_cols[1].image(stream_mask[1], clamp=True, channels='GRAY', use_container_width=True, caption="HE病变")
row1_cols[2].image(stream_mask[2], clamp=True, channels='GRAY', use_container_width=True, caption="MA病变")
row1_cols[3].image(stream_mask[3], clamp=True, channels='GRAY', use_container_width=True, caption="SE病变")