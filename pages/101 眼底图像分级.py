import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import pages.funds.stream_test as segment_seg
import pages.grading.stream_test as segment
import time
import numpy as np
import os
import cv2 as cv
from PIL import Image, ImageDraw
import torch



st.set_page_config(page_title="病变分级", page_icon="📊")

st.markdown("# 病变分级")
st.sidebar.header("病变分级")
st.write(
    """输入一幅眼底图像，输出病变等级"""
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

"""
007-2809-100
"""


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

stream_mask = segment_seg.predict(file_name, st)
stream_grading = segment.predict(file_name, st)
numpy_grading = stream_grading.detach().numpy()
st.subheader('Output Image')
row1_cols = st.columns(4)
row1_cols[0].image(stream_mask[0], clamp=True, channels='GRAY', use_container_width=True, caption="EX病变")
row1_cols[1].image(stream_mask[1], clamp=True, channels='GRAY', use_container_width=True, caption="HE病变")
row1_cols[2].image(stream_mask[2], clamp=True, channels='GRAY', use_container_width=True, caption="MA病变")
row1_cols[3].image(stream_mask[3], clamp=True, channels='GRAY', use_container_width=True, caption="SE病变")

row2_cols = st.columns(2)
with row2_cols[0]:
    st.write("**4个病变等级的概率值:**", numpy_grading)  # 直接显示列表

row3_cols = st.columns(2)
row3_cols[0].markdown(f"**病变等级为: {torch.max(stream_grading, dim=1)[1].cpu().data.numpy()[0]}**")














