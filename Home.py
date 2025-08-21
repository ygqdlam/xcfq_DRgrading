import streamlit as st

# 参考代码：
# Demo: https://juejin.cn/post/7268955025211342859
# https://blog.csdn.net/weixin_46043195/article/details/132101899
# 侧边栏排序问题 https://blog.csdn.net/weixin_40646871/article/details/136979894

st.set_page_config(
    page_title="医疗图像分割系统",
    page_icon="👋",
)

st.write("# 欢迎使用 医疗图像处理系统! 👋")

st.sidebar.success("在上方选择一个演示。")

st.markdown(
    """
    ### 想了解更多吗？
    - 查看 [streamlit.io](https://streamlit.io)
    - 阅读我们的 [文档](https://docs.streamlit.io)
    - 在我们的 [社区论坛](https://discuss.streamlit.io) 提问
"""
)
