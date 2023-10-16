import streamlit as st
import img_det
import age_gender_emotion

st.markdown("""
<style>
    .title {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.write('<h1 class="title">Age, Gender, and Emotion Detection App</h1>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("The 'Age, Gender, and Emotion Detection App' is a powerful AI-driven tool that instantly analyzes human faces to determine age, gender, and emotional states. This versatile application finds applications in market research, user experience enhancement, and more, providing valuable insights through real-time facial analysis.")
    st.sidebar.title("Options")

    # Add a choice in the sidebar to select the mode
    app_mode = st.sidebar.selectbox(
        "Choose the detection mode",
        ["Choose an option...", "Webcam", "Image"],
    )

    if app_mode == "Webcam":
        age_gender_emotion.main()
    elif app_mode == "Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "gif", "tiff", "tif"])
        if uploaded_image is not None:
            img_det.load_image_and_predict(uploaded_image)

if __name__ == "__main__":
    main()
