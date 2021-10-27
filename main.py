import numpy as np
import pandas as pd
from keras.models import load_model
import streamlit as st
from PIL import Image,ImageChops

model = load_model("HWD.h5")

def main():
    st.title("Handwriting digit recongization model using python")
    html_temp = """
       <div style="background-color:tomato;padding:10px">
       <h2 style="color:white;text-align:center;">Handwriting digit recongization model using python </h2>
       </div>
       """
    st.markdown(html_temp,unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).resize((100,100)).convert("L")
        image2 = ImageChops.invert(image).resize((28,28))
        x = np.asarray(image2)
        x = x/255.0


        st.image(image2, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = model.predict(x.reshape((1,28,28,1)))
        st.write('%s ' % (np.argmax(label) ))


if __name__ =='__main__':
    main()
