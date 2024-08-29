import streamlit as st
import pickle
from PIL import Image

def main():
    st.title(":rainbow[DIABATIC FAILURE PREDICTION]")
    image = Image.open('image.jpg')
    st.image(image,width = 800)

    Pregnancies = st.text_input(":red[Pregnancies]","Type_here...")
    Glucose = st.text_input(":green[Glucose]","Type_here...")
    Bloodpressure = st.text_input(":blue[Bloodpressure]","Type_here...")
    SkinThickness = st.text_input(":orange[SkinThickness]","Type_here...")
    Insulin = st.text_input(":violet[Insulin]","Type_here...")
    BMI = st.text_input(":blue[BMI]","Type_here...")
    DiabetesPedigreeFunction = st.text_input(":green[DiabetesPedigreeFunction]","Type_here...")
    Age = st.text_input("Age","Type_here...")

    features = [Pregnancies,Glucose,Bloodpressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]

    model = pickle.load(open('model.sav','rb'))
    scaler = pickle.load(open('scaler.sav','rb'))


    pred = st.button('PREDICT')

    if pred:
        prediction = model.predict(scaler.transform([features]))

        if prediction==0:
            st.write('Not suffering heart Disease')

        else:
            st.write('Suffering heart Disease')


main()