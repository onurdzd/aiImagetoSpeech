from dotenv import find_dotenv,load_dotenv
from transformers import pipeline
import requests
import os 
import streamlit as st

load_dotenv(".env")
HF_API_TOKEN=os.getenv("HF_API_TOKEN")
HF_TEXT_SPEECH_LINK=os.getenv("HF_TEXT_SPEECH_TOKEN")
HF_IMAGE_TEXT_MODEL=os.getenv("HF_IMAGE_TEXT_TOKEN")

def textospeech(message):
    API_URL = HF_TEXT_SPEECH_LINK
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payloads={
        "inputs":message
    }

    response=requests.post(API_URL,headers=headers,json=payloads)
    print(response)
    with open("audio.flac","wb") as file:
        file.write(response.content)


def img2text(url):
    image_to_text = pipeline("image-to-text", model=HF_IMAGE_TEXT_MODEL)

    text=image_to_text(url)

    print(text)
    return text


def main():
    st.set_page_config(page_title="img to audio")

    st.header("Image to audio test")
    uploaded_file=st.file_uploader("choose an image",type="jpg")

    if uploaded_file is not None:
        bytes_data=uploaded_file.getvalue()
        with open(uploaded_file.name,"wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file,caption="Uplaoded image",use_column_width=True)
        scenario=img2text(uploaded_file.name)
        textospeech(scenario[0]['generated_text'])    

        with st.expander("scenario"):
            st.write(scenario)

        st.audio("audio.flac")

if __name__=="__main__":
    main()


# img2text("happy_baby.jpg")

