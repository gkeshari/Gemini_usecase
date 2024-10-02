import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
from pathlib import Path
# from IPython.display import Markdown
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# model = genai.GenerativeModel("gemini-1.5-flash")


def main():
    st.title("Gemini API Demo")
    
    # Sidebar for use case selection
    use_case = st.sidebar.selectbox("Select Use Case", ["Text Conversation", "Image Analysis", "Audio Analysis", "Video Analysis"])
    
    if use_case == "Text Conversation":
        text_conversation()
    elif use_case == "Image Analysis":
        image_analysis()
    elif use_case == "Audio Analysis":
        audio_analysis()
    elif use_case == "Video Analysis":
        video_analysis()

def text_conversation():
    st.header("Text Conversation")
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": "Hello"},
            {"role": "model", "parts": "Great to meet you. What would you like to know?"},
        ]
    )
    
    user_input = st.text_input("Enter your message:")
    if st.button("Send"):
        response = chat.send_message(user_input)
        st.write("Model:", response.text)

def image_analysis():
    st.header("Image Analysis")
    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        user_input = st.text_input("Enter your prompt")
        
        if st.button("Analyze Image"):
            sample_file = Image.open(uploaded_file)
            # myfile = genai.upload_file(uploaded_file)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            result = model.generate_content([user_input,sample_file])
            st.write("Analysis:", result.text)

def audio_analysis():
    st.header("Audio Analysis")
    
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/mp3")
        user_input = st.text_input("Enter your prompt")
        if st.button("Analyze Audio"):
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Read the file content
            # uploaded_file.read() to get the file content directly from the Streamlit UploadedFile object
            file_content = uploaded_file.read()
            
            # Create a request with the audio data
            result = model.generate_content([
                user_input,
                {
                    "mime_type": f"audio/{uploaded_file.type}",  #uploaded_file.type to dynamically set the correct MIME type. This assumes that Streamlit correctly identifies the file type.
                    "data": file_content
                }
            ])
            
            st.write("Analysis:", result.text)

def video_analysis():
    st.header("Video Analysis")
   
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
    if uploaded_file is not None:
        st.video(uploaded_file)
       
        prompt = st.text_input("Enter a prompt for video analysis:")
        if st.button("Analyze Video"):
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Upload the temporary file
                video_file = genai.upload_file(path=tmp_file_path)
                
                # Wait for the file to be processed
                import time
                while video_file.state.name == "PROCESSING":
                    st.write("Processing video...", end='')
                    time.sleep(10)
                    video_file = genai.get_file(video_file.name)
                
                if video_file.state.name == "FAILED":
                    raise ValueError(f"Video processing failed: {video_file.state.name}")

                # Choose a Gemini model
                model = genai.GenerativeModel("gemini-1.5-pro")
               
                # Make the LLM request
                st.write("Analyzing video...")
                response = model.generate_content([prompt, video_file], request_options={"timeout": 600})
                st.write("Analysis:", response.text)
            
            finally:
                # Clean up the temporary file
                os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
