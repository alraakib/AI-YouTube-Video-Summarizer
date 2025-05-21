import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube
import openai
import os
from pathlib import Path
import tiktoken

# Set up OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_youtube_transcript(url):
    """Extract English transcript from YouTube video"""
    try:
        video_id = YouTube(url).video_id
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return ' '.join([t['text'] for t in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def download_audio(url):
    """Download audio stream from YouTube video"""
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
        if audio_stream:
            output_path = Path("temp_audio")
            output_path.mkdir(exist_ok=True)
            return audio_stream.download(output_path=output_path)
        return None
    except Exception as e:
        st.error(f"Download error: {e}")
        return None

def transcribe_audio(file_path):
    """Transcribe audio using OpenAI Whisper"""
    try:
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript['text']
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def count_tokens(text):
    """Count tokens for GPT-3.5 input limit"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def generate_summary(text, max_tokens=3000):
    """Generate summary using GPT-3.5 with token limit handling"""
    try:
        if count_tokens(text) > max_tokens:
            text = text[:int(max_tokens * 3.5)]  # Approximate character limit
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes video transcripts."},
                {"role": "user", "content": f"Create a detailed summary with key points in bullet format:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="YouTube Video Summarizer", layout="wide")
st.title("🎥 AI-Powered YouTube Video Summarizer")

url = st.text_input("Enter YouTube Video URL:", placeholder="https://youtube.com/...")

if st.button("Generate Summary"):
    if url:
        with st.spinner("Analyzing video..."):
            # Try to get transcript
            transcript = get_youtube_transcript(url)
            
            # If no transcript, try audio transcription
            if not transcript:
                audio_path = download_audio(url)
                if audio_path:
                    transcript = transcribe_audio(audio_path)
                    os.remove(audio_path)  # Cleanup audio file
            
            if transcript:
                # Display transcript
                st.subheader("📝 Full Transcript")
                st.expander("View Transcript").write(transcript)
                
                # Generate and display summary
                st.subheader("📌 AI-Generated Summary")
                summary = generate_summary(transcript)
                if summary:
                    st.write(summary)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("Download Summary", summary, file_name="summary.txt")
                    with col2:
                        st.download_button("Download Transcript", transcript, file_name="transcript.txt")
                else:
                    st.error("Failed to generate summary")
            else:
                st.error("Could not process video. Please check URL or try another video.")
    else:
        st.warning("Please enter a valid YouTube URL")

st.markdown("---")
st.markdown("Built with ❤️ using OpenAI, Streamlit, and Python")
