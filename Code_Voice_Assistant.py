import openai
import pyttsx3
import gradio as gr
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()  # Load environment variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")  # Stored as a secret

# Initialize TTS engine
engine = pyttsx3.init()

# Function to transcribe audio to text using Whisper (OpenAI version 0.28)
def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = openai.Audio.transcribe(
                model="whisper-1", 
                file=audio_file
            )
            return transcription.get("text", "Could not transcribe audio.")
    except Exception as e:
        return f"Transcription error: {e}"

# Function to generate response using GPT, including feedback, translation, and intent recognition
def generate_response(user_input):
    try:
        if "translate" in user_input.lower():
            target_language = "Spanish"
            translation_response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Translate the following sentence to {target_language}: {user_input}",
                max_tokens=100
            )
            translation = translation_response.choices[0].text.strip()
            return f"Translation to {target_language}: {translation}"
        elif "correct" in user_input.lower():
            correction_response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Correct the following sentence for grammar: {user_input}",
                max_tokens=100
            )
            correction = correction_response.choices[0].text.strip()
            return f"Corrected sentence: {correction}"
        elif "feedback" in user_input.lower():
            return "Great job! Keep practicing your language skills!"
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful language assistant."},
                          {"role": "user", "content": user_input}]
            )
            return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# Function to convert text to speech
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Main function for Gradio interface
def voice_assistant(audio_file):
    try:
        user_input = transcribe_audio(audio_file)

        if not user_input or "error" in user_input.lower():
            return f"Transcription failed: {user_input}"

        assistant_reply = generate_response(user_input)

        speak_text(assistant_reply)

        return f"User: {user_input}\nAssistant: {assistant_reply}"
    except Exception as e:
        return f"Error: {e}"

# Gradio Interface with updated UI
interface = gr.Interface(
    fn=voice_assistant,
    inputs=gr.Audio(type="filepath", label="🎙️ Upload Your Voice Input"),
    outputs=gr.Textbox(label="📝 Assistant Response", lines=5),
    examples=[["example_audio_1.wav"], ["example_audio_2.wav"]],
    live=True,
    title="🌟 AI Language Assistant",
    description="""
    <div style='text-align: center;'>
        <h2 style='color: #1b5e20; font-family: Arial;'>Welcome to the AI Language Assistant! 🌻</h2>
        <p style='color: #1b5e20; font-size: 16px; font-family: Verdana;'>Practice your language skills with real-time transcription, grammar correction, and translations. Perfect for learners at all levels!</p>
    </div>
    """,
    theme="default",
    css="""
        body { font-family: 'Verdana', sans-serif; background-color: #fff9c4; }
        .interface { border: 1px solid #81c784; border-radius: 10px; padding: 20px; box-shadow: 2px 2px 15px rgba(0, 128, 0, 0.2); }
        h1, h2, p { text-align: center; color: #1b5e20; }
        .output_text { font-weight: bold; color: #1b5e20; }
        .footer { text-align: center; color: #1b5e20; font-size: 14px; margin-top: 20px; }
    """,
    allow_flagging="never",
)

# Footer for the interface
footer = """
<div class='footer'>
    <p>🌟 Powered by OpenAI | Designed for immersive language learning 🌟</p>
</div>
"""

# Launch the interface with the footer
if __name__ == "__main__":
    interface.launch(share=True)
