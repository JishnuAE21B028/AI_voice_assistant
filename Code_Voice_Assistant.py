import openai
import pyttsx3
import gradio as gr
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set the API key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize TTS engine
engine = pyttsx3.init()

# Transcribe audio to text using Whisper (OpenAI version 0.28)
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

# Generate response using GPT, including feedback, translation, and intent recognition
def generate_response(user_input):
    try:
        # Intent recognition for translation requests
        if "translate" in user_input.lower():
            target_language = "Spanish"  # This can be dynamic based on user input
            translation_response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Translate the following sentence to {target_language}: {user_input}",
                max_tokens=100
            )
            translation = translation_response.choices[0].text.strip()
            return f"Translation to {target_language}: {translation}"

        # Intent recognition for grammar correction
        elif "correct" in user_input.lower():
            correction_response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Correct the following sentence for grammar: {user_input}",
                max_tokens=100
            )
            correction = correction_response.choices[0].text.strip()
            return f"Corrected sentence: {correction}"

        # Feedback
        elif "feedback" in user_input.lower():
            return "Great job! Keep practicing your language skills!"

        # General response
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful language assistant."},
                          {"role": "user", "content": user_input}]
            )
            return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# Convert text to speech
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Main function for Gradio interface
def voice_assistant(audio_file):
    try:
        # Transcribe user audio
        user_input = transcribe_audio(audio_file)

        if not user_input or "error" in user_input.lower():
            return f"Transcription failed: {user_input}"

        # Generate GPT response
        assistant_reply = generate_response(user_input)

        # Speak the response
        speak_text(assistant_reply)

        return f"User: {user_input}\nAssistant: {assistant_reply}"
    except Exception as e:
        return f"Error: {e}"

# Gradio Interface
interface = gr.Interface(
    fn=voice_assistant,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    live=True,
    title="AI Voice Assistant"
)

if __name__ == "__main__":
    interface.launch(share=True)
