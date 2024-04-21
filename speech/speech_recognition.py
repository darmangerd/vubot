"""
This module implements the speech recognition part of the application.
Author: Sophie Caroni
Date of creation: 21.04.2024
Last modified on: 21.04.2024
"""

import whisper
import sounddevice as sd
import numpy as np


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status)
    global audio_buffer
    audio_buffer.extend(indata.copy())


def live_audio_to_text(model='base'):
    global audio_buffer
    audio_buffer = []

    # Load the Whisper model
    model = whisper.load_model(model)

    try:
        # Start recording from the default microphone at 16000 Hz
        with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback):
            print("Recording started. Speak into your microphone.")
            input("Press Enter to stop recording...")

        # Convert buffer to np array
        audio_np = np.concatenate(audio_buffer, axis=0)

        # Transcribe the recorded audio
        result = model.transcribe(audio_np, temperature=0)
        return result['text']

    except Exception as e:
        print("An error occurred:", e)
        return None


def file_audio_to_text(audio_file, model='base'):
    model = whisper.load_model(model)
    result = model.transcribe(audio_file)
    return result['text']


def main():
    text = live_audio_to_text()
    print("Transcribed text:", text)


if __name__ == '__main__':
    main()
