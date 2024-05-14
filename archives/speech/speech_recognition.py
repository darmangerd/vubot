"""
This module implements the speech recognition part of the application.
Author: Sophie Caroni
Date of creation: 21.04.2024
Last modified on: 21.04.2024
"""

import whisper
import sounddevice as sd
import numpy as np
import queue


def live_rec_callback(indata, frames, time, status):
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
        with sd.InputStream(samplerate=16000, channels=1, callback=live_rec_callback()):
            print("Recording started. Please speak into your microphone and stop the recording once done.")
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


def live_audio_callback(indata, frames, time, status):
    """Put microphone data into a queue."""
    if status:
        print(status)
    global audio_queue
    audio_queue.put(indata.copy())


def real_time_transcribe():
    # Initialize the Whisper model
    model = whisper.load_model("base")

    # Define the sample rate and block size
    sample_rate = 16000
    block_size = 16000  # number of audio samples collected before the buffer is transcribed - 8000 processes every 0.5 s

    # Create a queue to handle real time audio data
    global audio_queue
    audio_queue = queue.Queue()

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=live_audio_callback, blocksize=block_size):
            print("Real-time transcription started. Please speak into your microphone.")
            while True:
                audio_chunk = audio_queue.get()
                if audio_chunk is not None:
                    # Convert audio chunk to numpy array
                    audio_np = np.concatenate(audio_chunk)
                    # Transcribe audio
                    result = model.transcribe(audio_np, temperature=0)
                    print(result['text'])


    except KeyboardInterrupt:
        print("Exiting...")

    except Exception as e:
        print("An error occurred:", e)


def main():
    # Test microphone recording speech recognition
    # text = live_audio_to_text()
    # print("Transcribed text:", text)

    # Test real time speech recognition
    real_time_transcribe()


if __name__ == '__main__':
    main()
