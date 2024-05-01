# Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")

# implement using microphone in live audio
import sounddevice as sd
import numpy as np
import queue
import torch

def live_rec_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status)
    global audio_buffer
    audio_buffer.extend(indata.copy())

def live_audio_to_text():
    global audio_buffer
    audio_buffer = []

    try:
        # Start recording from the default microphone at 16000 Hz
        with sd.InputStream(samplerate=16000, channels=1, callback=live_rec_callback):
            print("Recording started. Please speak into your microphone and stop the recording once done.")
            input("Press Enter to stop recording...")

        # Convert buffer to np array
        audio_np = np.concatenate(audio_buffer, axis=0)

        # Transcribe the recorded audio
        inputs = processor(audio_np, return_tensors="pt", padding="longest")
        with torch.no_grad():
            result = model(inputs).logits

        return processor.batch_decode(result, skip_special_tokens=True)[0]

    except Exception as e:
        print("An error occurred:", e)
        return None
    

def main():
    print(live_audio_to_text())

if __name__ == "__main__":
    main()