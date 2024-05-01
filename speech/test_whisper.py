import whisper
import pyaudio
import threading
import numpy as np
from collections import deque

import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")




def process_audio_stream():

    # Load the Whisper model
    model = whisper.load_model("medium.en")

    # Initialize PyAudio and open a stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)

    print("Listening for 'what is' in the audio...")

    audio_buffer = np.array([], dtype=np.float32)
    transcript_history = deque(maxlen=4)  # Queue to hold last four transcripts

    try:
        while True:
            # Read data from the microphone
            audio_data = stream.read(4096, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Convert from int16 to float32 and normalize
            audio_float = (audio_array / 32768.0).astype(np.float32)
            
            # Append to buffer
            audio_buffer = np.concatenate((audio_buffer, audio_float))

            
            
            # Check if audio buffer is loud enough before processing
            if np.sqrt(np.mean(audio_float**2)) > 0.01:  # Adjust this threshold based on your microphone sensitivity
                # Process in chunks of about 10 seconds
                if len(audio_buffer) >= 16000 * 2:  # 3 seconds of audio at 16kHz
                    result = model.transcribe(audio_buffer)
                    transcript = result['text']
                    print("Transcript:", transcript)

                    # Add new transcript to history and check for the phrase
                    transcript_history.append(transcript)
                    if any("what is" in t.lower() for t in transcript_history):
                        print("Detected 'what is' in recent transcripts.")

                # Clear buffer after processing
                # audio_buffer = np.array([], dtype=np.float32)
            else:
                # If the buffer is not loud enough, consider clearing it or handling it differently
                print("Buffer not loud enough, waiting for more audio...")

            # clear the buffer to prevent memory overflow
            if len(audio_buffer) >= 80000 :
                audio_buffer = np.array([], dtype=np.float32)
                audio_float = np.array([], dtype=np.float32)
                audio_array = np.array([], dtype=np.int16)
                # reinitialize the model
                model = whisper.load_model("medium.en")


    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Close the PyAudio stream
        stream.stop_stream()     
        stream.close()
        p.terminate()

# Start the audio processing in a separate thread
thread = threading.Thread(target=process_audio_stream)
thread.start()
