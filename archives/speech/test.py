import whisper
import pyaudio
import threading
import numpy as np
from collections import deque

# Load the Whisper model
model = whisper.load_model("small.en")

def process_audio_stream():
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

            # Process in chunks of about 10 seconds
            if len(audio_buffer) >= 16000 * 3:  # 3 seconds of audio at 16kHz
                result = model.transcribe(audio_buffer)
                transcript = result['text']
                print("Transcript:", transcript)

                # Add new transcript to history and check for the phrase
                transcript_history.append(transcript)
                if any("what is" in t.lower() for t in transcript_history):
                    print("Detected 'what is' in recent transcripts.")

                # Clear buffer after processing
                audio_buffer = np.array([], dtype=np.float32)

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