import whisper
import pyaudio
import threading
import numpy as np
import webrtcvad
from collections import deque

# Load the Whisper model
model = whisper.load_model("small.en")

def process_audio_stream():
    # Initialize PyAudio and open a stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    
    vad = webrtcvad.Vad(1)  # Set aggressiveness from 0 to 3 (3 is most aggressive)

    print("Listening for 'what is' in the audio...")

    audio_buffer = np.array([], dtype=np.float32)
    transcript_history = deque(maxlen=4)  # Queue to hold last four transcripts

    try:
        while True:
            # Read data from the microphone
            data = stream.read(4096, exception_on_overflow=False)
            # Split data into 20 ms frames
            n = 320  # 20 ms frame for 16kHz audio
            frames = [data[i:i+n] for i in range(0, len(data), n)]

            for frame in frames:
                if len(frame) < n:
                    continue  # Skip the last incomplete frame

                is_speech = vad.is_speech(frame, 16000)  # Check if the frame contains speech
                if is_speech:
                    audio_array = np.frombuffer(frame, dtype=np.int16)
                    audio_float = (audio_array / 32768.0).astype(np.float32)
                    audio_buffer = np.concatenate((audio_buffer, audio_float))
                    print("Speech detected, processing buffer...")

                    # Process in chunks of about 10 seconds
                    if len(audio_buffer) >= 4000:
                        result = model.transcribe(audio_buffer)
                        transcript = result['text']
                        print("Transcript:", transcript)

                        # Add new transcript to history and check for the phrase
                        transcript_history.append(transcript)
                        if any("what is" in t.lower() for t in transcript_history):
                            print("Detected 'what is' in recent transcripts.")

                        # Clear buffer after processing
                        audio_buffer = np.array([], dtype=np.float32)
                else:
                    print("No speech detected, discarding frame...")

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
