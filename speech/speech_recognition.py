"""
This module implements the speech recognition part of the application.
Author: Sophie Caroni
Date of creation: 21.04.2024
Last modified on: 21.04.2024
"""

import whisper


def file_audio_to_text(audio_file, model='base'):
    model = whisper.load_model(model)
    result = model.transcribe(audio_file)
    return result['text']


def main():
    text = file_audio_to_text(audio_file="speech_test_audio.mp3")
    print(text)


if __name__ == '__main__':
    main()
