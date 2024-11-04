import asyncio
import logging
import threading
import numpy as np
import sounddevice as sd
from io import BytesIO
from pydub import AudioSegment
from edge_tts import Communicate  # Assuming edge_tts is being used

# Define the event for stopping playback when user starts speaking
tts_stop_event = threading.Event()

async def text_to_speech_streamed(text, voice="en-US-JennyNeural"):
    """
    Streams TTS audio playback from text using Edge TTS. Playback is interrupted if the user speaks.
    
    Args:
        text (str): The text to be converted to speech.
        voice (str): The voice to use for TTS (default is en-US-JennyNeural).
    """
    try:
        tts_stop_event.clear()  # Ensure the event is cleared at the start
        communicate = Communicate(text, voice=voice)
        audio_stream = BytesIO()  # Buffer to accumulate audio data

        # Stream TTS in chunks
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
            elif chunk["type"] == "text":
                # You can process text chunks here if needed (e.g., real-time transcript display)
                pass

        audio_stream.seek(0)
        audio_segment = AudioSegment.from_file(audio_stream, format="mp3")
        
        # Play audio segment with interruption support
        play_audio(audio_segment)

    except Exception as e:
        logging.error(f"Error during TTS streaming playback: {e}")

def play_audio(audio_segment):
    """
    Plays an AudioSegment using sounddevice, with support for interruption.
    
    Args:
        audio_segment (AudioSegment): The audio segment to play.
    """
    sample_rate = audio_segment.frame_rate
    channels = audio_segment.channels
    samples = np.array(audio_segment.get_array_of_samples())
    
    # Reshape for stereo if needed
    if channels > 1:
        samples = samples.reshape((-1, channels))

    chunk_size = int(sample_rate * 0.1)  # 100ms chunks
    with sd.OutputStream(samplerate=sample_rate, channels=channels, dtype=samples.dtype) as stream:
        index = 0
        while index < len(samples):
            if tts_stop_event.is_set():  # Check for interruption
                logging.info("TTS playback interrupted due to user speech.")
                break
            chunk = samples[index:index + chunk_size]
            stream.write(chunk)
            index += chunk_size
