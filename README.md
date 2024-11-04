# Voice Assistant with System Control, Weather, and Tool Integration

## Overview
This is a voice-based assistant built to handle a variety of tasks on a Windows system, such as controlling brightness, volume, and system settings. Additionally, it integrates with web services like OpenWeather to provide real-time weather updates, DuckDuckGo for web search, and offers voice-based interaction using Text-to-Speech (TTS) capabilities. 

The system relies on various Python packages to accomplish tasks like noise reduction, transcription, and image processing. It also provides a graphical user interface (GUI) to easily toggle microphone settings.

## Features
- **System Control**: Adjust brightness, volume, mute/unmute, Wi-Fi, Bluetooth, and power settings.
- **Weather Updates**: Fetch real-time weather data and forecasts using OpenWeather API.
- **Web Search**: Retrieve information from the web via DuckDuckGo.
- **Voice Interaction**: Use Text-to-Speech (TTS) for responses and speech recognition for command input.
- **Screenshot and Webcam Capture**: Capture screen or webcam feed for analysis or description using AI.

## Project Structure
- **voice.py**: Main script containing the assistant's logic, including tool routing, code generation, weather data retrieval, and system control functions.
- **tts.py**: TTS streaming functionality with support for playback interruption when user speech is detected.

## Requirements
Create a `requirements.txt` file with the following dependencies:

```
opencv-python-headless python-dotenv sounddevice numpy webrtcvad groq noisereduce duckduckgo-search RestrictedPython mss Pillow geocoder pydub edge-tts screen-brightness-control pycaw requests
```

Install these dependencies by running:
``` bash
pip install -r requirements.txt
```

Update line 193 in case selenium is being used
```python
profile_path = "C:/Users/.../AppData/Local/Google/Chrome/User Data"
```

## Environment Variables
Create a `.env` file in the project root to store API keys and other sensitive information:
```python
GROQ_API_KEY=your_groq_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

## Usage
Run the assistant by executing `voice.py`:
```python
python voice.py
```

### GUI
A simple GUI is provided to control the microphone mute/unmute state. This will start automatically when running the script.

### Commands
The assistant is capable of handling various commands through voice input, such as:
- **System Control**: "Turn up the brightness", "Mute the volume"
- **Weather**: "What’s the weather forecast?", "How is the weather right now?"
- **Web Search**: "Search the web for Python tutorials"
- **Math Calculation**: "Calculate 25 plus 30 divided by 5"
- **Screenshot**: "Take a screenshot of the main screen"
- **Webcam**: "What’s in front of me?"

## Key Functions
### voice.py
- **route_query**: Determines if a user query requires a tool (e.g., calculator, web search) and extracts the required inputs.
- **execute_code**: Executes generated Python code safely in a temporary file, capturing output.
- **adjust_brightness**: Controls screen brightness, either increasing/decreasing or setting to a specified percentage.
- **adjust_volume**: Controls system volume, either increasing/decreasing or setting to a specified level.
- **toggle_wifi** and **toggle_bluetooth**: Enable or disable Wi-Fi and Bluetooth using Windows commands.
- **get_current_weather** and **get_weather_forecast**: Retrieve weather information from OpenWeather API.

### tts.py
- **text_to_speech_streamed**: Streams TTS audio playback from Edge TTS, allowing for interruption if user speech is detected.
- **play_audio**: Plays an `AudioSegment` using `sounddevice`, with support for interruption.

## Additional Notes
- Ensure all necessary API keys are correctly added to the `.env` file.
- The project currently supports Windows-based system commands and may require adaptations for other operating systems.
- Logging is set up for monitoring actions, errors, and results in the system.
