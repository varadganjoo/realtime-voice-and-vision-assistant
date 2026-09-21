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
- **safety.py**: Tool dispatch, the approval gate for sensitive tools, and the isolated runner for generated code.
- **tts.py**: TTS streaming functionality with support for playback interruption when user speech is detected.

## Requirements
Install the dependencies by running:
``` bash
pip install -r requirements.txt
```

Update line 193 in case selenium is being used
```python
profile_path = "C:/Users/.../AppData/Local/Google/Chrome/User Data"
```

## Environment Variables
Copy `.env.example` to `.env` and add your keys. `.env` is ignored by git.
```
GROQ_API_KEY=your_groq_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

Model IDs are read from the environment so a retired model is a config change, not a code change. Defaults: `llama-3.3-70b-versatile` for code generation and final answers, `llama-3.1-8b-instant` for routing and general chat, `qwen/qwen3.8-27b` for vision, and `whisper-large-v3` for transcription. Override them with `GROQ_TEXT_MODEL`, `GROQ_FAST_MODEL`, `GROQ_VISION_MODEL` and `GROQ_TRANSCRIPTION_MODEL`.

## Safety
The assistant can shut down, restart or sleep the PC, switch Wi-Fi and Bluetooth, and run Python it writes itself. All tool calls go through one dispatcher in `safety.py`, and those six tools need your approval in a Yes/No dialog first. The gate fails closed: no answer within 30 seconds, no GUI, or an error in the dialog all mean "no".

Generated code runs in a separate process with a 60 second timeout, in an empty temporary folder, with API keys and other secrets removed from its environment. That is isolation, not a sandbox: the code still runs as your user account, which is why it also needs your approval.

## Tests
The safety layer is tested without audio, Windows or API keys:
```bash
pip install pytest
pytest
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
- **execute_code**: Runs generated Python in a separate process with a timeout, an empty working folder and no API keys in its environment (see `safety.py`).
- **adjust_brightness**: Controls screen brightness, either increasing/decreasing or setting to a specified percentage.
- **adjust_volume**: Controls system volume, either increasing/decreasing or setting to a specified level.
- **toggle_wifi** and **toggle_bluetooth**: Enable or disable Wi-Fi and Bluetooth using Windows commands.
- **get_current_weather** and **get_weather_forecast**: Retrieve weather information from OpenWeather API.

### safety.py
- **call_tool**: The single dispatch point for tool calls, with the approval gate for sensitive tools.
- **make_tk_confirmer**: Shows the approval dialog on the GUI thread and waits for an answer.
- **execute_code**: The isolated subprocess runner.

### tts.py
- **text_to_speech_streamed**: Streams TTS audio playback from Edge TTS, allowing for interruption if user speech is detected.
- **play_audio**: Plays an `AudioSegment` using `sounddevice`, with support for interruption.

## Additional Notes
- Ensure all necessary API keys are correctly added to the `.env` file.
- The project currently supports Windows-based system commands and may require adaptations for other operating systems.
- Logging is set up for monitoring actions, errors, and results in the system.
