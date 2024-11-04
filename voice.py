import asyncio
from datetime import datetime
import cv2
from dotenv import load_dotenv
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import webrtcvad
import queue
import threading
import sys
import wave
import io
from groq import Groq
import collections
from noisereduce import reduce_noise
import os
import json  # For JSON handling
import time
from duckduckgo_search import DDGS  # Ensure you have the 'duckduckgo_search' library installed
import logging
import ast
import operator
import re
import traceback
from io import StringIO
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack_sequence, guarded_unpack_sequence
import base64
import subprocess
import tempfile
import tts
import mss
from PIL import Image
import geocoder
from tts import text_to_speech_streamed
# from tts import tts_stop_event

# Additional imports for new functionalities
import screen_brightness_control as sbc  # For brightness control
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # For volume control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# New imports for weather functionalities
import requests  # For making API calls to OpenWeather
from dotenv import load_dotenv  # For loading environment variables

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Groq client securely
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")
client = Groq(api_key=groq_api_key)

# OpenWeather API Key
openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
if not openweather_api_key:
    openweather_api_key = "{YOUR_OPENWEATHER_API_KEY}"  # Replace with your OpenWeather API key

# Parameters
sample_rate = 16000  # 16 kHz
frame_duration_ms = 30  # Frame size in milliseconds
frame_size = int(sample_rate * frame_duration_ms / 1000)  # Number of samples per frame
channels = 1  # Mono audio

# Initialize VAD
vad = webrtcvad.Vad(3)  # Aggressiveness mode (0-3)

# Queue to hold audio frames
audio_queue = queue.Queue()

# Define a flag to control muting
is_muted = False

if not hasattr(tts, 'tts_stop_event'):
    tts.tts_stop_event = threading.Event()

# Pre-buffering before and after speech detection
pre_speech_padding = 300  # Buffer duration before speech in milliseconds
post_speech_padding = 1000  # Increased to 1 second (1000 ms) to allow for pauses in speech
pre_speech_frames = int(pre_speech_padding / frame_duration_ms)
post_speech_frames = int(post_speech_padding / frame_duration_ms)

# Ring buffer to hold pre-speech audio frames
ring_buffer = collections.deque(maxlen=pre_speech_frames)

# File for chat history
chat_history_file = "chat_history.json"

last_tool_name = None
last_tool_output = None
last_llm_response = None

g = geocoder.ip('me')
city = g.city
state = g.state
country = g.country
latlng = g.latlng
latitude = latlng[0]
longitude = latlng[1]

# Flag to stop LLM response streaming
stop_streaming = threading.Event()

# Define Safe Evaluation Function
def safe_eval(expr):
    """
    Safely evaluate a mathematical expression.
    """
    expr = expr.replace("^", "**")
    operators_map = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def eval_(node):
        if isinstance(node, ast.Constant):  # Handles numbers in Python 3.8+
            return node.value
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            return operators_map[type(node.op)](eval_(node.left), eval_(node.right))
        elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
            return operators_map[type(node.op)](eval_(node.operand))
        else:
            raise TypeError(f"Unsupported operation: {type(node).__name__}")

    try:
        node = ast.parse(expr, mode='eval').body
        return eval_(node)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")

def generate_code(prompt, error_message=None):
    """Generate code using the LLaMA model with error feedback and detect Selenium usage."""
    try:
        # Ensure that the prompt is not None or empty
        if not prompt or not prompt.strip():
            logging.error("The prompt for code generation is empty.")
            return "Error: The code generation prompt is empty."
        
        sanitized_chat_history = get_recent_history()
        prompt = f"Chat History:\n{sanitized_chat_history}\n\nCurrent Prompt: {prompt}"

        # Construct the prompt by including any error feedback (if available)
        if error_message:
            prompt = f"{prompt}\nThe last attempt to execute this code failed with the following error:\n{error_message}\nPlease generate a more accurate version of the code to handle the error."

        # Set up the LLaMA model interaction for code generation
        response = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",  # Model for code generation
            messages=[
                {"role": "system", "content": f"You are a code generation assistant on a windows PC. Generate only Python code based on the user's request. Do not include any explanations, comments, or additional text. The system being used is Windows. Do not enclose the code in any formatting or code blocks. The user's location is: {city}, {state}, {country}, {latlng}. Do not use Tesseract, use easyocr instead if needed."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "```python"}
            ],
            stop="```",
            max_tokens=512,
            temperature=0.0,  # Lower temperature for deterministic output
            stream=False
        )

        logging.info(f"Response: {response}")

        if response and response.choices:
            generated_code = response.choices[0].message.content.strip()
            logging.info(f"Generated Code: {generated_code}")

            # Check if Selenium is being used in the generated code
            if "from selenium" in generated_code or "webdriver.Chrome()" in generated_code:
                logging.info("Selenium detected in generated code, injecting profile setup.")
                generated_code = modify_for_selenium_profile(generated_code)

            return generated_code
        else:
            logging.error("Failed to generate code.")
            return "Error: Code generation failed."

    except Exception as e:
        logging.error(f"Error during code generation: {e}")
        return f"Error during code generation: {e}"

def modify_for_selenium_profile(code):
    """Modify the generated Selenium code to include the Chrome profile setup."""
    profile_path = "C:/Users/Varad/AppData/Local/Google/Chrome/User Data"
    
    # Insert the Selenium profile configuration if not already present
    profile_setup_code = f"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

def setup_selenium_with_profile():
    chrome_options = Options()
    chrome_options.add_argument(f"user-data-dir={profile_path}")
    chrome_options.add_argument("profile-directory=Default")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver
"""

    # Find where to inject the setup code in the generated script
    # Replace any occurrences of webdriver.Chrome() with a call to setup_selenium_with_profile()
    modified_code = re.sub(r'webdriver\.Chrome\(\)', 'setup_selenium_with_profile()', code)

    # Combine the profile setup and modified code
    final_code = profile_setup_code + "\n" + modified_code

    return final_code
    
def execute_code(code):
    """Execute the generated Python code safely using a temporary file and capture output."""
    try:
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(code)

        # Execute the temporary file
        try:
            # Run the temporary Python file as a subprocess
            result = subprocess.run(
                [sys.executable, temp_file_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60  # Set a timeout for execution
            )

            # Capture stdout and stderr
            output = result.stdout.strip()
            errors = result.stderr.strip()

            # Combine output and errors
            if output and errors:
                execution_result = f"Output:\n{output}\n\nErrors:\n{errors}"
            elif output:
                execution_result = output
            elif errors:
                execution_result = errors
            else:
                execution_result = "Execution completed with no output."

            if result.returncode != 0:
                logging.error(f"Execution failed with return code {result.returncode}")

        except subprocess.TimeoutExpired:
            execution_result = "Error: Code execution timed out."
        except Exception as e:
            execution_result = f"Error during execution: {e}"
        finally:
            # Delete the temporary file
            os.unlink(temp_file_name)

        return execution_result

    except Exception as e:
        logging.error(f"Error during code execution: {e}")
        return f"Error during code execution: {e}"

# Define Code Generation and Execution Tool
def extract_code_from_response(response_text):
    """
    Extracts the code from the LLM response. It handles code blocks enclosed in triple backticks (
)
    and removes any extra formatting or explanations.
    
    :param response_text: The text response from which to extract the code.
    :return: Extracted code as a string, or None if no code was found.
    """
    # Remove any leading or trailing whitespace
    response_text = response_text.strip()

    # If the response starts and ends with triple backticks, remove them
    if response_text.startswith("```") and response_text.endswith("```"):
        response_text = response_text[3:-3].strip()
        # Remove optional 'python' identifier if present
        if response_text.startswith("python"):
            response_text = response_text[6:].strip()

    # Remove any lines that are not code (e.g., explanations or markdown formatting)
    code_lines = []
    in_code_block = False
    for line in response_text.splitlines():
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or line.strip():  # Only capture non-empty lines or lines in a code block
            code_lines.append(line)

    code = "\n".join(code_lines)
    return code if code else None

def code_generate_and_execute(prompt, retries=3):
    """Generate Python code based on the user's request, execute it, and retry upon failure."""
    attempt = 0
    error_message = None

    while attempt <= retries:
        try:
            # Generate the code based on the given prompt and any error from the last run
            extracted_code = generated_code_response = generate_code(prompt, error_message)

            # Extract the code from the response
            # extracted_code = extract_code_from_response(generated_code_response)
            if not extracted_code:
                return "Error: No valid Python code found in the generated response."

            # Log the extracted code for debugging
            logging.info(f"Extracted Code:\n{extracted_code}")

            # Execute the extracted code
            execution_result = execute_code(extracted_code)

            # Check if there were any errors in the execution result
            if "Error" in execution_result or "Exception" in execution_result:
                error_message = execution_result  # Provide feedback to LLM
                raise RuntimeError(execution_result)  # Treat this as an execution failure

            # If the execution is successful, return the result
            return execution_result

        except Exception as e:
            attempt += 1
            error_message = str(e)  # Capture the error message to provide feedback to LLM
            logging.error(f"Error during code generation and execution (Attempt {attempt}/{retries}): {e}")
            if attempt > retries:
                return f"Error: Failed after {retries} retries. Last error: {e}"

    return f"Error: Maximum retries ({retries}) reached. Could not successfully execute the code."

# Define Tools
def web_search(query):
    """Perform a web search using DuckDuckGo and return a formatted string."""
    try:
        logging.info(f"Performing web search for query: {query}")
        results = DDGS().text(query, max_results=5)  # Limiting to 5 results
        logging.info(f"Web search results: {results}")
        if not results:
            logging.warning("No results found for web search.")
            return "No results found for your query."

        # Extract relevant information from results with safety checks
        formatted_results = []
        for result in results:
            title = result.get("title", "No Title Available")
            href = result.get("href", "No URL Available")
            body = result.get("body", "No description available.")

            # Append formatted result only if title and href are present
            formatted_results.append(f"TITLE: {title}\n   BODY: {body}\n   HREF: {href}")

        # Join all formatted results with a separator
        formatted_response = "\n=======\n".join(formatted_results)

        logging.info(f"Web search results formatted: {formatted_response}")

        return formatted_response

    except Exception as e:
        logging.error(f"Error during web search: {e}")
        return f"Error during web search: {str(e)}"

# Define Calculate Tool
def calculate(expression):
    """Evaluate a mathematical expression safely and return a formatted string."""
    try:
        logging.info(f"Calculating expression: {expression}")
        result = safe_eval(expression)
        logging.info(f"Calculation result: {result}")
        return f"The result of the calculation is: {result}"
    except Exception as e:
        logging.error(f"Error during calculation: {e}")
        return f"Error during calculation: {str(e)}"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def screenshot_tool(screen_number=1):
    """Take a screenshot of the specified screen, analyze it using the Groq Vision AI model, and then delete the local image."""
    try:
        # Validate screen_number
        if not isinstance(screen_number, int) or screen_number < 1:
            screen_number = 1  # Default to screen 1

        with mss.mss() as sct:
            monitors = sct.monitors  # List of monitors; index 0 is the full virtual screen
            if screen_number >= len(monitors):
                logging.warning(f"Screen {screen_number} not found. Defaulting to screen 1.")
                screen_number = 1
            monitor = monitors[screen_number]

            # Capture the screen
            screenshot = sct.grab(monitor)

            # Convert to PIL Image
            img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')

            # Resize the image to reduce size
            max_size = (800, 600)  # Adjust as needed
            img.thumbnail(max_size, Image.LANCZOS)

            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                temp_filename = tmp_file.name
                img.save(tmp_file, format='JPEG', quality=85)

            # Getting the base64 string
            base64_image = encode_image(temp_filename)

            # Create the chat completion request
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this screenshot in detail"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model="llama-3.2-90b-vision-preview",
            )

            # Get the assistant's response using dot notation
            response = chat_completion.choices[0].message.content

            # Delete the local photo
            os.remove(temp_filename)

            return response

    except Exception as e:
        logging.error(f"Error in screenshot_tool: {e}")
        return f"Error in screenshot_tool: {e}"

# New functions for system settings

# Wi-Fi Control Function
def toggle_wifi(state):
    """
    Turn Wi-Fi on or off.
    :param state: Either 'on' or 'off'
    :return: A confirmation message.
    """
    if state.lower() == 'on':
        subprocess.run('netsh interface set interface "Wi-Fi" enabled', shell=True)
        return "Wi-Fi turned on."
    elif state.lower() == 'off':
        subprocess.run('netsh interface set interface "Wi-Fi" disabled', shell=True)
        return "Wi-Fi turned off."
    else:
        return "Invalid state. Use 'on' or 'off'."

# Bluetooth Control Function
def toggle_bluetooth(state):
    """
    Turn Bluetooth on or off using PowerShell.
    :param state: Either 'on' or 'off'
    :return: A confirmation message.
    """
    script = f'''
    $btAdapter = Get-PnpDevice -Class Bluetooth -FriendlyName "*"
    if ("{state}" -eq "on") {{
        Enable-PnpDevice -InstanceId $btAdapter.InstanceId -Confirm:$false
        Write-Output "Bluetooth turned on."
    }} elseif ("{state}" -eq "off") {{
        Disable-PnpDevice -InstanceId $btAdapter.InstanceId -Confirm:$false
        Write-Output "Bluetooth turned off."
    }} else {{
        Write-Output "Invalid state. Use 'on' or 'off'."
    }}
    '''
    result = subprocess.run(["powershell", "-Command", script], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()  # Return PowerShell output
    else:
        return f"Error toggling Bluetooth: {result.stderr.strip()}"

# Brightness Control Function
def adjust_brightness(level):
    """
    Adjust screen brightness.
    :param level: 'up', 'down', or a percentage like '50%'
    :return: A confirmation message.
    """
    try:
        if 'up' in level:
            sbc.set_brightness('+10')
            return "Brightness increased."
        elif 'down' in level:
            sbc.set_brightness('-10')
            return "Brightness decreased."
        else:
            percentage = int(re.findall(r'\d+', level)[0])
            sbc.set_brightness(percentage)
            return f"Brightness set to {percentage}%."
    except Exception as e:
        logging.error(f"Error adjusting brightness: {e}")
        return f"Error adjusting brightness: {str(e)}"

# Volume Control Function
def adjust_volume(level):
    """
    Adjust system volume.
    :param level: 'up', 'down', or a percentage like '50%'
    :return: A confirmation message.
    """
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))

        current_volume = volume.GetMasterVolumeLevelScalar() * 100  # Current volume level

        if 'up' in level:
            new_volume = min(current_volume + 10, 100)
            volume.SetMasterVolumeLevelScalar(new_volume / 100, None)
            return f"Volume increased to {int(new_volume)}%."
        elif 'down' in level:
            new_volume = max(current_volume - 10, 0)
            volume.SetMasterVolumeLevelScalar(new_volume / 100, None)
            return f"Volume decreased to {int(new_volume)}%."
        else:
            percentage = int(re.findall(r'\d+', level)[0])
            volume.SetMasterVolumeLevelScalar(percentage / 100, None)
            return f"Volume set to {percentage}%."
    except Exception as e:
        logging.error(f"Error adjusting volume: {e}")
        return f"Error adjusting volume: {str(e)}"

def capture_and_analyze_image():
    try:
        # Open a connection to the webcam (0 is typically the default camera)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Error: Could not access the webcam."

        # Capture a frame
        ret, frame = cap.read()
        cap.release()  # Release the webcam
        if not ret:
            return "Error: Failed to capture an image."

        # Save the frame as a temporary image file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image_path = tmp_file.name
            cv2.imwrite(image_path, frame)

        # Convert the image to base64
        base64_image = encode_image(image_path)
        os.remove(image_path)  # Clean up the temporary file

        # Use Groq Vision model to analyze the image
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",  # Or use llama-3.2-11b-vision-preview
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is an image from a live webcam feed of Varad. What's in this image? Only describe what is in focus in the image. The background details are not necessary."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.7,
            max_tokens=512,
            stream=False
        )

        # Get the assistant's response from the Groq API
        analysis = response.choices[0].message.content.strip()
        return analysis

    except Exception as e:
        logging.error(f"Error in capture_and_analyze_image: {e}")
        return f"Error analyzing image: {e}"

# Mute/Unmute Function
def toggle_mute(state):
    """
    Mute or unmute system volume.
    :param state: 'mute' or 'unmute'
    :return: A confirmation message.
    """
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))

        if 'mute' in state:
            volume.SetMute(1, None)
            return "Volume muted."
        else:
            volume.SetMute(0, None)
            return "Volume unmuted."
    except Exception as e:
        logging.error(f"Error toggling mute: {e}")
        return f"Error toggling mute: {str(e)}"

# Shutdown Function
def shutdown(_=None):
    """
    Shutdown the system.
    """
    try:
        subprocess.run('shutdown /s /t 1', shell=True)
        return "Shutting down the system."
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")
        return f"Error during shutdown: {str(e)}"

# Restart Function
def restart(_=None):
    """
    Restart the system.
    """
    try:
        subprocess.run('shutdown /r /t 1', shell=True)
        return "Restarting the system."
    except Exception as e:
        logging.error(f"Error during restart: {e}")
        return f"Error during restart: {str(e)}"

# Sleep Function
def sleep(_=None):
    """
    Put the system to sleep.
    """
    try:
        subprocess.run('rundll32.exe powrprof.dll,SetSuspendState 0,1,0', shell=True)
        return "System is going to sleep."
    except Exception as e:
        logging.error(f"Error during sleep: {e}")
        return f"Error during sleep: {str(e)}"

# New functions for weather data

def get_current_weather(lat, lon, api_key):
    """Retrieve current weather data from OpenWeather API."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather_info = {
                'description': data['weather'][0]['description'].capitalize(),
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'pressure': data['main']['pressure'],
                'visibility': data.get('visibility', 0) / 1000  # Convert to kilometers
            }
            return weather_info
        else:
            logging.error(f"Error fetching current weather: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logging.error(f"Exception during fetching current weather: {e}")
        return None

def get_weather_forecast(lat, lon, api_key):
    """Retrieve weather forecast data from OpenWeather API."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            forecasts = []
            for entry in data['list']:
                forecast = {
                    'datetime': entry['dt_txt'],
                    'temperature': entry['main']['temp'],
                    'description': entry['weather'][0]['description'].capitalize(),
                    'humidity': entry['main']['humidity'],
                    'wind_speed': entry['wind']['speed'],
                    'pressure': entry['main']['pressure'],
                    'visibility': entry.get('visibility', 0) / 1000  # Convert to kilometers
                }
                forecasts.append(forecast)
            return forecasts
        else:
            logging.error(f"Error fetching weather forecast: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logging.error(f"Exception during fetching weather forecast: {e}")
        return None

def get_forecast_tool(_=None):
    """Tool function to retrieve and return weather forecast."""
    forecasts = get_weather_forecast(latitude, longitude, openweather_api_key)
    if forecasts:
        forecast_text = f"Weather forecast for {city}, {country}:\n"
        for forecast in forecasts[:5]:  # Limit to the next 5 forecasts
            forecast_text += (f"Date & Time: {forecast['datetime']}\n"
                              f"Temperature: {forecast['temperature']}°C\n"
                              f"Weather: {forecast['description']}\n"
                              f"Humidity: {forecast['humidity']}%\n"
                              f"Wind Speed: {forecast['wind_speed']} m/s\n"
                              f"Pressure: {forecast['pressure']} hPa\n"
                              f"Visibility: {forecast['visibility']} km\n\n")
        return forecast_text
    else:
        return "Unable to retrieve weather forecast at this time."

# Register Available Tools
available_functions = {
    "calculate": calculate,
    "web_search": web_search,
    "code_generate_and_execute": code_generate_and_execute,
    "screenshot_tool": screenshot_tool,
    "toggle_wifi": toggle_wifi,
    "toggle_bluetooth": toggle_bluetooth,
    "adjust_brightness": adjust_brightness,
    "adjust_volume": adjust_volume,
    "toggle_mute": toggle_mute,
    "shutdown": shutdown,
    "restart": restart,
    "sleep": sleep,
    "get_forecast_tool": get_forecast_tool,
    "capture_and_analyze_image": capture_and_analyze_image
}

# Audio Callback Function
def audio_callback(indata, frames, time_info, status):
    global is_muted
    if is_muted:
        return
    if status:
        logging.warning(f"Audio callback status: {status}")
    audio_queue.put(indata.copy())

# Background Noise Reduction
def reduce_background_noise(audio_data):
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    reduced_audio = reduce_noise(y=audio_array, sr=sample_rate)
    return reduced_audio.tobytes()

# Function to handle microphone mute/unmute
def toggle_microphone_mute():
    global is_muted
    is_muted = not is_muted
    status = "Muted" if is_muted else "Unmuted"
    mute_button.config(text=status)
    logging.info(f"Microphone is now {status}")

def start_listening():
    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype='int16', blocksize=frame_size, callback=audio_callback):
        logging.info("Listening... Press Ctrl+C to stop.")
        threading.Thread(target=record_and_transcribe, daemon=True).start()
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            logging.info("Stopping...")
            sys.exit()

# Create the GUI
def create_gui():
    global mute_button
    root = tk.Tk()
    root.title("Voice Assistant Control")

    mute_button = tk.Button(root, text="Unmuted", command=toggle_microphone_mute, font=("Arial", 16), bg="green", fg="white", width=15)
    mute_button.pack(pady=20)

    root.geometry("300x150")
    root.mainloop()

# Chat History Management
def read_chat_history():
    """
    Read chat history from JSON file with UTF-8 encoding.
    If role is 'tool', treat 'tool_output' as 'content' and capture 'tool_call_id'.
    """
    if not os.path.exists(chat_history_file):
        return []

    try:
        with open(chat_history_file, "r", encoding="utf-8") as file:
            data = json.load(file)

            # Process the history entries
            for entry in data:
                if entry["role"] == "tool":
                    # If the role is 'tool', ensure we treat 'tool_output' as 'content'
                    entry["content"] = entry.get("content", "")
                    entry["tool_call_id"] = entry.get("tool_call_id", "")
                else:
                    entry["content"] = entry.get("content", "")
            return data

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logging.error(f"Chat history file is corrupted or has encoding issues: {e}. Starting fresh.")
        # Reset the chat history file with an empty list
        with open(chat_history_file, "w", encoding="utf-8") as file:
            json.dump([], file, ensure_ascii=False, indent=4)
        return []

def sanitise_tool_output(output):
    """
    Sanitize the tool output to escape characters that can cause issues.
    """
    try:
        # Replace problematic curly braces
        output = output.replace('{', '{{').replace('}', '}}')
    
        # Remove or escape other problematic characters if necessary
        output = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', output)
    
        return output
    except Exception as e:
        logging.error(f"Error sanitising tool output: {e}")
        return "Error sanitising output."

def write_to_chat_history(role, content, tool_name=None, tool_args=None, tool_output=None, tool_call_id=None):
    """
    Write a message to chat history in JSON format, ensuring UTF-8 encoding and proper sanitisation.
    Add tool execution details if available, including the tool_call_id.
    """
    history = read_chat_history()

    # If the role is 'tool', format the message as specified
    if role == "tool":
        message = {
            "role": role,
            "content": str(tool_output),  # Save tool_output as content
            "tool_call_id": tool_call_id,  # Save tool_call_id
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_output": tool_output
        }
    else:
        # For other roles, standard message formatting
        message = {"role": role, "content": content}

    try:
        # Append the message to the chat history
        history.append(message)
        # Write the updated chat history to the JSON file
        with open(chat_history_file, "w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=4)
        logging.info(f"Message written to chat history: {message}")
    except Exception as e:
        logging.error(f"Error writing to chat history: {e}")

def get_recent_history(limit=10):
    """Retrieve the chat history as text and sanitize it by escaping braces."""
    if not os.path.exists(chat_history_file):
        return ""

    try:
        with open(chat_history_file, "r", encoding="utf-8") as file:
            chat_history_text = file.read()

            # Replace '{' with '{{' and '}' with '}}' to escape any format strings
            sanitized_chat_history = chat_history_text.replace('{', '{{').replace('}', '}}')

            return sanitized_chat_history

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logging.error(f"Chat history file is corrupted or has encoding issues: {e}. Starting fresh.")
        # Reset the chat history file with an empty list
        with open(chat_history_file, "w", encoding="utf-8") as file:
            json.dump([], file, ensure_ascii=False, indent=4)
        return ""

def get_last_tool_usage():
    """Retrieve the last tool usage from the chat history."""
    history = read_chat_history()
    # Reverse the history to find the last tool usage
    for msg in reversed(history):
        if msg.get("tool_name"):
            logging.debug(f"Found last tool usage: {msg}")
            return msg
    logging.warning("No tool usage found in chat history.")
    return None

# Recording and Transcription
def record_and_transcribe():
    post_speech_counter = 0
    triggered = False
    voiced_frames = []

    while True:
        frame = audio_queue.get()
        if len(frame) == 0:
            continue

        frame_bytes = frame.tobytes()
        is_speech = vad.is_speech(frame_bytes, sample_rate)

        if is_speech:
            if hasattr(tts, 'tts_stop_event'):
                tts.tts_stop_event.set()

            if not triggered:
                triggered = True
                logging.info("Speech detected. Recording started.")
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()

            voiced_frames.append(frame_bytes)
            post_speech_counter = 0
        else:
            if triggered:
                ring_buffer.append(frame_bytes)
                post_speech_counter += 1
                if post_speech_counter > post_speech_frames:
                    logging.info("Speech ended. Stopping recording.")
                    triggered = False
                    audio_data = b''.join(voiced_frames)
                    audio_data = reduce_background_noise(audio_data)
                    transcribed_text = process_audio(audio_data)

                    if transcribed_text:
                        write_to_chat_history("user", transcribed_text)
                        threading.Thread(target=process_query, args=(transcribed_text,), daemon=True).start()

                    voiced_frames = []
                    ring_buffer.clear()
            else:
                ring_buffer.append(frame_bytes)

# Audio Processing and Transcription
def process_audio(audio_data):
    try:
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        wav_io.seek(0)
        file = ("audio.wav", wav_io.read())

        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3",
            response_format="text"
        )
        logging.info(f"Transcription: {transcription.strip()}")
        return transcription.strip()
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return ""

# Routing Functions
def route_query(query):
    """Routing logic to let LLM decide if tools are needed, validate inputs, and extract them for specific tools."""
    sanitized_chat_history = get_recent_history()
    current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    # Get current weather data
    current_weather = get_current_weather(latitude, longitude, openweather_api_key)
    if current_weather:
        weather_description = (f"Current weather: {current_weather['description']}, "
                               f"Temperature: {current_weather['temperature']}°C, "
                               f"Feels like: {current_weather['feels_like']}°C, "
                               f"Humidity: {current_weather['humidity']}%, "
                               f"Wind Speed: {current_weather['wind_speed']} m/s, "
                               f"Pressure: {current_weather['pressure']} hPa, "
                               f"Visibility: {current_weather['visibility']} km.")
    else:
        weather_description = "Current weather data is unavailable."

    system_prompt = f"""Your name is Jenny. You are a smart female assistant. Your user's name is Varad, your builder and your only user. He is a male. Today is {current_datetime}. You are a routing assistant. Use the chat history and the user's query to determine if tools are needed and extract the proper inputs for each tool.

Chat History:
{sanitized_chat_history}

User query: {query}

User's location: {city}, {state}, {country}, {latlng}.
Current time: {current_datetime}
{weather_description}

You are hosted on a Windows PC.

Based on the user's query and the chat history, do the following:
- If a mathematical calculation is required, respond with 'TOOL: CALCULATE {{expression}}', where {{expression}} is the exact mathematical expression to be evaluated.
- If real-time information retrieval (such as news article or facts) is needed from the internet, respond with 'TOOL: WEB_SEARCH {{query}}', where {{query}} is the exact search query.
- If code generation and execution are needed, respond with 'TOOL: CODE_GENERATE_AND_EXECUTE {{prompt}}', where {{prompt}} describes the action.
- If the user asks to look at what is on their screen, respond with 'TOOL: SCREENSHOT_TOOL {{screen_number}}', where {{screen_number}} is the screen number to capture. If the user does not specify a screen number, default to 1.
- If the user wants to adjust system settings like Wi-Fi, Bluetooth, brightness, volume, mute, shutdown, restart, or sleep, respond with 'TOOL: {{tool_name}} {{input}}', where {{tool_name}} is the specific tool and {{input}} is the necessary input.
- If the user wants to know the weather forecast, respond with 'TOOL: GET_FORECAST_TOOL'.
- If the user is trying to show something using a webcam or is saying to have a look at something, respond with 'TOOL: CAPTURE_AND_ANALYZE_IMAGE'.
- The available tools are: CALCULATE, WEB_SEARCH, CODE_GENERATE_AND_EXECUTE, SCREENSHOT_TOOL, TOGGLE_WIFI, TOGGLE_BLUETOOTH, ADJUST_BRIGHTNESS, ADJUST_VOLUME, TOGGLE_MUTE, SHUTDOWN, RESTART, SLEEP, GET_FORECAST_TOOL, CAPTURE_AND_ANALYZE_IMAGE.
- If multiple tools are required, respond with 'TOOLS:' followed by the sequence of tools to use and their inputs.
- For example: 'TOOLS: CALCULATE {{expression}}; WEB_SEARCH {{query}}; TOGGLE_WIFI {{on}}'
- If no tools are needed, respond with 'NO TOOL'.
- The WEB_SEARCH tool is not for audio, images or videos. If the user requests those, use the CODE_GENERATE_AND_EXECUTE tool.
- If the user has asked something where the context is unclear and it wouldn't require any code generation or execution, respond with 'CAPTURE_AND_ANALYZE_IMAGE'
- If the user has asked to control their desktop in any way, for example: opening applications, copying pasting, moving the mouse, etc., use the 'CODE_GENERATE_AND_EXECUTE' tool

Ensure that the response is in uppercase and that you extract the correct input for each tool. Do not mention or call any tool name that is not listed.

Response:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # You may use another model if needed
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=200,  # Ensure there’s enough room for the tool routing and input extraction
            temperature=0,  # Use deterministic behavior to get consistent results
            stream=False
        )

        routing_decision = response.choices[0].message.content.strip().upper()
        logging.info(f"Routing Decision: {routing_decision}")

        tools_with_inputs = []

        if routing_decision.startswith('TOOLS:') or routing_decision.startswith('TOOL:'):
            # Remove 'TOOLS:' or 'TOOL:' and split by semicolon to get individual tool usages
            tools_part = re.sub(r'^(TOOLS|TOOL):\s*', '', routing_decision).strip()
            tools_list = [t.strip() for t in tools_part.split(';') if t.strip()]

            # Parse the tool usages and inputs
            for tool_entry in tools_list:
                # Try to match 'TOOL_NAME {input}' or 'TOOL_NAME input'
                match = re.match(r'(\w+)\s*(?:\{(.*?)\}|(.*))', tool_entry)
                if match:
                    tool_name = match.group(1).lower()
                    tool_input = match.group(2) or match.group(3)
                    # If the tool is 'capture_and_analyze_image', ensure input is None
                    if tool_name == "capture_and_analyze_image":
                        tools_with_inputs.append((tool_name, None))
                    else:
                        if tool_input:
                            tool_input = tool_input.strip()
                        tools_with_inputs.append((tool_name, tool_input))
                else:
                    logging.warning(f"Could not parse tool entry: {tool_entry}")
            return tools_with_inputs
        elif "NO TOOL" in routing_decision:
            return []
        else:
            logging.warning("Could not parse routing decision, defaulting to no tools.")
            return []

    except Exception as e:
        logging.error(f"Error during routing: {e}")
        return []

def generate_user_friendly_response_multi_tool(user_query, tool_results):
    """Generate a user-friendly response using recent tool results and user query context."""
    try:
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        
        # Prepare the context including the recent tool results
        tool_results_text = ""
        for idx, tool_info in enumerate(tool_results, start=1):
            tool_name = tool_info.get("tool_name", "").upper()
            tool_input = tool_info.get("tool_input", "No input provided")
            tool_output = tool_info.get("tool_output", "No output provided")
            
            tool_results_text += f"Tool {idx} ({tool_name}):\nInput: {tool_input}\nOutput: {tool_output}\n\n"

        logging.info(f"Tool Results: {tool_results_text}")
        # Include tool results and recent queries in the system prompt for context
        system_prompt = f"""Your name is Jenny. You are a smart female assistant. You have used tools to get information to solve the user's query. The user has asked the following question:

User query: {user_query}

User's location: {city}, {state}, {country}, {latlng}.
Current time: {current_datetime}
Current weather: {get_current_weather(latitude, longitude, openweather_api_key)}

=====
Here are the tools used and the information you got from them:\n

{tool_results_text}
=====

Try to mostly rely on the tool output to respond to the user.
Please provide a concise, user-friendly response based on the tool results and the user's query. If the user is referring to a recent action or asking to 'try again,' refer to the previous tool's output. Your response should be natural and explain the results in context.
Do not provide a response with code unless the user specifically asks you to show the code. The user is already able to see the code somewhere else."""

        # Send the system prompt and user query to the LLM
        response = client.chat.completions.create(
            model="llama-3.2-11b-text-preview",  # Use the appropriate LLM model
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
            max_tokens=500,  # Adjusted max tokens
            temperature=0.5,  # Slightly increase temperature for more creativity
            stream=False
        )

        # Extract the LLM response
        llm_response = response.choices[0].message.content.strip()

        # Save the final response to the chat history
        write_to_chat_history("assistant", llm_response)

        logging.info(f"User-friendly LLM response: {llm_response}")
        return llm_response

    except Exception as e:
        logging.error(f"Error generating user-friendly response: {e}")
        return f"An error occurred while generating the response: {e}"

def run_with_tool(query):
    global last_tool_name, last_tool_output, last_llm_response
    tools_with_inputs = route_query(query)
    logging.info(f"Route decision: {tools_with_inputs}")

    tool_results = []
    if tools_with_inputs:
        for tool_name, validated_input in tools_with_inputs:
            tool_result = None
            if tool_name in available_functions:
                try:
                    function_to_call = available_functions[tool_name]
                    tool_result = function_to_call(validated_input) if validated_input else function_to_call()
                    write_to_chat_history("tool", tool_result, tool_name=tool_name, tool_args={"input": validated_input}, tool_output=tool_result)
                except Exception as e:
                    logging.error(f"Error during {tool_name}: {e}")
                    tool_result = f"Error during {tool_name}: {e}"
            else:
                tool_result = f"Unknown tool: {tool_name}"
            tool_results.append({"tool_name": tool_name, "tool_input": validated_input, "tool_output": tool_result})

        return generate_user_friendly_response_multi_tool(query, tool_results)
    else:
        return run_general(query)

# General-Purpose Model Interaction
def run_general(query):
    """Use the general model to answer the query including context from history."""
    try:
        # Retrieve the sanitized chat history text
        sanitized_chat_history = get_recent_history()

        # Get the current date and time
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

        # Get current weather data
        current_weather = get_current_weather(latitude, longitude, openweather_api_key)
        if current_weather:
            weather_description = (f"Current weather: {current_weather['description']}, "
                                   f"Temperature: {current_weather['temperature']}°C, "
                                   f"Feels like: {current_weather['feels_like']}°C, "
                                   f"Humidity: {current_weather['humidity']}%, "
                                   f"Wind Speed: {current_weather['wind_speed']} m/s, "
                                   f"Pressure: {current_weather['pressure']} hPa, "
                                   f"Visibility: {current_weather['visibility']} km.")
        else:
            weather_description = "Current weather data is unavailable."

        # Prepare the system prompt including the chat history
        system_prompt = f"""Your name is Jenny. You are a smart female assistant. Your user's name is Varad, your builder and your only user. He is a male. Today is {current_datetime}. The user's location is: {city}, {state}, {country}, {latlng}. {weather_description} You are a helpful assistant.

Chat History:
{sanitized_chat_history}
"""

        # Prepare the messages list with the system prompt
        messages = [{"role": "system", "content": system_prompt}]

        # Add the user's query to the messages
        sanitized_query = sanitise_tool_output(query)
        messages.append({"role": "user", "content": sanitized_query})

        # Make the request to the general-purpose model with the prepared messages
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # General-purpose model
            messages=messages,
            max_tokens=1024,
            stream=False
        )

        # Extract the response from the LLM and save it to the chat history
        final_response = response.choices[0].message.content.strip()
        logging.info(f"LLM Response: {final_response}")
        write_to_chat_history("assistant", final_response)
        return final_response

    except Exception as e:
        logging.error(f"Error during general model interaction: {e}")
        return "An error occurred while processing your request."

# Query Processing with Routing and Memory Check
def process_query(query):
    """
    Processes the user's query by routing it to the appropriate tools or directly to the LLM.
    If the query requires a tool (like calculation, web search, or system control), the tool is invoked,
    and the result is passed to the user-friendly response generator. If no tool is needed, the query 
    is sent directly to the LLM. The final response is then passed to TTS for playback.
    
    Args:
        query (str): The user's query or command.
    
    Returns:
        str: The final response from either the tool results or the LLM response.
    """
    # Run tool routing with query to determine if tools are needed and gather validated inputs
    tools_with_inputs = route_query(query)
    logging.info(f"Route decision: {tools_with_inputs}")
    
    # Initialize list to capture each tool's results, for use in generating a user-friendly response
    tool_results = []

    # Step 1: Check if tools are needed and execute them if specified
    if tools_with_inputs:
        for tool_name, validated_input in tools_with_inputs:
            tool_result = None
            # Verify the tool exists in available functions
            if tool_name in available_functions:
                try:
                    # Execute the tool function, passing the validated input if required
                    function_to_call = available_functions[tool_name]
                    tool_result = function_to_call(validated_input) if validated_input else function_to_call()
                    logging.info(f"{tool_name} result: {tool_result}")

                    # Log tool execution to chat history
                    write_to_chat_history(
                        "tool",
                        tool_result,
                        tool_name=tool_name,
                        tool_args={"input": validated_input},
                        tool_output=tool_result
                    )

                except Exception as e:
                    logging.error(f"Error during {tool_name}: {e}")
                    tool_result = f"Error during {tool_name}: {e}"
            else:
                # Log any unknown tools (should be unlikely with proper routing)
                tool_result = f"Unknown tool: {tool_name}"
                logging.warning(f"Unknown tool requested: {tool_name}")

            # Add the tool's name, input, and output to results list for user-friendly response generation
            tool_results.append({
                "tool_name": tool_name,
                "tool_input": validated_input,
                "tool_output": tool_result
            })

        # Step 2: Generate a user-friendly response based on tool results
        final_response = generate_user_friendly_response_multi_tool(query, tool_results)
    
    else:
        # Step 3: If no tools are needed, send query directly to the general-purpose LLM
        final_response = run_general(query)
    
    # Step 4: Pass the final response to the TTS for playback, allowing interruption if user starts speaking
    try:
        asyncio.run(text_to_speech_streamed(final_response))
    except Exception as e:
        logging.error(f"Error during TTS playback: {e}")

    # Return the final response for logging or further processing if needed
    return final_response

# Main Function to Start Listening
def main():
    gui_thread = threading.Thread(target=create_gui)
    gui_thread.daemon = True
    gui_thread.start()
    start_listening()

if __name__ == '__main__':
    main()