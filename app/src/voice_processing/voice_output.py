# Importing necessary libraries
import pyttsx3  # Library for text-to-speech conversion
import threading  # Library for handling threads

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Create a lock for thread-safe access to the pyttsx3 engine
engine_lock = threading.Lock()

def speak(text):
    """
    Convert the given text to speech.
    """
    print("//=== speak ===//")
    with engine_lock:  # Ensure thread-safe access to the engine
        engine.setProperty('rate', 300)  # Set the speech rate (words per minute)
        # engine.setProperty('volume', 1)  # Uncomment to set the volume level (1 = max)
        engine.say(text)  # Queue the text to be spoken
        engine.runAndWait()  # Process the speech queue and wait until speaking is finished

def on_volumn():
    """
    Set the volume to maximum (1).
    """
    print('=== on_volumn ===')
    engine.setProperty('volume', 1)  # Set the volume to maximum

def off_volumn():
    """
    Mute the volume (set to 0).
    """
    print('=== off_volumn ===')
    engine.setProperty('volume', 0)  # Set the volume to mute

def set_volumn(volume):
    """
    Set the volume.
    """
    print('=== set_volumn ===')
    # engine.setProperty('volume', 0.5)  # Set the volume to 50%
    engine.setProperty('volume', volume)  # Set the volume to volume

def get_volumn():
    """
    Get the current volume level.
    """
    print('=== get_volumn ===')
    volume = engine.getProperty('volume')  # Get the current volume level
    print(f'current volume: {volume}')
    return volume  # Return the current volume level
