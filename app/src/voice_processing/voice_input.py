# Importing necessary libraries
import speech_recognition as sr  # Library for speech recognition
from src.voice_processing.voice_output import speak  # Import the speak function for voice output
import threading  # Library for handling threads

# Create a lock for thread-safe access to the speech recognition engine
engine_lock = threading.Lock()


def listen_for_commands():
    """
    Listen for voice commands using the microphone and return the recognized command.
    """

    print("//=== listen_for_commands ===//")
    # Initialize the speech recognizer and microphone
    with engine_lock:  # Ensure thread-safe access to the recognizer and microphone
        recognizer = sr.Recognizer()  # Initialize the speech recognizer
        microphone = sr.Microphone()  # Initialize the microphone

        with microphone as source:
            # print("listening for command ...")
            # speak("listening for command ...")

            # Adjust for ambient noise to improve recognition accuracy
            recognizer.adjust_for_ambient_noise(source)

            # Listen for audio input with a timeout and phrase time limit
            audio = recognizer.listen(source)
            # audio = recognizer.listen(source, timeout=1)
            # audio = recognizer.listen(source, timeout=3, phrase_time_limit=3, stream=False)

            try:
                # Use Google's speech recognition API to recognize the audio
                command = recognizer.recognize_google(audio, language="en-US")
                # command = recognizer.recognize_assemblyai(audio, api_token="YOUR_API_TOKEN")
                # command = recognizer.recognize_ibm(audio,key="YOUR_IBM_API_KEY", language="en-US")
                # command = recognizer.recognize_azure(audio, key="YOUR_AZURE_API_KEY", language="en-US")
                # command = recognizer.recognize_amazon(audio)
                # command = recognizer.recognize_bing(audio, key="YOUR_BING_API_KEY", language="en-US")
                print(f"command: {command}")
                # speak("ok")
                return command.lower()
                # return command

            except sr.UnknownValueError:
                # Handle case where the speech is not understood
                print("unknown")
                return None
            except sr.WaitTimeoutError:
                # Handle case where no speech is detected within the timeout
                print("timeout")
                return None
            except sr.RequestError:
                # Handle case where there is an issue with the recognition service
                print("error")
                return None
            except Exception as e:
                # Handle any other exceptions
                print("error: ", str(e))
                return None

def stop_listening():
    """
    Stop listening for voice commands.
    """
    print("//=== stop_listening ===//")
    # Stop listening logic here if needed
    pass

def process_command():
    """
    Process the recognized voice command.
    """
    print("//=== process_command ===//")
    # Placeholder for processing logic
    # command = listen()
    # if command:
    #     print(f"Voice Command: {command}")
    #     return command.lower()
    # return None

def process_voice_commands(command):
    """
    Process the recognized voice command.
    """
    print("//=== process_voice_commands ===//")
    if command:
        print(f"Processing command: {command}")
        # Add logic to process the command and execute actions
        # Example: if "capture" in command:
        #     self.execute_command("capture", detections)
    else:
        print("No command recognized.")

def process_detections(detections):
    """
    Process the detected objects and execute commands based on voice input.
    """
    print("//=== process_detections ===//")
    if detections:
        print(f"Detected objects: {detections}")
        # Add logic to process detections and execute commands
        # Example: if "person" in detections:
        #     self.execute_command("alert", detections)
    else:
        print("No objects detected.")

def execute_command(command, detections, video_stream):
    """
    Execute the command based on the recognized voice command.
    Args:
        command (str): The recognized voice command.
        detections (list): List of detected objects.
        video_stream: The video stream object (if applicable).
    """
    print(f"Executing command: {command}")
    # Add logic to execute the command
    # if command == "exit":
    #     print("Exiting...")
    #     print('video_stream := ', video_stream)
    #     video_stream.release()  # Release the video stream if applicable
    #     cv2.destroyAllWindows()  # Close all OpenCV windows
    #     sys.exit(0)  # Exit the program with a success status
