import pyttsx3

def voice_alert(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()
    print(f"Voice alert: {message}")


if __name__ == "__main__":
    # voice_alert("Traffic sign detected: Stop sign")
    voice_alert("Stop sign")