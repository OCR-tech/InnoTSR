# Importing necessary libraries
from voice_processing.voice_output import speak  # Import the speak function for voice output


#//==================================//
class voice_processing:
    """
    A class for processing voice commands and generating appropriate responses.
    """

    def __init__(self, data):
        """
        Initialize the voice_processing class.
        """
        self.data = data  # Store the input data for processing

        # Uncomment the following line for debugging during initialization
        # print('=== voice_processing_init ===')
        # self.voice_processing_init()

    def voice_alert(self, detector_instance):
        """
        Process the data from the detector instance and generate voice alerts.
        """

        print('=== voice_alert ===')
        # Extract the data list from the detector instance
        list = self.data
        print('list:=', list)  # Print the list for debugging

        # Define keyword lists for different commands
        keyword_list1 = ['stop sign', 'stop sign1', 'stop sign2', 'stop sign3']     # Keywords for "Stop" command
        keyword_list2 = ['bicycle', 'bicycle1', 'bicycle2', 'bicycle3']             # Keywords for "Alert" command
        keyword_list3 = ['motorcycle', 'motorcycle1', 'motorcycle2', 'motorcycle3'] # Keywords for "Turn left" command
        keyword_list4 = ['bus', 'bus1', 'bus2:', 'bus3']                            # Keywords for "Turn right" command
        keyword_list5 = ['person', 'person1', 'person2', 'person3']                 # Keywords for "Go straight" command

        # Check if any keyword from the lists matches the detected words
        if any(word in list for word in keyword_list1):
            print('=== stop === ')
            speak("Stop! Obstacle detected")
            return

        elif any(word in list for word in keyword_list2):
            print('=== alert === ')
            speak("Alert!")
            return

        elif any(word in list for word in keyword_list3):
            print('=== turn left === ')
            speak("Turn left")
            return

        elif any(word in list for word in keyword_list4):
            print('=== turn right === ')
            speak("Turn right")
            return

        elif any(word in list for word in keyword_list5):
            print('=== go straight === ')
            speak("Go straight")
            return
