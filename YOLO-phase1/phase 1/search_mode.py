# search_mode.py

import time
import pyttsx3


class SearchMode:

    def __init__(self, target_object):
        """
        This function runs when we create the SearchMode object.
        It sets up everything we need.
        """

        # The object we are searching for (example: "cup")
        self.target = target_object

        # This will help the system speak
        self.tts = pyttsx3.init()

        # Control how fast it speaks
        self.tts.setProperty('rate', 170)

        # Store the last thing we said (so we don't repeat)
        self.last_message = None

        # Prevent speaking too frequently
        self.last_speech_time = 0
        self.speech_cooldown = 2.0  # seconds

        # Distance threshold (area size means closer)
        self.reach_threshold = 30000

    # --------------------------------------------
    # This function runs every frame
    # --------------------------------------------
    def process(self, detections, frame_width):

        # Current time (used for cooldown)
        current_time = time.time()

        # Filter detections to only target object
        target_objects = [
            obj for obj in detections
            if obj["class"] == self.target
        ]

        # If we don't see the object at all
        if len(target_objects) == 0:
            self.try_speak(f"Scanning for {self.target}", current_time)
            return

        # If we see multiple, choose the biggest (closest)
        closest_object = max(target_objects, key=lambda x: x["area"])

        # Extract region and area
        region = closest_object["region"]
        area = closest_object["area"]

        # Decide what instruction to give
        message = self.decide_instruction(region, area)

        # Try speaking it
        self.try_speak(message, current_time)

    # --------------------------------------------
    # Decide what to tell the user
    # --------------------------------------------
    def decide_instruction(self, region, area):

        # If object is very close and centered
        if area > self.reach_threshold and region == "center":
            return "Object in front of you. You can reach it now."

        # If object is not centered, guide direction
        if region == "left":
            return "Turn left slowly."

        elif region == "right":
            return "Turn right slowly."

        elif region == "center":
            return "Move forward slowly."

        # Just in case something unexpected happens
        return "Adjust position."

    # --------------------------------------------
    # Speak only if needed
    # --------------------------------------------
    def try_speak(self, message, current_time):

        # If message is the same as last time, don't repeat
        if message == self.last_message:
            return

        # If cooldown time hasn't passed, don't speak
        if current_time - self.last_speech_time < self.speech_cooldown:
            return

        # Print message in terminal
        print("SEARCH MODE:", message)

        # Speak the message
        try:
            self.tts.say(message)
            self.tts.runAndWait()
        except:
            print("Speech error.")

        # Remember what we said
        self.last_message = message
        self.last_speech_time = current_time