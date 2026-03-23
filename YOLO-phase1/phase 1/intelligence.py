import time
import pyttsx3


class IntelligenceEngine:
    def __init__(self):

        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 170)

        self.objects_memory = {}
        self.last_speech_time = 0
        self.speech_cooldown = 2.0  # seconds

        # Area thresholds (tune these!)
        self.very_close_threshold = 25000
        self.close_threshold = 15000

    # -----------------------------------
    # MAIN LOGIC
    # -----------------------------------
    def process(self, detections, frame_width=None):

        current_time = time.time()

        for obj in detections:

            obj_id = obj["id"]
            label = obj["class"]
            area = obj["area"]
            region = obj["region"]

            previous = self.objects_memory.get(obj_id, None)

            approaching = False
            entering_path = False

            if previous:
                prev_area = previous["area"]
                prev_region = previous["region"]

                # Detect approaching (area increasing)
                if area - prev_area > 1500:
                    approaching = True

                # Detect entering path
                if prev_region in ["left", "right"] and region == "center":
                    entering_path = True

            message = None

            # ---- PERSON LOGIC ----
            if label == "person":

                if area > self.very_close_threshold:
                    message = f"Person very close {region}"

                elif entering_path:
                    message = "Person entering your path"

                elif approaching and region == "center":
                    message = "Person approaching ahead"

                elif region == "center":
                    message = "Person ahead"

                elif region == "left":
                    message = "Person on your left"

                elif region == "right":
                    message = "Person on your right"

            # ---- VEHICLE LOGIC ----
            elif label in ["car", "bus", "truck", "motorcycle", "bicycle"]:

                if area > self.close_threshold:
                    message = f"{label} close {region}"

                elif approaching:
                    message = f"{label} approaching {region}"

            # -----------------------------------
            # SPEAK IF MESSAGE CHANGED
            # -----------------------------------
            if message:

                if not previous or previous.get("last_message") != message:

                    if current_time - self.last_speech_time > self.speech_cooldown:
                        print("ALERT:", message)
                        self.speak(message)

                        self.last_speech_time = current_time

                        if obj_id not in self.objects_memory:
                            self.objects_memory[obj_id] = {}

                        self.objects_memory[obj_id]["last_message"] = message

            # Update memory
            self.objects_memory[obj_id] = {
                "area": area,
                "region": region,
                "last_message": self.objects_memory.get(obj_id, {}).get("last_message", None)
            }

    # -----------------------------------
    # TEXT TO SPEECH
    # -----------------------------------
    def speak(self, text):
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except:
            print("Speech error.")