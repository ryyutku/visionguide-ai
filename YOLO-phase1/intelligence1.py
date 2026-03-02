# intelligence1.py

import pyttsx3
import time

class IntelligenceEngine:
    def __init__(self):
        self.memory = {}
        self.engine = pyttsx3.init()
        self.last_global_speech = 0
        self.global_cooldown = 2  # seconds

    def speak(self, text):
        current_time = time.time()
        if current_time - self.last_global_speech < self.global_cooldown:
            return

        self.last_global_speech = current_time
        print("ALERT:", text)
        self.engine.say(text)
        self.engine.runAndWait()

    def update(self, detections):

        for obj in detections:
            obj_id = obj["id"]
            area = obj["area"]
            region = obj["region"]
            cls = obj["class"]

            if obj_id not in self.memory:
                # First time seeing object
                self.memory[obj_id] = {
                    "last_area": area,
                    "last_region": region,
                    "alerted": False
                }

                if region == "center":
                    self.speak(f"{cls} ahead")
                continue

            prev = self.memory[obj_id]

            # Detect approaching
            if area > prev["last_area"] * 1.2:
                if region == "center":
                    self.speak(f"{cls} approaching")

            # Detect movement toward center
            if prev["last_region"] in ["left", "right"] and region == "center":
                self.speak(f"{cls} entering path")

            # Detect too close
            if area > 120000:
                self.speak(f"{cls} very close")

            # Update memory
            prev["last_area"] = area
            prev["last_region"] = region