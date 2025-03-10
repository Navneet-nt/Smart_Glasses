from ultralytics import YOLO
import cv2
import pyttsx3
import speech_recognition as sr
from threading import Thread
import queue  # Add this import
from queue import Queue
import time
import numpy as np
import threading
from queue import Empty
from concurrent.futures import ThreadPoolExecutor


class SmartObstacleAssistant:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO('obstacle_detection.pt')

        # Classify obstacles by avoidability
        self.unavoidable_obstacles = ['Barrier', 'Gate', 'Roadblock']
        self.avoidable_obstacles = ['Car', 'Door', 'Drain', 'Fence', 'Pole',
                                    'Tree', 'Rickshaw', 'Pothole']
        self.obstacles = self.unavoidable_obstacles + self.avoidable_obstacles

        # Navigation parameters
        self.safe_distance = 0.3
        self.turn_threshold = 0.4

        # Initialize TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 240)
        self.engine.setProperty('volume', 0.9)

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.8  # Reduce pause threshold for better command detection
        self.recognizer.operation_timeout = 5  # Set operation timeout

        # Message queues
        self.tts_queue = Queue()
        self.command_queue = Queue()

        # State variables
        self.is_running = True
        self.verbose_mode = True
        self.paused = False
        self.current_frame_objects = set()  # Track objects in current frame
        self.previous_frame_objects = set()  # Track objects in previous frame
        self.last_announcement = {}
        self.last_navigation_time = 0
        self.navigation_cooldown = 2

        # Start TTS thread
        self.tts_thread = Thread(target=self._speak_thread, daemon=True)
        self.tts_thread.start()

        # Command keywords
        self.commands = {
            "stop": self.stop_detection,
            "pause": self.toggle_pause,
            "resume": self.toggle_pause,
            "quiet": self.toggle_verbose,
            "verbose": self.toggle_verbose,
            "help": self.show_help,
            "tell what is around me": self.announce_surroundings,

            # command variations
            "stop detection": self.stop_detection,
            "pause detection": self.toggle_pause,
            "resume detection": self.toggle_pause,
            "quiet mode": self.toggle_verbose,
            "verbose mode": self.toggle_verbose,
            "show help": self.show_help,
            "what is around": self.announce_surroundings,
            "what's around": self.announce_surroundings,
            "tell me surroundings": self.announce_surroundings
        }

    def announce_surroundings(self):
        """Announce all objects in the current frame with their distances"""
        if not self.current_frame_objects:
            self.tts_queue.put("No objects detected around you")
            return

        # Group objects by distance
        far_objects = []
        close_objects = []
        very_close_objects = []

        for obj in self.current_frame_objects:
            if obj['distance'] == "far":
                far_objects.append(obj)
            elif obj['distance'] == "close":
                close_objects.append(obj)
            else:  # very close
                very_close_objects.append(obj)

        # Announce very close objects first
        if very_close_objects:
            message = "Very close objects: "
            for obj in very_close_objects:
                is_unavoidable = obj['object'] in self.unavoidable_obstacles
                message += f"{obj['object']} on your {obj['position']}"
                if is_unavoidable:
                    message += " (UNAVOIDABLE), "
                else:
                    message += ", "
            self.tts_queue.put(message.rstrip(", "))

        # Then close objects
        if close_objects:
            message = "Close objects: "
            for obj in close_objects:
                is_unavoidable = obj['object'] in self.unavoidable_obstacles
                message += f"{obj['object']} on your {obj['position']}"
                if is_unavoidable:
                    message += " (UNAVOIDABLE), "
                else:
                    message += ", "
            self.tts_queue.put(message.rstrip(", "))

        # Finally far objects
        if far_objects:
            message = "Far objects: "
            for obj in far_objects:
                # Calculate approximate distance in meters (rough estimation)
                distance_meters = round(10 * (1 - obj['distance_value']))  # Simple conversion
                message += f"{obj['object']} about {distance_meters} meters away on your {obj['position']}, "
            self.tts_queue.put(message.rstrip(", "))

    def stop_detection(self):
        """Stop the detection process"""
        self.is_running = False
        self.tts_queue.put("Stopping detection system")

    def toggle_pause(self):
        """Toggle pause/resume state"""
        self.paused = not self.paused
        state = "paused" if self.paused else "resumed"
        self.tts_queue.put(f"Detection {state}")

    def toggle_verbose(self):
        """Toggle between verbose and quiet mode"""
        self.verbose_mode = not self.verbose_mode
        mode = "verbose" if self.verbose_mode else "quiet"
        self.tts_queue.put(f"Switching to {mode} mode")

    def show_help(self):
        """Announce available voice commands"""
        help_text = ("Available commands are: stop to end detection, "
                     "pause or resume to control detection, "
                     "quiet or verbose to change announcement style, "
                     "and help to hear these commands again")
        self.tts_queue.put(help_text)

    def process_video(self, video_path=0):
        """Process video stream with command handling"""
        cap = cv2.VideoCapture(video_path)

        # command thread with error handling
        command_thread = Thread(target=self._safe_command_thread, daemon=True)
        command_thread.start()

        print("\nVoice Command System Ready!")
        print("Available commands:")
        for cmd in set(self.commands.values()):  # Use set to avoid duplicate functions
            print(f"- '{cmd.__name__.replace('_', ' ')}'")
        print("\nListening for commands...")

        while cap.isOpened() and self.is_running:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow("Smart Obstacle Detection Assistant", processed_frame)

            # Handle commands
            try:
                command_func = self.command_queue.get_nowait()
                command_func()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error executing command: {str(e)}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()
        cap.release()
        cv2.destroyAllWindows()

    def _safe_command_thread(self):
        """Wrapper for command thread with error handling"""
        try:
            self.listen_for_commands()
        except Exception as e:
            print(f"Command thread error: {str(e)}")
            self.tts_queue.put("Voice command system encountered an error. Please restart the application.")

    def announce_frame_changes(self, current_detections):
        """Announce only close/very close objects and give warnings for unavoidable obstacles"""
        self.current_frame_objects = current_detections  # Update current frame objects

        for detection in current_detections:
            if detection['distance'] in ["close", "very close"]:
                key = f"{detection['object']}_{detection['position']}_{detection['distance']}"
                current_time = time.time()

                # Check if we recently announced this object
                if key not in self.last_announcement or \
                   (current_time - self.last_announcement[key]) > 5:  # 5 second cooldown
                    
                    # Special handling for unavoidable obstacles
                    if detection['object'] in self.unavoidable_obstacles:
                        message = f"WARNING! Unavoidable {detection['object']} {detection['distance']} on your {detection['position']}"
                        if detection['distance'] == "very close":
                            message += ". STOP IMMEDIATELY!"
                        self.tts_queue.put(message)
                    else:
                        # Only announce very close objects unless in verbose mode
                        if detection['distance'] == "very close" or self.verbose_mode:
                            message = f"{detection['object']} {detection['distance']} on your {detection['position']}"
                            self.tts_queue.put(message)

                    self.last_announcement[key] = current_time

    def get_navigation_instruction(self, detections, frame_width, frame_height):
        """Generate navigation instructions prioritizing unavoidable obstacles"""
        if not detections:
            return None

        # Sort by distance and prioritize unavoidable obstacles
        detections.sort(key=lambda x: (
            x['object'] in self.unavoidable_obstacles,
            x['distance'] == "very close",
            x['distance_value']
        ), reverse=True)

        for obj in detections:
            if obj['distance'] == "very close":
                if obj['object'] in self.unavoidable_obstacles:
                    return "STOP! Unavoidable obstacle ahead!"
                
                if obj['position'] == "left":
                    return "Turn right to avoid obstacle"
                elif obj['position'] == "right":
                    return "Turn left to avoid obstacle"
                elif obj['position'] == "center":
                    return "Stop or turn to either side"

        return None
    
    def process_frame(self, frame):
        """Process a single frame for obstacle detection and navigation."""
        results = self.model.predict(frame, conf=0.25)
        frame_height, frame_width = frame.shape[:2]
        current_detections = []

        for result in results[0].boxes:
            x_min, y_min, x_max, y_max = map(int, result.xyxy[0])
            confidence = float(result.conf[0])
            class_id = int(result.cls[0])
            class_label = self.model.names[class_id]

            if class_label not in self.obstacles:
                continue

            center_x = (x_min + x_max) // 2
            box_height = y_max - y_min
            distance_value = box_height / frame_height

            if distance_value > 0.5:
                distance = "very close"
            elif distance_value > 0.3:
                distance = "close"
            else:
                distance = "far"

            if center_x < frame_width / 3:
                position = "left"
            elif center_x > 2 * frame_width / 3:
                position = "right"
            else:
                position = "center"

            detection = {
                'object': class_label,
                'position': position,
                'distance': distance,
                'distance_value': distance_value,
                'confidence': confidence
            }

            current_detections.append(detection)

            color = (0, 0, 255) if distance == "very close" else (0, 255, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            label_text = f"{class_label} ({distance})"
            cv2.putText(frame, label_text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.announce_frame_changes(current_detections)

        if current_detections and time.time() - self.last_navigation_time > self.navigation_cooldown:
            instruction = self.get_navigation_instruction(current_detections, frame_width, frame_height)
            if instruction:
                self.tts_queue.put(instruction)
                self.last_navigation_time = time.time()

        return frame


    def listen_for_commands(self):
        """Continuously listen for voice commands and enqueue them"""
        while self.is_running:
            with sr.Microphone() as source:
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    command = self.recognizer.recognize_google(audio).lower()
                    if command in self.commands:
                        self.command_queue.put(self.commands[command])
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue

    def _speak_thread(self):
        while self.is_running:
            try:
                message = self.tts_queue.get(timeout=1)
                if message == "STOP":
                    break
                self.engine.say(message)
                self.engine.runAndWait()
                time.sleep(0.1)
            except:
                continue

    def cleanup(self):
        self.tts_queue.put("STOP")
        self.tts_thread.join()


if __name__ == "__main__":
    assistant = SmartObstacleAssistant()
    print("Starting Smart Obstacle Detection Assistant...")
    print("Available voice commands:")
    print("- 'stop': Stop detection")
    print("- 'pause/resume': Pause or resume detection")
    print("- 'quiet/verbose': Toggle verbosity")
    print("- 'help': Show available commands")

    assistant.process_video("video_test.mp4")
