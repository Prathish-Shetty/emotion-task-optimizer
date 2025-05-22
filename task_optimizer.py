#task_optimizer.py
import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import cv2
import speech_recognition as sr
from textblob import TextBlob
import hashlib
import json
from pathlib import Path
import logging
import traceback
import time
from typing import Dict, List, Optional, Tuple
from tensorflow.keras.models import load_model

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("task_optimizer.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

class EmployeeData:
    def __init__(self, employee_id: str):
        self.employee_id = self._hash_id(employee_id)
        self.mood_history = []
        self.task_history = []

    @staticmethod
    def _hash_id(employee_id: str) -> str:
        """Hash employee ID for privacy"""
        return hashlib.sha256(employee_id.encode()).hexdigest()

    def add_mood_entry(self, mood: str, timestamp: datetime.datetime):
        """Add a new mood entry with timestamp"""
        self.mood_history.append({
            'timestamp': timestamp,
            'mood': mood
        })
    
    def add_task_entry(self, task: str, completion_status: bool, timestamp: datetime.datetime):
        """Add a task entry with completion status"""
        self.task_history.append({
            'timestamp': timestamp,
            'task': task,
            'completed': completion_status
        })

class EmotionDetector:
    def __init__(self):
        # Load OpenCV's pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Load trained CNN model
        self.model = load_model("cnn_facial_emotion.h5")
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        self.speech_recognizer = sr.Recognizer()
    
    def detect_facial_emotion(self, frame) -> str:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                logger.info("No faces detected in the frame")
                return 'neutral'  # No face detected
                
            # Get the largest face in the frame
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
                
            # Extract and preprocess face region
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            
            # Debug output to verify face detection
            cv2.imwrite('debug_face.jpg', roi)
            
            # Normalize and reshape for model input
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension
            roi = np.expand_dims(roi, axis=-1)  # Add channel dimension
            
            # Make prediction
            prediction = self.model.predict(roi)
            emotion_idx = np.argmax(prediction[0])
            emotion = self.emotion_labels[emotion_idx]
            
            logger.info(f"Detected emotion: {emotion} with confidence {prediction[0][emotion_idx]:.2f}")
            
            # Print all confidence values for debugging
            for i, label in enumerate(self.emotion_labels):
                logger.info(f"{label}: {prediction[0][i]:.4f}")
                
            return emotion

        except Exception as e:
            logger.error(f"Error in facial emotion detection: {str(e)}")
            logger.error(traceback.format_exc())
            return 'neutral'

    def detect_speech_emotion(self, audio_file: str) -> str:
        """
        Detect emotion from speech
        Returns: predicted emotion based on speech analysis
        """
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.speech_recognizer.record(source)
                text = self.speech_recognizer.recognize_google(audio)
                
                # Analyze sentiment using TextBlob
                analysis = TextBlob(text)
                polarity = analysis.sentiment.polarity
                
                if polarity > 0.3:
                    return 'happy'
                elif polarity < -0.3:
                    return 'sad'
                else:
                    return 'neutral'
                
        except Exception as e:
            logger.error(f"Error in speech emotion detection: {str(e)}")
            return 'neutral'

class TaskRecommender:
    def __init__(self):
        self.task_categories = {
            'angry': ['simple tasks', 'organizing', 'independent work'],
            'disgust': ['analytical tasks', 'evaluation work', 'documentation'],
            'fear': ['routine tasks', 'familiar work', 'guided activities'],
            'happy': ['complex problem solving', 'creative tasks', 'team collaboration'],
            'sad': ['simple tasks', 'organizing', 'skill development'],
            'surprise': ['creative tasks', 'exploration activities', 'learning opportunities'],
            'neutral': ['routine tasks', 'documentation', 'planning'],
            'stressed': ['breaks', 'low-pressure tasks', 'administrative work']
        }

    def get_task_recommendation(self, mood: str) -> List[str]:
        """Get task recommendations based on current mood"""
        # Make sure mood is lowercase for consistent matching
        mood = mood.lower()
        
        # Log the incoming mood and the recommendation
        logger.info(f"Recommending tasks for mood: {mood}")
        
        # If the mood isn't in our categories, fall back to neutral
        if mood not in self.task_categories:
            logger.warning(f"Unknown mood: {mood}, defaulting to neutral")
            mood = 'neutral'
            
        recommendations = self.task_categories[mood]
        logger.info(f"Recommended tasks: {recommendations}")
        
        return recommendations

    def analyze_task_history(self, task_history: List[Dict]) -> Dict:
        """Analyze task completion patterns"""
        df = pd.DataFrame(task_history)
        completion_rate = df['completed'].mean() if not df.empty else 0
        return {
            'completion_rate': completion_rate,
            'total_tasks': len(task_history),
            'completed_tasks': df['completed'].sum() if not df.empty else 0
        }

class MoodAnalytics:
    def __init__(self):
        self.stress_threshold = 3  # Number of consecutive stressed/sad days

    def analyze_mood_trends(self, mood_history: List[Dict]) -> Dict:
        """Analyze mood patterns and identify concerning trends"""
        df = pd.DataFrame(mood_history)
        
        if len(df) < 1:
            return {'status': 'insufficient_data', 'stress_level': 0, 'dominant_mood': 'unknown', 'mood_variability': 0}
            
        recent_moods = df.tail(self.stress_threshold)
        stress_count = sum(1 for mood in recent_moods['mood'] if mood in ['stressed', 'sad', 'angry', 'fear'])
        
        return {
            'status': 'alert' if stress_count >= self.stress_threshold else 'normal',
            'stress_level': stress_count / self.stress_threshold if self.stress_threshold > 0 else 0,
            'dominant_mood': df['mood'].mode().iloc[0] if not df.empty and len(df['mood'].mode()) > 0 else 'unknown',
            'mood_variability': len(df['mood'].unique()) / len(df) if not df.empty and len(df) > 0 else 0
        }

class TaskOptimizer:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.task_recommender = TaskRecommender()
        self.mood_analytics = MoodAnalytics()
        self.employees = {}
        
        # Load existing employee data if available
        try:
            self.load_employee_data()
        except Exception as e:
            logger.error(f"Error loading employee data: {str(e)}")

    def save_employee_data(self):
        """Save employee data to a JSON file"""
        try:
            with open("employees.json", "w") as f:
                json.dump(
                    {eid: {
                        'employee_id': emp.employee_id,
                        'mood_history': emp.mood_history,
                        'task_history': emp.task_history
                    } for eid, emp in self.employees.items()},
                    f,
                    default=str,
                    indent=2
                )
            logger.info("Employee data saved successfully")
        except Exception as e:
            logger.error(f"Error saving employee data: {str(e)}")
    
    def load_employee_data(self):
        """Load employee data from JSON file"""
        try:
            if Path("employees.json").exists():
                with open("employees.json", "r") as f:
                    data = json.load(f)
                    
                for eid, emp_data in data.items():
                    employee = EmployeeData(eid)
                    employee.employee_id = emp_data['employee_id']
                    employee.mood_history = emp_data['mood_history']
                    employee.task_history = emp_data['task_history']
                    self.employees[eid] = employee
                    
                logger.info(f"Loaded data for {len(self.employees)} employees")
        except Exception as e:
            logger.error(f"Error loading employee data: {str(e)}")
    
    def register_employee(self, employee_id: str):
        """Register a new employee"""
        if employee_id not in self.employees:
            self.employees[employee_id] = EmployeeData(employee_id)
            logger.info(f"Registered new employee with hashed ID: {self.employees[employee_id].employee_id}")

    def process_employee_state(self, 
                            employee_id: str,
                            video_frame,
                            audio_file: Optional[str] = None) -> Dict:
        """Process employee state and provide recommendations"""
        if employee_id not in self.employees:
            self.register_employee(employee_id)
            
        # Detect emotions
        facial_emotion = self.emotion_detector.detect_facial_emotion(video_frame)
        logger.info(f"Detected facial emotion: {facial_emotion}")
        
        speech_emotion = 'neutral'
        if audio_file:
            speech_emotion = self.emotion_detector.detect_speech_emotion(audio_file)
            logger.info(f"Detected speech emotion: {speech_emotion}")
        
        # Combine emotions (prioritize facial emotion, use speech as fallback)
        # Modified emotion combination logic
        if facial_emotion != 'neutral':
            final_mood = facial_emotion
        elif speech_emotion != 'neutral':
            final_mood = speech_emotion
        else:
            final_mood = 'neutral'
        
        logger.info(f"Final determined mood: {final_mood}")
        
        # Update employee records
        self.employees[employee_id].add_mood_entry(final_mood, datetime.datetime.now())
        self.save_employee_data()
        
        # Get recommendations and analytics
        recommendations = self.task_recommender.get_task_recommendation(final_mood)
        mood_analysis = self.mood_analytics.analyze_mood_trends(
            self.employees[employee_id].mood_history
        )
        
        return {
            'current_mood': final_mood,
            'task_recommendations': recommendations,
            'mood_analysis': mood_analysis,
            'alert_required': mood_analysis['status'] == 'alert'
        }

    def get_team_analytics(self, team_ids: List[str]) -> Dict:
        """Get aggregated team mood analytics"""
        team_moods = []
        for emp_id in team_ids:
            if emp_id in self.employees:
                emp_data = self.employees[emp_id]
                if emp_data.mood_history:
                    team_moods.append(emp_data.mood_history[-1]['mood'])
        
        if not team_moods:
            return {'status': 'no_data'}
        
        return {
            'team_mood_distribution': {
                mood: team_moods.count(mood) / len(team_moods)
                for mood in set(team_moods)
            },
            'team_stress_level': sum(1 for mood in team_moods if mood in ['stressed', 'sad', 'angry', 'fear']) / len(team_moods)
        }

# Example usage
def main():
    # Initialize the optimizer
    optimizer = TaskOptimizer()

    # Mock video frame (in practice, this would come from a camera)
    mock_frame = np.zeros((300, 300, 3), dtype=np.uint8)

    # Process an employee
    result = optimizer.process_employee_state(
        employee_id="EMP001",
        video_frame=mock_frame
    )

    # Print results
    print("Employee Analysis Results:")
    print(json.dumps(result, indent=2))

    # Get team analytics
    team_result = optimizer.get_team_analytics(["EMP001", "EMP002"])
    print("\nTeam Analysis Results:")
    print(json.dumps(team_result, indent=2))

if __name__ == "__main__":
    main()