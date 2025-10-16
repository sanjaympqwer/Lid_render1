#!/usr/bin/env python3

import cv2
import numpy as np
import time
import argparse
import sys
import os
from ultralytics import YOLO
from collections import defaultdict
import threading
import torch
import requests
from datetime import datetime, timezone

DEFAULT_SERVER_URL = os.environ.get('LOST_SERVER_URL', 'http://localhost:5000')
DEFAULT_LOCATION = os.environ.get('LOST_LOCATION', 'College')


class AlertSystem:
    def __init__(self, threshold=30, server_url: str = DEFAULT_SERVER_URL, location: str = DEFAULT_LOCATION):
        self.threshold = threshold
        self.alerts = {}
        self.last_beep_time = {}
        self.people_positions = []
        self.item_owner_associations = {}
        self.person_item_history = {}
        self.item_alert_thresholds = {}
        self.unattached_start_time = {}
        self.server_url = server_url.rstrip('/')
        self.location = location

    def post_event(self, event: str, class_name: str, persistent_id: int, details: str | None = None):
        payload = {
            'event': event,
            'className': class_name,
            'persistentId': persistent_id,
            'location': self.location,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': details or ''
        }
        try:
            requests.post(f"{self.server_url}/api/lost-events", json=payload, timeout=2)
        except Exception:
            pass
    
    def play_beep_sound(self):
        try:
            import winsound
            winsound.Beep(1000, 500)
        except:
            try:
                print('\a')
            except:
                pass
    
    def update_people_positions(self, people_positions, people_track_ids):
        self.people_positions = list(zip(people_positions, people_track_ids))
    
    def is_owner_near_item(self, item_position, item_track_id, distance_threshold=100):
        if not self.people_positions:
            return False
        
        owner_track_id = self.item_owner_associations.get(item_track_id)
        if owner_track_id is None:
            return False
        
        item_x, item_y = item_position[0], item_position[1]
        
        for person_pos, person_track_id in self.people_positions:
            if person_track_id == owner_track_id:
                person_x, person_y = person_pos[0], person_pos[1]
                distance = np.sqrt((item_x - person_x)**2 + (item_y - person_y)**2)
                if distance < distance_threshold:
                    return True
        return False
    
    def track_person_item_interaction(self, person_track_id, item_track_id, interaction_type="near"):
        if person_track_id not in self.person_item_history:
            self.person_item_history[person_track_id] = {}
        
        if item_track_id not in self.person_item_history[person_track_id]:
            self.person_item_history[person_track_id][item_track_id] = []
        
        self.person_item_history[person_track_id][item_track_id].append({
            'type': interaction_type,
            'time': time.time()
        })
        
        if len(self.person_item_history[person_track_id][item_track_id]) > 10:
            self.person_item_history[person_track_id][item_track_id] = \
                self.person_item_history[person_track_id][item_track_id][-10:]
    
    def determine_item_owner(self, item_track_id, item_position):
        if not self.people_positions:
            return None
        
        item_x, item_y = item_position[0], item_position[1]
        best_owner = None
        best_score = 0
        
        for person_pos, person_track_id in self.people_positions:
            person_x, person_y = person_pos[0], person_pos[1]
            distance = np.sqrt((item_x - person_x)**2 + (item_y - person_y)**2)
            
            interaction_score = 0
            if person_track_id in self.person_item_history and \
               item_track_id in self.person_item_history[person_track_id]:
                interactions = self.person_item_history[person_track_id][item_track_id]
                for interaction in interactions[-5:]:
                    if interaction['type'] == 'near':
                        interaction_score += 1
            
            total_score = interaction_score - (distance / 50)
            
            if total_score > best_score:
                best_score = total_score
                best_owner = person_track_id
        
        return best_owner if best_score > 0 else None
    
    def update_items(self, all_items, people_positions, people_track_ids):
        current_time = time.time()
        
        person_count = len(people_positions)
        print(f"👥 Person count in frame: {person_count}")
        
        if person_count >= 1:
            # At least 1 person detected - all items should be ACTIVE
            print(f"✅ {person_count} person(s) detected - All items ACTIVE, clearing all alerts")
            
            # Clear all alerts and timers since people are present
            for item in all_items:
                class_name = item.detection.class_name.lower()
                
                # Always post resolve event when person is detected (regardless of previous state)
                try:
                    self.post_event('resolve', item.detection.class_name, item.track_id, 'person detected in frame')
                    print(f"✅ {item.detection.class_name} (ID: {item.track_id}) - Person present, status: ACTIVE")
                except Exception:
                    pass
                
                # Clear alerts if they exist
                if item.track_id in self.alerts:
                    del self.alerts[item.track_id]
                    if item.track_id in self.last_beep_time:
                        del self.last_beep_time[item.track_id]
                    print(f"✅ {item.detection.class_name} (ID: {item.track_id}) - Alert cleared")
                
                # Clear any pending unattached tracking
                if item.track_id in self.unattached_start_time:
                    del self.unattached_start_time[item.track_id]
                    print(f"✅ {item.detection.class_name} (ID: {item.track_id}) - Timer cleared")
        else:
            # No person detected (person_count = 0) - start timer for all items
            print(f"🚨 No person detected - Starting 10s timer for all items")
            
            for item in all_items:
                class_name = item.detection.class_name.lower()
                
                # Check if item should start timer
                if item.track_id not in self.unattached_start_time:
                    # Item just became unattached - start tracking time
                    self.unattached_start_time[item.track_id] = current_time
                    print(f"⏱️ {item.detection.class_name} (ID: {item.track_id}) - No person detected, starting 10s timer")
                else:
                    # Item continues to be unattached - check if 10 seconds have passed
                    unattached_duration = current_time - self.unattached_start_time[item.track_id]
                    
                    if unattached_duration >= 10.0:  # 10 seconds delay
                        # Now mark as INACTIVE and start alert
                        if item.track_id not in self.alerts:
                            self.alerts[item.track_id] = current_time
                            self.last_beep_time[item.track_id] = 0
                            self.item_alert_thresholds[item.track_id] = self.get_alert_threshold(class_name)
                            print(f"🚨 {item.detection.class_name} (ID: {item.track_id}) - 10s timeout reached - Status: INACTIVE")
                            # Report start of unattached period
                            try:
                                self.post_event('start', item.detection.class_name, item.track_id, 'no person detected for 10s, alert started')
                            except Exception:
                                pass
                        else:
                            # Item continues to be unattached - continue alert
                            duration = current_time - self.alerts[item.track_id]
                            item_threshold = self.item_alert_thresholds.get(item.track_id, self.threshold)
                            
                            if duration >= item_threshold:
                                # Play beep sound every 5 seconds for ongoing alerts
                                if current_time - self.last_beep_time.get(item.track_id, 0) >= 5:
                                    self.play_beep_sound()
                                    self.last_beep_time[item.track_id] = current_time
                                    # Periodic update when beeping
                                    try:
                                        self.post_event('update', item.detection.class_name, item.track_id, f'no person for {duration:.1f}s')
                                    except Exception:
                                        pass
                            
                            # Enhanced alert message for primary items
                            priority_indicator = ""
                            if class_name in ['cell phone', 'keys', 'backpack', 'suitcase', 'bottle']:
                                priority_indicator = " 🔥 HIGH PRIORITY"
                            
                            print(f"🚨 ALERT{priority_indicator}: {item.detection.class_name} (ID: {item.track_id}) "
                                  f"no person detected for {duration:.1f} seconds!")
                    else:
                        # Still within 10 second grace period
                        remaining_time = 10.0 - unattached_duration
                        print(f"⏱️ {item.detection.class_name} (ID: {item.track_id}) - Grace period: {remaining_time:.1f}s remaining")

    def update(self, unattended_items):
        current_time = time.time()
        
        for item in unattended_items:
            # Get the current position of this item
            item_position = None
            if hasattr(item, 'position'):
                item_position = item.position
            
            # Get enhanced alert threshold for this specific item
            class_name = item.detection.class_name.lower()
            alert_threshold = self.get_alert_threshold(class_name)
            
            # Determine item owner if not already assigned
            if item.track_id not in self.item_owner_associations and item_position:
                owner = self.determine_item_owner(item.track_id, item_position)
                if owner:
                    self.item_owner_associations[item.track_id] = owner
                    print(f"🔗 Associated {item.detection.class_name} (ID: {item.track_id}) with Person (ID: {owner})")
            
            # Check if the owner is near this item
            owner_nearby = False
            if item_position and item.track_id in self.item_owner_associations:
                owner_nearby = self.is_owner_near_item(item_position, item.track_id)
            
            if owner_nearby:
                # Owner is nearby, clear the alert
                if item.track_id in self.alerts:
                    owner_id = self.item_owner_associations[item.track_id]
                    print(f"✅ Owner (Person ID: {owner_id}) returned to {item.detection.class_name} (ID: {item.track_id}) - Alert cleared!")
                    del self.alerts[item.track_id]
                    if item.track_id in self.last_beep_time:
                        del self.last_beep_time[item.track_id]
                    # Report resolution
                    try:
                        self.post_event('resolve', item.detection.class_name, item.track_id, 'owner returned, alert cleared')
                    except Exception:
                        pass
            else:
                # No owner nearby, continue with alert logic
                if item.track_id not in self.alerts:
                    self.alerts[item.track_id] = current_time
                    self.last_beep_time[item.track_id] = 0
                    self.item_alert_thresholds[item.track_id] = alert_threshold
                    if item.track_id in self.item_owner_associations:
                        owner_id = self.item_owner_associations[item.track_id]
                        print(f"🚨 {item.detection.class_name} (ID: {item.track_id}) left by Person (ID: {owner_id}) - Starting alert timer")
                    # Report start
                    try:
                        self.post_event('start', item.detection.class_name, item.track_id, 'unattended started')
                    except Exception:
                        pass
                else:
                    duration = current_time - self.alerts[item.track_id]
                    item_threshold = self.item_alert_thresholds.get(item.track_id, self.threshold)
                    
                    if duration >= item_threshold:
                        # Play beep sound every 5 seconds for ongoing alerts
                        if current_time - self.last_beep_time.get(item.track_id, 0) >= 5:
                            self.play_beep_sound()
                            self.last_beep_time[item.track_id] = current_time
                            # Periodic update when beeping
                            try:
                                self.post_event('update', item.detection.class_name, item.track_id, f'unattended for {duration:.1f}s')
                            except Exception:
                                pass
                        
                        owner_info = ""
                        if item.track_id in self.item_owner_associations:
                            owner_id = self.item_owner_associations[item.track_id]
                            owner_info = f" (Owner: Person {owner_id})"
                        
                        # Enhanced alert message for primary items
                        priority_indicator = ""
                        if class_name in ['cell phone', 'keys', 'backpack', 'suitcase', 'bottle']:
                            priority_indicator = " 🔥 HIGH PRIORITY"
                        
                        print(f"🚨 ALERT{priority_indicator}: {item.detection.class_name} (ID: {item.track_id}){owner_info} "
                              f"unattended for {duration:.1f} seconds!")
    
    def get_alert_threshold(self, class_name):
        """Get alert threshold for specific item types."""
        class_lower = class_name.lower()
        
        # Enhanced alert thresholds for primary items
        if class_lower == 'cell phone':
            return 10  # Very fast alerts for phones
        elif class_lower == 'keys':
            return 8   # Very fast alerts for keys
        elif class_lower in ['backpack', 'suitcase']:
            return 15  # Fast alerts for bags
        elif class_lower == 'bottle':
            return 20  # Fast alerts for bottles
        elif class_lower == 'person':
            return 20  # Fast alerts for people leaving items
        else:
            return 30  # Default threshold for other items

class LostItemDetector:
    def __init__(self):
        """Initialize the lost item detection system."""
        print("🔧 Initializing AI-Based Lost Item Detector...")
        
        # Load YOLO model - use nano model for better FPS
        try:
            self.model = YOLO('yolov8n.pt')  # Use nano model for better FPS
            print("✓ YOLO nano model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO nano model: {e}")
            print("Downloading YOLO nano model...")
            self.model = YOLO('yolov8n.pt')
        
        # Enable GPU acceleration if available
        if torch.cuda.is_available():
            self.model.to('cuda')
            print("✓ GPU acceleration enabled")
        else:
            print("⚠ Using CPU (GPU not available)")
        
        # Initialize tracking with persistent object IDs
        self.tracked_objects = {}
        self.object_positions = defaultdict(list)
        self.object_classes = {}  # Store class names for each track ID
        self.object_history = {}  # Store persistent object history
        self.next_object_id = 1  # Global object ID counter
        self.alert_system = AlertSystem(threshold=30)  # 30 seconds threshold
        
        # FPS optimization settings
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0
        self.process_every_n_frames = 1  # Process every frame initially
        self.last_processing_time = 0
        self.target_fps = 30
        
        # Persistent tracking settings
        self.max_disappeared_frames = 30  # Keep object ID for 30 frames after disappearance
        self.disappeared_objects = {}  # Track disappeared objects
        self.reappearance_threshold = 50  # Distance threshold for object reappearance
        
        # Define wanted classes for lost item detection - Focused on specific items
        self.wanted_classes = [
            'person', 'backpack', 'handbag', 'suitcase', 
            'cell phone', 'laptop', 'keyboard', 'mouse',
            'book', 'umbrella', 'bottle',
            'tv', 'remote', 'clock', 'vase', 'scissors',
            'teddy bear', 'toothbrush', 'keys'
        ]
        
        # Primary focus classes - the most important items to detect
        self.primary_classes = [
            'person', 'backpack', 'suitcase', 'cell phone', 'bottle', 'keys'
        ]
        
        # Define confidence thresholds for different object types - Optimized for specific items
        self.confidence_thresholds = {
            # People (easy to detect)
            'person': 0.35,     # Lower threshold for people (easier to detect, want to catch all)
            
            # Bags and luggage (medium difficulty) - Primary focus
            'backpack': 0.45,   # Lower threshold for backpacks (important to detect)
            'handbag': 0.6,     # Medium threshold for handbags (smaller, harder to detect)
            'suitcase': 0.45,   # Lower threshold for suitcases (important to detect)
            
            # Electronics (high value, harder to detect) - Primary focus
            'cell phone': 0.65, # Lower threshold for phones (very important to detect)
            'laptop': 0.6,      # Medium threshold for laptops (valuable, need accurate detection)
            'keyboard': 0.6,    # Medium threshold for keyboards
            'mouse': 0.7,       # High threshold for mouse (small object)
            'tv': 0.5,          # Lower threshold for TV (large, distinctive)
            #'remote': 0.8,      # Very high threshold for remote (very small)
            
            # Common items (medium difficulty) - Primary focus
            'book': 0.5,        # Medium threshold for books
            'umbrella': 0.6,    # Medium threshold for umbrellas
            'bottle': 0.45,     # Lower threshold for bottles (important to detect)
            #'cup': 0.5,         # Medium threshold for cups
            #'bowl': 0.5,        # Medium threshold for bowls
            #'clock': 0.6,       # Medium threshold for clocks
            #'vase': 0.5,        # Medium threshold for vases
            #'scissors': 0.7,    # High threshold for scissors (small, sharp objects)
            #'teddy bear': 0.5,  # Medium threshold for teddy bears
            #'toothbrush': 0.8,  # Very high threshold for toothbrush (very small)
            'keys': 0.7         # Lower threshold for keys (very important to detect)
        }
        
        # Unattended items that can be lost (categorized by importance)
        self.unattended_items = [
            # High-value items (electronics) - Primary focus
            'cell phone', 'laptop', 'keyboard', 'mouse', 'tv', 'remote',
            # Personal items (bags and luggage) - Primary focus
            'backpack', 'handbag', 'suitcase',
            # Small valuable items - Primary focus
            'keys', 'scissors',
            # Common items - Primary focus
            'bottle',
        ]
        
        # High-priority items that should trigger immediate alerts
        self.high_priority_items = [
            'cell phone', 'keys', 'backpack', 'suitcase', 'bottle'
        ]
        
        print("✓ Lost Item Detector initialized")
    
    def process_frame(self, frame):
        """Process a single frame for object detection and tracking."""
        # FPS optimization: Skip frames if needed
        self.frame_count += 1
        current_time = time.time()
        
        # Adaptive frame skipping based on processing time
        if current_time - self.last_processing_time < (1.0 / self.target_fps):
            # Skip this frame if we're running too fast
            return frame, []
        
        # Resize frame for faster processing (maintain aspect ratio)
        height, width = frame.shape[:2]
        if width > 640:  # Only resize if frame is larger than 640px
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Run YOLO detection with optimized parameters for FPS
        results = self.model.track(
            frame, 
            persist=True, 
            verbose=False,
            conf=0.4,  # Lower minimum confidence to catch all wanted classes
            iou=0.5,   # NMS IoU threshold
            max_det=20  # Reduced maximum detections for better FPS
        )
        
        self.last_processing_time = current_time
        
        if results is None or len(results) == 0:
            return frame, []
        
        result = results[0]
        unattended_items = []
        all_items = []  # Track all items (both attached and unattached)
        people_positions = []
        people_track_ids = []
        
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            # Track disappeared objects before processing new detections
            self.track_disappeared_objects()
            
            for box, track_id, class_id, confidence in zip(boxes, track_ids, class_ids, confidences):
                class_name = result.names[class_id]
                
                # Step 1: Use Class Confidence Threshold - only use predictions with high confidence
                if confidence < 0.35:  # Lower base threshold to catch more items
                    continue  # Skip very low confidence detections
                
                # Step 2: Filter Only Specific Classes - ignore other detected objects
                if class_name.lower() not in self.wanted_classes:
                    continue  # Skip unwanted classes
                
                # Get enhanced detection settings for this specific item
                detection_settings = self.get_enhanced_detection_settings(class_name)
                threshold = detection_settings['min_confidence']
                
                # Apply class-specific confidence threshold
                if confidence >= threshold:
                    # Store object position and class
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    position = (center_x, center_y)
                    
                    # Assign persistent ID to maintain same ID for same objects
                    persistent_id = self.assign_persistent_id(track_id, class_name, position)
                    
                    # Store object position and class with persistent ID
                    self.object_positions[track_id].append((center_x, center_y, time.time()))
                    self.object_classes[track_id] = class_name
                    
                    # Update object history
                    self.update_object_history(persistent_id, position)
                    
                    # Keep only last 15 positions (0.5 second at 30 fps) for better FPS
                    if len(self.object_positions[track_id]) > 15:
                        self.object_positions[track_id] = self.object_positions[track_id][-15:]
                    
                    # Track people positions for proximity detection
                    if class_name.lower() == 'person':
                        people_positions.append((center_x, center_y))
                        people_track_ids.append(track_id)
                    
                    # Track ALL items (both attached and unattended)
                    if class_name.lower() != 'person':
                        # Create object with position information
                        obj = type('Object', (), {
                            'track_id': persistent_id,  # Use persistent ID instead of track_id
                            'detection': type('Detection', (), {'class_name': class_name})(),
                            'position': (center_x, center_y)
                        })()
                        all_items.append(obj)
                        
                        # Check if item is near any person
                        is_near_person = False
                        for person_pos, person_track_id in zip(people_positions, people_track_ids):
                            person_x, person_y = person_pos[0], person_pos[1]
                            distance = np.sqrt((center_x - person_x)**2 + (center_y - person_y)**2)
                            if distance < 100:  # If person is near item
                                is_near_person = True
                                self.alert_system.track_person_item_interaction(person_track_id, persistent_id, "near")
                        
                        # If item is NOT near any person, it's potentially unattended
                        if not is_near_person:
                            unattended_items.append(obj)
                    
                    # Step 3: Enhanced Label-Based Coloring using detection settings
                    color = detection_settings['color']
                    thickness = detection_settings['thickness']
                    
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
                    
                    # Add enhanced label with confidence, persistent ID and tracking info
                    label = f"{class_name} ID:{persistent_id} ({confidence:.2f})"
                    
                    # Enhanced labels for primary items
                    if class_name.lower() == 'person':
                        label += " - PERSON"
                    else:
                        # Check if item is near any person
                        is_near_person = False
                        for person_pos in people_positions:
                            person_x, person_y = person_pos[0], person_pos[1]
                            distance = np.sqrt((center_x - person_x)**2 + (center_y - person_y)**2)
                            if distance < 100:  # If person is near item
                                is_near_person = True
                                break
                        
                        # Check if any person is detected in frame (not just near this item)
                        person_count = len(people_positions)
                        
                        if person_count >= 1:
                            # Person(s) detected in frame - item is ACTIVE
                            if class_name.lower() in ['cell phone', 'keys']:
                                label += f" - 🔥 ACTIVE ({person_count} person)"
                            elif class_name.lower() in ['backpack', 'suitcase']:
                                label += f" - 🎒 ACTIVE ({person_count} person)"
                            elif class_name.lower() == 'bottle':
                                label += f" - 🍼 ACTIVE ({person_count} person)"
                            else:
                                label += f" - ACTIVE ({person_count} person)"
                        else:
                            # No person detected - check if it's in grace period or truly INACTIVE
                            if hasattr(self.alert_system, 'unattached_start_time') and persistent_id in self.alert_system.unattached_start_time:
                                # Check grace period
                                unattached_duration = time.time() - self.alert_system.unattached_start_time[persistent_id]
                                if unattached_duration < 10.0:
                                    # Still in grace period
                                    remaining_time = 10.0 - unattached_duration
                                    if class_name.lower() in ['cell phone', 'keys']:
                                        label += f" - 🔥 GRACE PERIOD ({remaining_time:.0f}s)"
                                    elif class_name.lower() in ['backpack', 'suitcase']:
                                        label += f" - 🎒 GRACE PERIOD ({remaining_time:.0f}s)"
                                    elif class_name.lower() == 'bottle':
                                        label += f" - 🍼 GRACE PERIOD ({remaining_time:.0f}s)"
                                    else:
                                        label += f" - GRACE PERIOD ({remaining_time:.0f}s)"
                                else:
                                    # Grace period expired - truly INACTIVE
                                    if class_name.lower() in ['cell phone', 'keys']:
                                        label += " - 🔥 INACTIVE (no person)"
                                    elif class_name.lower() in ['backpack', 'suitcase']:
                                        label += " - 🎒 INACTIVE (no person)"
                                    elif class_name.lower() == 'bottle':
                                        label += " - 🍼 INACTIVE (no person)"
                                    else:
                                        label += " - INACTIVE (no person)"
                            else:
                                # Not yet tracked as unattached
                                if class_name.lower() in ['cell phone', 'keys']:
                                    label += " - 🔥 ACTIVE"
                                elif class_name.lower() in ['backpack', 'suitcase']:
                                    label += " - 🎒 ACTIVE"
                                elif class_name.lower() == 'bottle':
                                    label += " - 🍼 ACTIVE"
                                else:
                                    label += " - ACTIVE"
                    
                    # Adjust text position to avoid going off screen
                    text_x = max(int(box[0]), 10)
                    text_y = max(int(box[1]) - 10, 25)
                    
                    cv2.putText(frame, label, (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Update alert system with people positions and all items
        self.alert_system.update_people_positions(people_positions, people_track_ids)
        self.alert_system.update_items(all_items, people_positions, people_track_ids)
        
        return frame, unattended_items
    
    def is_stationary(self, track_id, threshold=30):
        """Check if an object has been stationary for a while."""
        if track_id not in self.object_positions or len(self.object_positions[track_id]) < 10:
            return False
        
        positions = self.object_positions[track_id]
        
        # Calculate movement over the last 10 positions (0.33 seconds at 30 fps) for better FPS
        if len(positions) >= 10:
            recent_positions = positions[-10:]
            first_pos = recent_positions[0]
            last_pos = recent_positions[-1]
            
            # Calculate total movement
            movement = np.sqrt((last_pos[0] - first_pos[0])**2 + (last_pos[1] - first_pos[1])**2)
            
            # Also check time duration
            time_duration = last_pos[2] - first_pos[2]
            
            return movement < threshold and time_duration > 0.3
        
        return False
    
    def is_primary_item(self, class_name):
        """Check if an item is in our primary focus list."""
        return class_name.lower() in self.primary_classes
    
    def is_high_priority_item(self, class_name):
        """Check if an item is high priority for immediate alerts."""
        return class_name.lower() in self.high_priority_items
    
    def get_enhanced_detection_settings(self, class_name):
        """Get enhanced detection settings for specific items."""
        class_lower = class_name.lower()
        
        # Enhanced settings for primary items
        if class_lower == 'person':
            return {
                'min_confidence': 0.35,
                'stationary_threshold': 25,  # More sensitive for people
                'alert_threshold': 20,  # Faster alerts for people leaving items
                'color': (0, 255, 0),
                'thickness': 3
            }
        elif class_lower == 'backpack':
            return {
                'min_confidence': 0.45,
                'stationary_threshold': 20,  # More sensitive for backpacks
                'alert_threshold': 15,  # Faster alerts for backpacks
                'color': (0, 0, 255),
                'thickness': 3
            }
        elif class_lower == 'suitcase':
            return {
                'min_confidence': 0.45,
                'stationary_threshold': 20,  # More sensitive for suitcases
                'alert_threshold': 15,  # Faster alerts for suitcases
                'color': (0, 0, 255),
                'thickness': 3
            }
        elif class_lower == 'cell phone':
            return {
                'min_confidence': 0.65,
                'stationary_threshold': 15,  # Very sensitive for phones
                'alert_threshold': 10,  # Very fast alerts for phones
                'color': (0, 0, 255),
                'thickness': 4
            }
        elif class_lower == 'bottle':
            return {
                'min_confidence': 0.45,
                'stationary_threshold': 25,  # Sensitive for bottles
                'alert_threshold': 20,  # Fast alerts for bottles
                'color': (0, 165, 255),
                'thickness': 2
            }
        elif class_lower == 'keys':
            return {
                'min_confidence': 0.7,
                'stationary_threshold': 10,  # Very sensitive for keys
                'alert_threshold': 8,  # Very fast alerts for keys
                'color': (0, 0, 255),
                'thickness': 4
            }
        else:
            # Default settings for other items
            return {
                'min_confidence': 0.6,
                'stationary_threshold': 30,
                'alert_threshold': 30,
                'color': (255, 255, 0),
                'thickness': 1
            }
    
    def optimize_for_fps(self, current_fps):
        """Dynamically adjust settings based on current FPS."""
        if current_fps < 15:  # If FPS is too low
            # Increase frame skipping
            self.process_every_n_frames = min(self.process_every_n_frames + 1, 3)
            # Reduce max detections
            #self.model.confidence_thresholds = min(self.model.confidence_thresholds + 0.1, 0.8)
            print(f"⚠ Low FPS detected ({current_fps:.1f}). Optimizing...")
        elif current_fps > 25:  # If FPS is good
            # Reduce frame skipping
            self.process_every_n_frames = max(self.process_every_n_frames - 1, 1)
            # Increase max detections
            #self.model.confidence_thresholds = max(self.model.confidence_thresholds - 0.05, 0.5)
    
    def calculate_object_similarity(self, obj1_pos, obj1_class, obj2_pos, obj2_class):
        """Calculate similarity between two objects for reappearance detection."""
        # Position similarity (distance-based)
        distance = np.sqrt((obj1_pos[0] - obj2_pos[0])**2 + (obj1_pos[1] - obj2_pos[1])**2)
        
        # Class similarity (same class = higher similarity)
        class_similarity = 1.0 if obj1_class == obj2_class else 0.3
        
        # Combined similarity score
        position_score = max(0, 1 - (distance / 100))  # Normalize distance
        total_similarity = (position_score * 0.7) + (class_similarity * 0.3)
        
        return total_similarity, distance
    
    def find_reappeared_object(self, current_pos, current_class):
        """Find if a disappeared object has reappeared."""
        best_match = None
        best_similarity = 0
        
        for disappeared_id, disappeared_info in self.disappeared_objects.items():
            disappeared_pos = disappeared_info['last_position']
            disappeared_class = disappeared_info['class']
            
            similarity, distance = self.calculate_object_similarity(
                current_pos, current_class, 
                disappeared_pos, disappeared_class
            )
            
            # Check if this could be the same object
            if (similarity > 0.6 and distance < self.reappearance_threshold and 
                current_class == disappeared_class):
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = disappeared_id
        
        return best_match
    
    def assign_persistent_id(self, track_id, class_name, position):
        """Assign a persistent ID to an object, maintaining consistency."""
        # Check if this track_id already has a persistent ID
        if track_id in self.tracked_objects:
            return self.tracked_objects[track_id]
        
        # Check if this could be a reappeared object
        reappeared_id = self.find_reappeared_object(position, class_name)
        if reappeared_id:
            # Reuse the ID of the disappeared object
            persistent_id = reappeared_id
            # Remove from disappeared objects
            del self.disappeared_objects[reappeared_id]
            print(f"🔄 Object {class_name} reappeared with ID {persistent_id}")
        else:
            # Assign new persistent ID
            persistent_id = self.next_object_id
            self.next_object_id += 1
        
        # Store the mapping
        self.tracked_objects[track_id] = persistent_id
        self.object_history[persistent_id] = {
            'class': class_name,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'total_frames': 1
        }
        
        return persistent_id
    
    def update_object_history(self, persistent_id, position):
        """Update object history with current position."""
        if persistent_id in self.object_history:
            self.object_history[persistent_id]['last_seen'] = time.time()
            self.object_history[persistent_id]['total_frames'] += 1
            self.object_history[persistent_id]['last_position'] = position
    
    def cleanup_disappeared_objects(self):
        """Clean up objects that have been disappeared for too long."""
        current_time = time.time()
        to_remove = []
        
        for disappeared_id, disappeared_info in self.disappeared_objects.items():
            frames_disappeared = disappeared_info['frames_disappeared']
            if frames_disappeared > self.max_disappeared_frames:
                to_remove.append(disappeared_id)
        
        for obj_id in to_remove:
            del self.disappeared_objects[obj_id]
            print(f"🗑️ Object ID {obj_id} permanently removed (disappeared too long)")
    
    def track_disappeared_objects(self):
        """Track objects that have disappeared from current frame."""
        current_track_ids = set(self.tracked_objects.keys())
        
        # Find objects that were tracked before but not in current frame
        for track_id, persistent_id in list(self.tracked_objects.items()):
            if track_id not in current_track_ids:
                # Object disappeared
                if persistent_id in self.object_history:
                    last_position = self.object_history[persistent_id].get('last_position', (0, 0))
                    self.disappeared_objects[persistent_id] = {
                        'class': self.object_history[persistent_id]['class'],
                        'last_position': last_position,
                        'frames_disappeared': 0
                    }
                    print(f"👻 Object {self.object_history[persistent_id]['class']} (ID: {persistent_id}) disappeared")
                
                # Remove from current tracking
                del self.tracked_objects[track_id]
        
        # Update disappeared frame counts
        for disappeared_id in self.disappeared_objects:
            self.disappeared_objects[disappeared_id]['frames_disappeared'] += 1
    
    def run_webcam(self):
        """Run the detection system using webcam."""
        print("📹 Starting webcam mode...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Set webcam properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
        
        print("✓ Webcam opened successfully")
        
        # Frame rate calculation
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from webcam")
                break
            
            # Calculate FPS (optimized for better performance)
            frame_count += 1
            if frame_count % 15 == 0:  # Update FPS every 15 frames for better responsiveness
                current_time = time.time()
                fps = 15 / (current_time - start_time)
                start_time = current_time
            
            # Process frame
            processed_frame, unattended_items = self.process_frame(frame)
            
            # Optimize FPS dynamically and cleanup disappeared objects
            if frame_count % 30 == 0:  # Check every 30 frames
                self.optimize_for_fps(fps)
                self.cleanup_disappeared_objects()
            
            # Add status text with accurate counts
            total_objects = len(self.object_positions)
            
            # Count different types of objects accurately (only wanted classes)
            person_count = sum(1 for track_id in self.object_positions.keys() 
                             if self.object_classes.get(track_id, '').lower() == 'person')
            item_count = sum(1 for track_id in self.object_positions.keys() 
                           if self.object_classes.get(track_id, '').lower() in self.unattended_items)
            
            # Count primary items specifically
            primary_items = {}
            for track_id in self.object_positions.keys():
                class_name = self.object_classes.get(track_id, '').lower()
                if class_name in self.primary_classes:
                    if class_name not in primary_items:
                        primary_items[class_name] = 0
                    primary_items[class_name] += 1
            
            # Add persistent tracking info
            persistent_objects = len(self.object_history)
            disappeared_count = len(self.disappeared_objects)
            
            status_text = f"Total: {total_objects} | People: {person_count} | Items: {item_count}"
            
            # Add primary items count
            if primary_items:
                primary_text = " | Primary: "
                primary_list = []
                for item_type, count in primary_items.items():
                    if item_type == 'person':
                        primary_list.append(f"👤{count}")
                    elif item_type == 'backpack':
                        primary_list.append(f"🎒{count}")
                    elif item_type == 'suitcase':
                        primary_list.append(f"💼{count}")
                    elif item_type == 'cell phone':
                        primary_list.append(f"📱{count}")
                    elif item_type == 'bottle':
                        primary_list.append(f"🍾{count}")
                    elif item_type == 'keys':
                        primary_list.append(f"🔑{count}")
                    else:
                        primary_list.append(f"{item_type}{count}")
                status_text += primary_text + ", ".join(primary_list)
            
            status_text += f" | Persistent IDs: {persistent_objects}"
            if disappeared_count > 0:
                status_text += f" | Disappeared: {disappeared_count}"
            if unattended_items:
                status_text += f" | Lost Items: {len(unattended_items)}"
            
            # Add active alerts count
            active_alerts = len(self.alert_system.alerts)
            if active_alerts > 0:
                status_text += f" | Active Alerts: {active_alerts}"
            
            # Add owner associations count
            owner_associations = len(self.alert_system.item_owner_associations)
            if owner_associations > 0:
                status_text += f" | Owner Associations: {owner_associations}"
            
            # Add FPS display
            fps_text = f"FPS: {fps:.1f}"
            
            # Add proximity detection info (only if people detected for better FPS)
            if self.alert_system.people_positions:
                proximity_text = f"People: {len(self.alert_system.people_positions)}"
                cv2.putText(processed_frame, proximity_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Optimized text rendering (reduced font size and thickness)
            cv2.putText(processed_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(processed_frame, fps_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Display frame
            cv2.imshow('AI Lost Item Detector', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"lost_item_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"📸 Screenshot saved as {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Webcam mode ended")
    
    def run_video(self, video_path, fast_mode=False):
        """Run the detection system on a video file."""
        print(f"🎥 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps} fps, Total frames: {total_frames}")
        
        frame_count = 0
        processed_count = 0
        
        if fast_mode:
            print("⚡ ULTRA-FAST mode: No display, maximum frame skipping")
            skip_frames = 10  # Process every 10th frame
            display_frames = False
        else:
            print("🚀 Optimized video processing - skipping frames for speed")
            skip_frames = 5   # Process every 5th frame
            display_frames = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for faster processing
            if frame_count % skip_frames != 0:
                continue
            
            processed_count += 1
            
            # Show progress every 50 processed frames
            if processed_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"\rProgress: {progress:.1f}% ({processed_count}/{total_frames//skip_frames} processed)", end="", flush=True)
            
            # Process frame
            processed_frame, unattended_items = self.process_frame(frame)
            
            # Display frame only if not in fast mode
            if display_frames and processed_count % 10 == 0:
                cv2.imshow('AI Lost Item Detector', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if display_frames:
            cv2.destroyAllWindows()
        
        print(f"\n✓ Video processing completed. Processed {processed_count} frames out of {frame_count} total frames")
        print(f"⚡ Speed optimization: {skip_frames}x faster processing")
    
    def check_sample_videos(self):
        """Check for existing sample videos and list them."""
        sample_videos = []
        common_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        # Check current directory for video files
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in common_extensions):
                sample_videos.append(file)
        
        if sample_videos:
            print("📹 Found existing video files:")
            for i, video in enumerate(sample_videos, 1):
                print(f"  {i}. {video}")
            return sample_videos
        else:
            print("📹 No video files found in current directory")
            return []
    
    def create_sample_video(self):
        """Create a realistic sample video for testing."""
        print("🎬 Creating realistic sample video...")
        
        # Create a more realistic video with people and objects
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('sample_video.mp4', fourcc, 30.0, (640, 480))
        
        for i in range(450):  # 15 seconds at 30 fps
            # Create frame with background
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add background color (light gray)
            frame[:] = (200, 200, 200)
            
            # Add a person (moving)
            person_x = int(100 + i * 1.5) % 500
            person_y = 300
            # Person body (rectangle)
            cv2.rectangle(frame, (person_x, person_y), (person_x + 30, person_y + 60), (0, 255, 0), -1)
            # Person head (circle)
            cv2.circle(frame, (person_x + 15, person_y - 10), 15, (0, 255, 0), -1)
            
            # Add a backpack (following the person)
            backpack_x = person_x - 10
            backpack_y = person_y + 20
            cv2.rectangle(frame, (backpack_x, backpack_y), (backpack_x + 50, backpack_y + 30), (0, 0, 255), -1)
            
            # Add a stationary laptop (lost item)
            if i > 150:  # Start appearing after 5 seconds
                cv2.rectangle(frame, (400, 200), (500, 250), (0, 0, 255), -1)
                cv2.putText(frame, "LAPTOP", (410, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add a moving cell phone (person drops it)
            if i > 200 and i < 350:  # Phone appears and disappears
                phone_x = int(200 + (i - 200) * 0.5)
                phone_y = 350
                cv2.rectangle(frame, (phone_x, phone_y), (phone_x + 20, phone_y + 30), (255, 0, 0), -1)
                cv2.putText(frame, "PHONE", (phone_x, phone_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Add a chair (stationary furniture)
            cv2.rectangle(frame, (50, 150), (120, 200), (128, 128, 128), -1)
            cv2.rectangle(frame, (50, 200), (120, 250), (128, 128, 128), -1)
            cv2.putText(frame, "CHAIR", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add some text to show it's a sample
            cv2.putText(frame, f"Sample Video - Frame {i}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(frame)
        
        out.release()
        print("✓ Realistic sample video created: sample_video.mp4")
        print("   - Contains: Person, Backpack, Laptop (lost), Phone, Chair")
        print("   - Duration: 15 seconds")
        print("   - Resolution: 640x480 @ 30fps")
    
    def create_simple_sample_video(self):
        """Create a simple sample video for quick testing."""
        print("🎬 Creating simple sample video...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('simple_sample.mp4', fourcc, 30.0, (640, 480))
        
        for i in range(300):  # 10 seconds at 30 fps
            # Create frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add moving rectangle (simulating a person)
            x = int(50 + i * 2) % 540
            y = 240
            cv2.rectangle(frame, (x, y), (x + 50, y + 50), (0, 255, 0), -1)
            
            # Add stationary rectangle (simulating a lost item)
            cv2.rectangle(frame, (300, 200), (350, 250), (0, 0, 255), -1)
            
            # Add text
            cv2.putText(frame, "Simple Sample", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print("✓ Simple sample video created: simple_sample.mp4")

def main():
    """Main function to run the lost item detection system."""
    parser = argparse.ArgumentParser(description='AI-Based Lost Item Finder')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for detection')
    parser.add_argument('--video', type=str, help='Path to video file for detection')
    parser.add_argument('--list-videos', action='store_true', help='List available video files in current directory')
    parser.add_argument('--performance', action='store_true', help='Enable performance mode (lower resolution, faster processing)')
    parser.add_argument('--fast-video', action='store_true', help='Ultra-fast video processing (no display, max frame skipping)')
    parser.add_argument('--server-url', type=str, default=DEFAULT_SERVER_URL, help='Server base URL to post lost events')
    parser.add_argument('--location', type=str, default=DEFAULT_LOCATION, help='Location/venue label to include in events')
    
    args = parser.parse_args()
    
    # Initialize detector
    # Inject server config into alert system after detector is constructed
    detector = LostItemDetector()
    detector.alert_system.server_url = args.server_url.rstrip('/')
    detector.alert_system.location = args.location
    
    # Apply performance mode settings if requested
    if args.performance:
        detector.target_fps = 60  # Higher target FPS
        detector.frame_skip = 3   # More aggressive frame skipping
        print("🚀 Performance mode enabled")
    
    if args.list_videos:
        # List available video files
        detector.check_sample_videos()
        return
    elif args.webcam:
        detector.run_webcam()
    elif args.video:
        # Check if the specified video file exists
        if os.path.exists(args.video):
            detector.run_video(args.video, fast_mode=args.fast_video)
        else:
            print(f"❌ Error: Video file '{args.video}' not found")
            print("Available video files:")
            detector.check_sample_videos()
    else:
        # Interactive mode - let user choose
        print("🎥 AI-Based Lost Item Detector")
        print("=" * 40)
        print("Choose your input source:")
        print("1. Webcam")
        print("2. Video file")
        print("3. List available video files")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            detector.run_webcam()
        elif choice == '2':
            # Check for available video files
            video_files = detector.check_sample_videos()
            if video_files:
                print("\nEnter the video filename (or full path):")
                video_path = input("Video file: ").strip()
                
                # Remove quotes if user added them
                video_path = video_path.strip('"\'')
                
                if os.path.exists(video_path):
                    detector.run_video(video_path)
                else:
                    print(f"❌ Error: Video file '{video_path}' not found")
                    print("Please check the filename and try again.")
            else:
                print("❌ No video files found in current directory")
                print("Please place a video file in the current directory or provide a full path.")
        elif choice == '3':
            detector.check_sample_videos()
        elif choice == '4':
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Please run the program again.")

if __name__ == "__main__":
    main()