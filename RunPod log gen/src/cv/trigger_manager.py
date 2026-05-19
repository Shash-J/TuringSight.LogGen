import time
from typing import List, Dict, Any

class TriggerManager:
    """
    Monitors the CustomCVMemory state and fires events when heuristics are met.
    Caps the number of events per evaluation cycle to prevent queue flooding.
    """
    def __init__(self, stationary_trigger_sec: int = 15, cooldown_sec: int = 60,
                 max_events_per_cycle: int = 3):
        self.stationary_trigger_sec = stationary_trigger_sec
        self.cooldown_sec = cooldown_sec
        self.max_events_per_cycle = max_events_per_cycle
        
        # Track last trigger times per entity to avoid spamming
        # Format: {entity_id: {"stationary": last_triggered_ts, "new_arrival": ts}}
        self.last_triggers: Dict[int, Dict[str, float]] = {}
        # Global cooldown to prevent initial burst
        self._initialized_ts: float = 0.0
        
    def check_triggers(self, cv_state: Dict[int, Dict[str, Any]], current_ts: float) -> List[Dict[str, Any]]:
        """
        Evaluates the current CV memory state and returns a list of triggered events.
        Caps output to max_events_per_cycle to avoid flooding the VLM queue.
        """
        # Skip the first 10 seconds after startup to let tracking stabilize
        if self._initialized_ts == 0.0:
            self._initialized_ts = current_ts
        if (current_ts - self._initialized_ts) < 10.0:
            return []
            
        active_count = len(cv_state)
        if active_count > 0:
            print(f"[DEBUG-CV] Tracking {active_count} entities in frame.")

        events = []
        
        for entity_id, data in cv_state.items():
            if len(events) >= self.max_events_per_cycle:
                break
                
            if entity_id not in self.last_triggers:
                self.last_triggers[entity_id] = {"stationary": 0.0, "new_arrival": 0.0}
                
            trigger_history = self.last_triggers[entity_id]
            
            # Heuristic 1: Stationary Object/Person
            if data.get("is_stationary") and data.get("stationary_since"):
                time_stationary = current_ts - data["stationary_since"]
                time_since_last_trigger = current_ts - trigger_history["stationary"]
                
                if time_stationary >= self.stationary_trigger_sec and time_since_last_trigger >= self.cooldown_sec:
                    events.append({
                        "event_type": "stationary_entity",
                        "entity_id": entity_id,
                        "cv_state": data,
                        "trigger_ts": current_ts,
                        "time_stationary_sec": round(time_stationary, 1)
                    })
                    trigger_history["stationary"] = current_ts
                    print(f"[TRIGGER] stationary_entity: entity {entity_id} stationary for {time_stationary:.0f}s")
                    
            # Heuristic 2: New Arrival 
            # (Wait 5 seconds after they appear to ensure we have a good buffer of frames)
            time_in_frame = current_ts - data.get("first_seen", current_ts)
            if time_in_frame >= 5.0 and trigger_history["new_arrival"] == 0.0:
                events.append({
                    "event_type": "new_arrival",
                    "entity_id": entity_id,
                    "cv_state": data,
                    "trigger_ts": current_ts,
                    "time_in_frame_sec": round(time_in_frame, 1)
                })
                trigger_history["new_arrival"] = current_ts
                print(f"[TRIGGER] new_arrival: entity {entity_id} in frame for {time_in_frame:.0f}s")
                
        # Clean up last_triggers for entities no longer in cv_state to prevent memory leaks
        dead_ids = [eid for eid in self.last_triggers if eid not in cv_state]
        for eid in dead_ids:
            del self.last_triggers[eid]
            
        return events
