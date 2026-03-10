"""
Episodic Memory for the Agentic Object Detection System.
Stores and retrieves experiences for learning and reasoning.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import time
import json
from pathlib import Path


class Episode:
    """Represents a single episode/experience."""
    
    def __init__(self, episode_id: int):
        self.episode_id = episode_id
        self.states: List[np.ndarray] = []
        self.actions: List[Dict] = []
        self.rewards: List[float] = []
        self.observations: List[np.ndarray] = []
        self.metadata: Dict = {}
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.total_reward: float = 0.0
    
    def add_experience(self, state: np.ndarray, action: Dict, reward: float,
                      observation: np.ndarray):
        """Add an experience to the episode."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations.append(observation)
        self.total_reward += reward
    
    def finish(self):
        """Mark episode as finished."""
        self.end_time = time.time()
    
    def get_duration(self) -> float:
        """Get episode duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict:
        """Convert episode to dictionary."""
        return {
            "episode_id": self.episode_id,
            "num_steps": len(self.states),
            "total_reward": self.total_reward,
            "duration": self.get_duration(),
            "actions": self.actions,
            "rewards": self.rewards,
            "metadata": self.metadata
        }


class EpisodicMemory:
    """
    Manages episodic memory for the agent.
    Stores episodes and provides retrieval based on similarity.
    """
    
    def __init__(self, max_episodes: int = 1000, max_steps_per_episode: int = 1000):
        """
        Initialize episodic memory.
        
        Args:
            max_episodes: Maximum number of episodes to store
            max_steps_per_episode: Maximum steps per episode
        """
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        
        self.episodes: deque = deque(maxlen=max_episodes)
        self.current_episode: Optional[Episode] = None
        self.episode_counter: int = 0
        
        # Statistics
        self.total_experiences: int = 0
        self.episode_rewards: List[float] = []
    
    def start_episode(self, initial_state: np.ndarray = None) -> Episode:
        """Start a new episode."""
        self.episode_counter += 1
        self.current_episode = Episode(self.episode_counter)
        
        if initial_state is not None:
            self.current_episode.states.append(initial_state)
        
        return self.current_episode
    
    def add_experience(self, state: np.ndarray, action: Dict, reward: float,
                      observation: np.ndarray):
        """Add experience to current episode."""
        if self.current_episode is None:
            self.start_episode(state)
        
        self.current_episode.add_experience(state, action, reward, observation)
        self.total_experiences += 1
        
        # Check if episode should end
        if len(self.current_episode.states) >= self.max_steps_per_episode:
            self.end_episode(reward)
    
    def end_episode(self, final_reward: float = 0.0):
        """End the current episode."""
        if self.current_episode is not None:
            self.current_episode.add_experience(
                np.array([]),  # Terminal state
                {"action": "terminate"},
                final_reward,
                np.array([])
            )
            self.current_episode.finish()
            
            # Store episode
            self.episodes.append(self.current_episode)
            self.episode_rewards.append(self.current_episode.total_reward)
            
            self.current_episode = None
    
    def get_recent_episodes(self, n: int = 10) -> List[Episode]:
        """Get the n most recent episodes."""
        return list(self.episodes)[-n:]
    
    def get_best_episodes(self, n: int = 10) -> List[Episode]:
        """Get the n episodes with highest rewards."""
        sorted_episodes = sorted(
            self.episodes, 
            key=lambda e: e.total_reward, 
            reverse=True
        )
        return sorted_episodes[:n]
    
    def get_episodes_by_reward_threshold(self, threshold: float) -> List[Episode]:
        """Get episodes with total reward above threshold."""
        return [e for e in self.episodes if e.total_reward >= threshold]
    
    def retrieve_similar(self, query_obs: np.ndarray, 
                       n: int = 5) -> List[Tuple[Episode, float]]:
        """
        Retrieve episodes with observations similar to query.
        
        Args:
            query_obs: Query observation
            n: Number of episodes to retrieve
            
        Returns:
            List of (episode, similarity_score) tuples
        """
        similarities = []
        
        for episode in self.episodes:
            if not episode.observations:
                continue
            
            # Compute average similarity
            sims = []
            for obs in episode.observations:
                if obs.size == 0 or query_obs.size == 0:
                    continue
                # Cosine similarity
                obs_norm = obs / (np.linalg.norm(obs) + 1e-8)
                query_norm = query_obs / (np.linalg.norm(query_obs) + 1e-8)
                sim = np.dot(obs_norm, query_norm)
                sims.append(sim)
            
            if sims:
                avg_sim = np.mean(sims)
                similarities.append((episode, avg_sim))
        
        # Sort by similarity and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        if not self.episodes:
            return {
                "num_episodes": 0,
                "total_experiences": self.total_experiences,
                "avg_reward": 0.0,
                "max_reward": 0.0,
                "min_reward": 0.0
            }
        
        rewards = [e.total_reward for e in self.episodes]
        
        return {
            "num_episodes": len(self.episodes),
            "total_experiences": self.total_experiences,
            "avg_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_episode_length": float(np.mean([len(e.states) for e in self.episodes])),
            "avg_duration": float(np.mean([e.get_duration() for e in self.episodes]))
        }
    
    def clear(self):
        """Clear all memory."""
        self.episodes.clear()
        self.current_episode = None
        self.episode_counter = 0
        self.total_experiences = 0
        self.episode_rewards.clear()
    
    def save(self, path: str):
        """Save memory to file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "max_episodes": self.max_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "episode_counter": self.episode_counter,
            "total_experiences": self.total_experiences,
            "episodes": [e.to_dict() for e in self.episodes]
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Memory saved to {path}")
    
    def load(self, path: str):
        """Load memory from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.max_episodes = data.get("max_episodes", 1000)
        self.max_steps_per_episode = data.get("max_steps_per_episode", 1000)
        self.episode_counter = data.get("episode_counter", 0)
        self.total_experiences = data.get("total_experiences", 0)
        
        # Recreate episodes (without actual numpy arrays for simplicity)
        self.episodes.clear()
        # Note: Full state restoration would require serializing numpy arrays
        
        print(f"Memory loaded from {path}")
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __repr__(self) -> str:
        return f"EpisodicMemory(episodes={len(self.episodes)}, experiences={self.total_experiences})"


class SemanticMemory:
    """
    Stores semantic knowledge extracted from episodes.
    Useful for storing learned concepts and patterns.
    """
    
    def __init__(self):
        """Initialize semantic memory."""
        self.concepts: Dict[str, Any] = {}
        self.model_preferences: Dict[str, Dict] = {}  # Scene -> preferred model
        self.class_statistics: Dict[str, Dict] = {}   # Class -> detection stats
    
    def add_concept(self, key: str, value: Any):
        """Add a semantic concept."""
        self.concepts[key] = value
    
    def get_concept(self, key: str, default: Any = None) -> Any:
        """Retrieve a concept."""
        return self.concepts.get(key, default)
    
    def update_model_preference(self, scene_type: str, model: str, success: bool):
        """Update model preference for a scene type."""
        if scene_type not in self.model_preferences:
            self.model_preferences[scene_type] = {"model": model, "success_count": 0, "total_count": 0}
        
        prefs = self.model_preferences[scene_type]
        prefs["total_count"] += 1
        if success:
            prefs["success_count"] += 1
        
        # Update preferred model if current one is better
        success_rate = prefs["success_count"] / prefs["total_count"]
        if success_rate > 0.7:
            prefs["model"] = model
    
    def get_preferred_model(self, scene_type: str) -> str:
        """Get preferred model for a scene type."""
        if scene_type in self.model_preferences:
            return self.model_preferences[scene_type]["model"]
        return "lightweight"  # Default
    
    def update_class_stats(self, class_name: str, score: float, bbox_size: float):
        """Update detection statistics for a class."""
        if class_name not in self.class_statistics:
            self.class_statistics[class_name] = {
                "count": 0,
                "avg_score": 0.0,
                "avg_bbox_size": 0.0
            }
        
        stats = self.class_statistics[class_name]
        n = stats["count"]
        
        # Incremental update
        stats["count"] += 1
        stats["avg_score"] = (n * stats["avg_score"] + score) / (n + 1)
        stats["avg_bbox_size"] = (n * stats["avg_bbox_size"] + bbox_size) / (n + 1)
    
    def get_class_stats(self, class_name: str) -> Dict:
        """Get statistics for a class."""
        return self.class_statistics.get(class_name, {})
    
    def __repr__(self) -> str:
        return f"SemanticMemory(concepts={len(self.concepts)}, models={len(self.model_preferences)})"
