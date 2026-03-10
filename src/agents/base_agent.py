"""
Base Agent class for the Agentic Object Detection System.
Provides common functionality for all agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path
import json
import time


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Provides common methods for observation, action selection, and learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            config: Configuration dictionary containing agent settings
        """
        self.config = config
        self.name = self.__class__.__name__
        self.obs_history: List[Dict] = []
        self.action_history: List[Dict] = []
        self.reward_history: List[float] = []
        self.episode_count = 0
        self.total_steps = 0
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup agent-specific components. Override in subclasses."""
        self.logger = self._get_logger()
        self.memory = None  # To be set by subclass
        self.policy = None  # To be set by subclass
    
    def _get_logger(self):
        """Get logger for the agent."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        
        import logging
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def observe(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Process the current state into an observation vector.
        
        Args:
            state: Raw state dictionary from environment
            
        Returns:
            Processed observation as numpy array
        """
        pass
    
    @abstractmethod
    def act(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Select an action based on the observation.
        
        Args:
            observation: Processed observation vector
            
        Returns:
            Action dictionary with action type and parameters
        """
        pass
    
    @abstractmethod
    def learn(self, observation: np.ndarray, action: Dict, reward: float, 
              next_obs: np.ndarray, done: bool):
        """
        Update agent based on experience.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode is done
        """
        pass
    
    def store_experience(self, observation: np.ndarray, action: Dict, 
                        reward: float, next_obs: np.ndarray, done: bool):
        """
        Store experience in memory.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode is done
        """
        experience = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_obs,
            "done": done,
            "timestamp": time.time()
        }
        
        self.obs_history.append(observation)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.total_steps += 1
        
        # Store in memory if available
        if self.memory is not None:
            if hasattr(self.memory, "add"):
                self.memory.add(experience)
            elif hasattr(self.memory, "add_experience"):
                self.memory.add_experience(
                    observation,
                    action,
                    reward,
                    next_obs
                )
    
    def reset_episode(self):
        """Reset agent state for a new episode."""
        self.episode_count += 1
        self.logger.info(f"Episode {self.episode_count} started")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "name": self.name,
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "avg_reward": np.mean(self.reward_history) if self.reward_history else 0.0,
            "total_reward": sum(self.reward_history)
        }
    
    def save(self, path: str):
        """Save agent state to file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config,
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "name": self.name
        }
        
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Agent state saved to {path}")
    
    def load(self, path: str):
        """Load agent state from file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.episode_count = state.get("episode_count", 0)
        self.total_steps = state.get("total_steps", 0)
        self.logger.info(f"Agent state loaded from {path}")
    
    def __repr__(self) -> str:
        return f"{self.name}(episodes={self.episode_count}, steps={self.total_steps})"
