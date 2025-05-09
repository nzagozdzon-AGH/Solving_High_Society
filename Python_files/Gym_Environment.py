import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
from typing import Optional, Dict, Tuple
from High_Society import HighSocietyGame
from agents import RandomAI, RLagent

class HighSocietyEnv(gym.Env):
    ALL_MONEY_CARDS = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25]
    CARD_ENCODING = {
        **{i: i-1 for i in range(1, 11)},
        -5: 10,
        'double': 11,
        'half': 12,
        'drop_card': 13
    }

    def __init__(self, num_players: int = 4):
        super().__init__()
        self.num_players = num_players
        self.player_names = [f"Player_{i}" for i in range(self.num_players)]
        
        # Action space configuration
        self.possible_moves = self._generate_all_possible_moves()
        self.action_to_move = {i: move for i, move in enumerate(self.possible_moves)}
        self.move_to_action = self._create_move_mapping()
        
        # Define spaces
        self.action_space = spaces.Discrete(len(self.possible_moves))
        self.observation_space = self._create_observation_space()
        self.game = None

    def _create_move_mapping(self):
        return {
            tuple(sorted(move)) if isinstance(move, list) else move: i
            for i, move in enumerate(self.possible_moves)
        }

    def _create_observation_space(self):
        return spaces.Dict({
            "action_mask": spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.float32),
            "num_players": spaces.Discrete(4),
            "current_bid": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "bidding_card": spaces.Discrete(14),
            "player_money": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "player_hand": spaces.Box(0, 1, shape=(14,), dtype=np.float32),
            "player_score": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "other_players": spaces.Box(0, 1, shape=(4, 2), dtype=np.float32),
            "deck_remaining": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
        })

    def _get_observation(self):
        if not self.game.players: # Error handling
            return self._empty_observation()
            
        raw_state = self.game.get_game_state(self.game.current_player)

        # Build mask, that will hold 0 for non existent players
        other_players = np.zeros((4, 2), dtype=np.float32)
        
        # Filling table with information about other players
        for i, p in enumerate(raw_state["other_players_states"]): 
            other_players[i] = [
                p['score'] / 440,  # Normalized score
                p['money'] / 106   # Normalized money (max 1+2+...+25=106)
            ]
            
        return {
            "action_mask": self._create_action_mask(),
            "num_players": self.num_players - 3,
            "current_bid": np.array([raw_state["current_bid"] / 106], dtype=np.float32),
            "bidding_card": self._encode_card(raw_state["bidding_card"]),
            "player_money": np.array([raw_state["current_player_money"] / 106], dtype=np.float32),
            "player_hand": self._encode_hand(raw_state["current_player_hand"]),
            "player_score": np.array([raw_state["current_player_score"] / 440], dtype=np.float32),
            "other_players": other_players,
            "deck_remaining": np.array([raw_state["remaining_deck_size"] / 16], dtype=np.float32),
        }

    def _generate_all_possible_moves(self):
        from itertools import combinations
        moves = ["pass"]  # Start with pass move
        
        # Add all combinations of 1-4 cards
        for r in range(1, 5):  # 1 to 4 cards
            for combo in combinations(self.ALL_MONEY_CARDS, r):
                moves.append(list(combo))
        
        return moves

    def _create_action_mask(self):
        # Initialize mask with zeros for all possible actions
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        
        # If game hasn't started or is over, return all-zero mask
        if not self.game.players:
            return mask
            
        # Get legal moves for the current player
        legal_moves = self.game.get_legal_moves(self.game.current_player)
        
        # Filter and mark legal moves in the mask
        for move in legal_moves:
            # Only consider moves with length <= 4
            if isinstance(move, list) and len(move) > 4:
                continue
                
            # Convert move to the format used in move_to_action mapping
            move_key = tuple(sorted(move)) if isinstance(move, list) else move
            
            # If move exists in our action mapping, mark it as valid (1.0)
            if move_key in self.move_to_action:
                mask[self.move_to_action[move_key]] = 1.0
                
        return mask

    def _encode_card(self, card):
        return self.CARD_ENCODING.get(card, 0)

    def _encode_hand(self, hand):
        encoding = np.zeros(14, dtype=np.float32)
        for card in hand:
            encoding[self._encode_card(card)] += 1.0
        return encoding / 3  # Max 3 copies of any card
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.num_players = random.randint(3, 5)
        # Update player names based on new number of players
        self.player_names = [f"Player_{i}" for i in range(self.num_players)]
        self.ai_agents = [self._create_dummy_agent()]
        self.ai_agents.extend(RandomAI() for i in range(self.num_players - 1))

        self.game = HighSocietyGame(
            player_names=self.player_names,
            ai_agents=self.ai_agents,
            debug_output=False
        )
        self.game._terminate_episode = False  # Resets flag
        self.game.winner = None # Resets winner
        self.game.play_game()
        return self._get_observation(), {}

    def step(self, action_idx: int) -> Tuple[Dict, float, bool, bool, Dict]:
        move = self.action_to_move[action_idx]
        self._handle_agent_action(move)
        
        done = False
        while not done and not self._is_agent_turn():
            self.game.round_start()
            done = self.game.is_game_over()

        obs = self._get_observation()
        reward = self._calculate_reward(done)
        return obs, reward, done, False, {}

    def _calculate_reward(self, done):
        if self.game._terminate_episode == True:
            return -10.0
        
        elif done:
            if isinstance(self.game.winner['agent'], RLagent): # Checks if winner is RLagent
                if max(self.game.players_scores) == 0: # Blocks strategy of winning by always passing
                    return -0.15
                else:
                    return 1.0 
            else:
                return -0.25
        
        else:
            # Find the player with RLagent
            rl_player = next((player for player in self.game.players if isinstance(player['agent'], RLagent)), None)
            if rl_player:
                state = self.game.get_game_state(rl_player)
                money_spent = 106 - (state["current_player_money"] + state["current_player_bid"])
                return (float(state["current_player_score"] - (money_spent/5))) / 440
            return 0.0

    def _empty_observation(self):
        return {
            key: np.zeros(space.shape, dtype=space.dtype)
            if isinstance(space, spaces.Box)
            else 0
            for key, space in self.observation_space.spaces.items()
        }

    def _create_dummy_agent(self):
        class DummyAgent():
            def __init__(self, **kwargs):
                self.last_action = None
            def choose_action(self, **kwargs):
                return self.last_action or "pass"
        return DummyAgent()

    def _handle_agent_action(self, move):
        self.ai_agents[0].last_action = move

    def _is_agent_turn(self):
        return True