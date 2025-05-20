import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
import random
import numpy as np
from typing import Optional, Dict, Tuple
from High_Society import HighSocietyGame
from agents import RandomAI


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
        # self.mask = np.zeros(self.action_space.n, dtype=np.float32) # Mask for masking actions. It's here initialized to save computation time.

    def _create_move_mapping(self):
        return {
            tuple(sorted(move)) if isinstance(move, list) else move: i
            for i, move in enumerate(self.possible_moves)
        }

    def _create_observation_space(self):
        return spaces.Dict({
            "action_mask": spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.float32),
            "num_players": spaces.Discrete(3),
            "current_bid": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "bidding_card": spaces.Discrete(14),
            "player_money": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "player_hand": spaces.Box(0, 1, shape=(14,), dtype=np.float32),
            "player_score": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "highest_score": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "poorest_money": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "score_of_winner": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "red_cards": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
        })

    def _get_observation(self):
        if not self.game.players: # Error handling
            return self._empty_observation()
            
        raw_state = self.game.get_game_state(self.game.current_player)

        return {
            "action_mask": self._create_action_mask(),
            "num_players": self.num_players - 3,
            "current_bid": np.array([raw_state["current_bid"] / 106], dtype=np.float32),
            "bidding_card": self._encode_card(raw_state["bidding_card"]),
            "player_money": np.array([raw_state["current_player_money"] / 106], dtype=np.float32),
            "player_hand": self._encode_hand(raw_state["current_player_hand"]),
            "player_score": np.array([raw_state["current_player_score"] / 440], dtype=np.float32),
            "highest_score": np.array([raw_state["highest_score"] / 440], dtype=np.float32),
            "poorest_money": np.array([raw_state["poorest_player_money"] / 106], dtype=np.float32),
            "score_of_winner": np.array([raw_state["score_of_winner"] / 440], dtype=np.float32),
            "red_cards": np.array([raw_state["number_of_red_cards"] / 4], dtype=np.float32),
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
        # Reset step counter for dynamic punishment
        self._episode_steps = 0
        super().reset(seed=seed)
        self.num_players = random.randint(3, 5)
        self.player_names = [f"Player_{i}" for i in range(self.num_players)]
        self.ai_agents = [self._create_dummy_agent()]
        self.ai_agents.extend(RandomAI() for _ in range(self.num_players - 1))

        self.game = HighSocietyGame(
            player_names=self.player_names,
            ai_agents=self.ai_agents,
            debug_output=False
        )
        self.game._terminate_episode = False
        self.game.winner = None
        self.game.start_game()  # Use new API
        # Start the first round
        if not self.game.is_game_over():
            self.game.start_round()
        return self._get_observation(), {}

    def step(self, action_idx: int) -> Tuple[Dict, float, bool, bool, Dict]:
        # Count this agent step for punishment scaling
        self._episode_steps += 1
        move = self.action_to_move[action_idx]
        self._handle_agent_action(move)
        done = False
        # Only step if game is not over
        if not self.game.is_game_over():
            result, _ = self.game.step(move)
            if result == 'illegal':
                # Episode ends on illegal move
                done = True
                obs = self._get_observation()
                # Apply scaled punishment for illegal move
                reward = self._calculate_reward(done)
                return obs, reward, done, False, {}
            # If round ended, start next round if game not over
            round_end_counter = 0
            while result == 'round_end' and not self.game.is_game_over():
                round_end_counter += 1
                if round_end_counter > 1000:
                    break
                self.game.start_round()
                # If after starting round, it's not RL agent's turn, auto-step AIs
                ai_step_counter = 0
                while not self.game.is_trainee_turn() and not self.game.is_game_over():
                    ai_step_counter += 1
                    if ai_step_counter > 1000:
                        break
                    ai_action = self.game.current_player['agent'].choose_action(
                        self.game.get_game_state(self.game.current_player),
                        self.game.get_legal_moves(self.game.current_player)
                    )
                    result, _ = self.game.step(ai_action)
                    if result == 'round_end' and not self.game.is_game_over():
                        self.game.start_round()
        obs = self._get_observation()
        done = self.game.is_game_over()
        reward = self._calculate_reward(done)
        return obs, reward, done, False, {}

    def _calculate_reward(self, done):
        # Reward logic is now based on player index 0
        if self.game._terminate_episode:  # Punishment for illegal moves
            # Scale penalty by number of steps taken (higher if early failure)
            if hasattr(self, '_episode_steps') and self._episode_steps > 0:
                return -1000.0 / (self._episode_steps)**2
            return -1000.0

        elif done:
            # Check if player 0 is the winner
            if self.game.players[0] == self.game.winner:
                return 100.0
            else:
                return -10.0

        else:
            # Intermediate rewards
            reward = 0.0
            
            # Encourage bidding (spending money)
            money_left = sum(self.game.players[0]['money'])
            if self.game.poorest_player != self.game.players[0]:
                reward -= 0.02 * money_left  # Penalize hoarding money
            
            # Reward/punish relative position
            if self.game.players[0] == self.game.highest_score_non_lowest_money:
                reward += 2.0  # Strong reward for leading without being poorest
            elif (self.game.players[0] == self.game.poorest_player) and self.game.player_with_highest_score != self.game.players[0]:
                reward -= 1.0  # Penalty for being poorest
            
            return reward

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
            def choose_action(self, game_state=None, legal_moves=None):
                return self.last_action or "pass"
        return DummyAgent()

    def _handle_agent_action(self, move):
        self.ai_agents[0].last_action = move