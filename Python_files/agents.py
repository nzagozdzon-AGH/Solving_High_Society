from abc import ABC, abstractmethod
import random

class BaseAgent(ABC): # Used for validatining proper agents
    @abstractmethod
    def choose_action(self, game_state, legal_moves):
        pass

class HumanAgent(BaseAgent):
    def choose_action(self, game_state, legal_moves):
        input_str = input("Enter bid values or write pass: ").strip().lower()
        return input_str

class RandomAI(BaseAgent):
    def choose_action(self, game_state, legal_moves):
        if random.random() < 0.5:  # 50% chance
            return 'pass'
        else:
            return random.choice(legal_moves)
    
class RulesBasedAgent(BaseAgent):
    def choosing_bid(self, game_state, legal_moves, max_bid_amount):
        legal_moves_without_pass = [move for move in legal_moves if move != 'pass']

        bid_move = []
        min_bid = 1000000 # arbitraly big number, so first sum(move) will alwayes be smaller
        for move in legal_moves_without_pass:
            if (sum(move) < min_bid):
                min_bid = sum(move)
                bid_move = move

        if bid_move == []: # Handles case, when none of legal moves is greater than current bid
            return 'pass' 
        elif sum(bid_move) <= max_bid_amount:
            return bid_move
        else:
            return 'pass'


    def choose_action(self, game_state, legal_moves):
        
        max_bid_amount = 0  # Default value
        bidding_card = game_state['bidding_card']

        if (game_state['current_player_money'] < 7) or (len(legal_moves) == 2): # Proctecion to not go bust for spending too much money
            return 'pass'
        
        
        elif bidding_card in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            max_bid_amount = 3 * bidding_card
            
        elif bidding_card in ['double', 'half']:
            max_bid_amount = 40

        elif bidding_card in [-5, 'drop_card']:
            max_bid_amount = 15

        return self.choosing_bid(game_state, legal_moves, max_bid_amount)

class RLagent(BaseAgent):
    def __init__(self, model, env):
        self.model = model
        self.env = env

    def choose_action(self, game_state, legal_moves):
        obs = self.env._get_observation() # 1) env._get_observation will read from game_state via self.env.game
        action, _ = self.model.predict(obs, action_masks=obs["action_mask"]) # 2) ask the model to pick (it will respect obs["action_mask"])
        return self.env.action_to_move[int(action)] # 3) return the raw move object, not the integer
