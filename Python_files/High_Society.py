# Building High Scoiety by Reiner Knizia

import random # To shuffle deck
from collections import Counter # To handle dupliactes in lists
import itertools # To gain all possible combinations of bids
import logging
from agents import RLagent


logging.basicConfig( # Logging is used to stop print statements from displaying, while training and evaluating models
    level=logging.ERROR,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

class HighSocietyGame:
    def __init__(self, player_names, ai_agents, debug_output=True):
        self.player_names = player_names
        self.ai_agents = ai_agents

        self.debug_output = debug_output
        logger.setLevel(logging.DEBUG if debug_output else logging.ERROR)

    def play_game(self):
        player_names = self.player_names
        ai_agents = self.ai_agents
        players = len(ai_agents)
        self.deck_of_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -5, 'double', 'double', 'double', 'half', 'drop_card']  # Deck of cards to bid for
        self.all_game_cards = self.deck_of_cards.copy() # It is used later to determine cards left in deck, without giving information about order of cards in deck
        random.shuffle(self.deck_of_cards)  # Shuffle the actual deck
        self.current_bid = 0 # The current bid amount
        self.bidding_card = None  # The card currently up for bidding
        self.players_scores = []  # List to store players scores
        self.players_money = []  # List to store sums of players money
        self.current_starting_index = 0 # Will be used to set bidding order for the auciton

        self.current_player = None # Tracks current player for RL in gym Enviroment
        self.winner = None # Tracks who won for RL in gym Enviroment
        self._terminate_episode = False  # Ends game if RL agent failed to make proper move

        self.players = [] # Initialize players with their hands and money
        for i in range(players):
            self.players.append({
                'player_name': player_names[i],
                'hand': [], # Cards one got during game
                'money': [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25],  # Starting money
                'score': 0, # Final score of a player
                'player_state' : 'bidding', # Its bidding or pass
                'cards_bidded' : [], # Cards player used to bid in given round
                'agent': ai_agents[i],  # For AI integration
            })
        # Starts the game loop
        while not self.is_game_over():
            self.round_start()




    def round_start(self):
        self.bidding_card = self.deck_of_cards.pop(0)  # Draw the first card for bidding
        if self.is_game_over() == True:
            logger.debug("Game Over")
            for player in self.players:
                player['score'] = self.calculate_scores(player)
                logger.debug(f"{player['player_name']} has score: {player['score']}")
                self.players_money.append(sum(player['money']))

            for player in self.players:
                if sum(player['money']) == min(self.players_money):
                    player['score'] -= 1000
                    logger.debug(f"{player['player_name']} spent the most money and is out of the game")

                self.players_scores.append(player['score']) 

            for player in self.players:
                if player['score'] == max(self.players_scores):
                    if self.winner != None: # Handling ties
                        if sum(player['money']) > sum(self.winner['money']): # Player with more money left wins
                            self.winner = player
                            logger.debug(f"{player['player_name']} wins tie with score: {player['score']}")
                            logger.debug(f"And with cards: {player['hand']}")
                        elif sum(player['money']) == sum(self.winner['money']): # Player with biggest card wins
                            max_card = 0
                            for card in player['hand']:
                                if (card in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) and (card > max_card):
                                    max_card = card
                            for card in self.winner['hand']:
                                if (card in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) and (card > max_card):
                                    max_card = card
                            if max_card in player['hand']:
                                self.winner = player
                                logger.debug(f"{player['player_name']} wins tie with score: {player['score']}")
                                logger.debug(f"And with cards: {player['hand']}")
                    else:
                        self.winner = player
                        logger.debug(f"{player['player_name']} wins with score: {player['score']}")
                        logger.debug(f"And with cards: {player['hand']}")
        else:
            self.current_bid = 0
            for player in self.players:
                player['player_state'] = 'bidding'
                player['cards_bidded'] = []
                player['money'].sort()

            current_order = (
                self.players[self.current_starting_index :]
                + self.players[: self.current_starting_index])

            if self.bidding_card in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'double', 'double', 'double']:
                self.good_cards_auction(current_order)
            else:
                self.bad_cards_auction(current_order)

                
    def good_cards_auction(self, current_order):
        while self.bidding_card != None:
            for player in current_order:
                self.current_player = player  # Update current player
                if player['player_state'] == 'bidding':
                    if self.check_if_won_bid(player) == True:
                        logger.debug(f"{player['player_name']} won the bid")
                        logger.debug(f"He bidded {self.current_bid}")
                        player['hand'].append(self.bidding_card)
                        player['score'] = self.calculate_scores(player)
                        self.bidding_card = None
                        self.current_starting_index = self.players.index(player)
                        return
                    
                    else:
                        logger.debug(f"Current bidding card: {self.bidding_card}")
                        logger.debug(f"Current bid: {self.current_bid}")
                        logger.debug('')
                        logger.debug(f"{player['player_name']}'s turn to bid")
                        logger.debug(f"{player['player_name']}'s hand: {player['hand']}")
                        logger.debug(f"{player['player_name']}'s money: {player['money']}")
                        logger.debug(f"{player['player_name']}'s bid: {sum(player['cards_bidded'])}")
                        logger.debug('')

                        self.player_move(player)
                else:
                    logger.debug(f"{player['player_name']} passed")

    def check_if_won_bid(self, player): # Helper function for good cards auction
        i = 0
        for enemy in self.players:
            if enemy != player:
                if enemy['player_state'] == 'bidding':
                    i += 1
        if i == 0:
            return True

    def bad_cards_auction(self, current_order):
        while self.bidding_card != None:
            for player in current_order:
                self.current_player = player  # Update current player
                logger.debug(f"Current bidding card: {self.bidding_card}")
                logger.debug(f"Current bid: {self.current_bid}")
                logger.debug('')
                logger.debug(f"{player['player_name']}'s turn to bid")
                logger.debug(f"{player['player_name']}'s hand: {player['hand']}")
                logger.debug(f"{player['player_name']}'s money: {player['money']}")
                logger.debug(f"{player['player_name']}'s bid: {sum(player['cards_bidded'])}")
                logger.debug('')

                self.player_move(player)

                if (player['player_state'] == 'pass'):
                    logger.debug(f"He gains {self.bidding_card}")
                    player['hand'].append(self.bidding_card)
                    player['score'] = self.calculate_scores(player)
                    self.current_starting_index = self.players.index(player)

                    for enemy in self.players:
                        if enemy != player:
                            logger.debug(f"Player {self.players.index(enemy)}'s paid: {sum(enemy['cards_bidded'])}")
                    return


    def get_game_state(self, player) -> dict:
        return {
        "current_bid": self.current_bid,
        "bidding_card": self.bidding_card,

        "current_player_bid": sum(player['cards_bidded']),
        "current_player_money": sum(player['money']),
        "current_player_hand": player['hand'],
        "current_player_score": self.calculate_scores(player),

        "other_players_states": [
            {'score' : p['score'], 'money' : sum(p['money'])}
            for p in self.players if p != player # for every other player different than one caliing this function
        ],

        "remaining_deck_size": len(self.deck_of_cards),
        }
    
    def get_legal_moves(self, player):
        legal_moves = []
        money = player['money']
        current_bid = self.current_bid

        for r in range(1, len(money) + 1):
            for combo in itertools.combinations(money, r):
                if sum(combo) > current_bid:
                    legal_moves.append(list(combo))

        legal_moves.append('pass')
        return legal_moves

    def player_move(self, player, action = None):
        self.current_player = player
        while True:
            action = player['agent'].choose_action(
                game_state=self.get_game_state(player),
                legal_moves=self.get_legal_moves(player),
            )
            
            if self.apply_move(player, action): # If player made proper action
                if action == 'pass':
                    logger.debug(f"{player['player_name']} passed")
                else:
                    logger.debug(f"Bid accepted! New highest bid: {self.current_bid}")
                break
            else:
                if isinstance(player['agent'], RLagent):
                    self._terminate_episode = False # Agent failed to make a move, game will end
                    action = 'pass' # Agent auto passes after making mistake
                    self.apply_move(player, action)
                    logger.debug(f"{player['player_name']} passed")
                    break
                logger.debug("Invalid bid")


    def apply_move(self, player, action):
        if action == 'pass':
            player['player_state'] = 'pass'
            player['money'].extend(player['cards_bidded'])
            return True
        
        # Convert string input to list if needed
        if not isinstance(action, list):
            try:
                bid_values = list(map(int, action.replace(' ', '').split(',')))
            except ValueError:
                return False
        else:
            bid_values = action # If player passed 1 bid

        # Validate and apply move
        if all(value in player['money'] for value in bid_values):
            if sum(player['cards_bidded']) + sum(bid_values) > self.current_bid:
                player['cards_bidded'].extend(bid_values)
                for value in bid_values:
                    player['money'].remove(value)
                self.current_bid = sum(player['cards_bidded'])
                return True
        return False

    def is_game_over(self):
        if self._terminate_episode == True:
            logger.debug("Game ends, becouse RLagent made invalid move")
            return True
        return not ('double' in self.deck_of_cards or 'half' in self.deck_of_cards) # Returns True if there are no more doubles and and halfs in the deck

    def calculate_scores(self, player):
        score = 0
        player_hand = player['hand'].copy()

        if 'drop_card' in player_hand:
            player_hand.remove('drop_card')

            numeric_cards = [card for card in player_hand 
                if isinstance(card, int) and 1 <= card <= 10]

            if numeric_cards:
                # Find the smallest numeric card
                smallest_card = min(numeric_cards)
                # Remove the first occurrence of this smallest card
                player_hand.remove(smallest_card)

        for card in player_hand:
            if card in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -5]:
                score += card

        for card in player_hand:
            if card == 'double':
                score *= 2
            if card == 'half':
                score /= 2
        
        return score

