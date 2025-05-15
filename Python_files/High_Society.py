# Building High Scoiety by Reiner Knizia

import random # To shuffle deck
import itertools # To gain all possible combinations of bids
import logging


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

    def start_game(self):
        # Initialize game state, but do not play the full game loop
        player_names = self.player_names
        ai_agents = self.ai_agents
        players = len(ai_agents)
        self.deck_of_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -5, 'double', 'double', 'double', 'half', 'drop_card']
        self.all_game_cards = self.deck_of_cards.copy()
        random.shuffle(self.deck_of_cards)
        self.current_bid = 0
        self.bidding_card = None
        self.players_scores = []
        self.players_money = []
        self.current_starting_index = 0
        self.number_of_red_cards = 4
        self.current_player = None
        self.winner = None
        self._terminate_episode = False
        self.poorest_player = {'money': [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25], 'score': 0, 'agent': None}
        self.player_with_highest_score = {'money': [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25], 'score': 0, 'agent': None}
        self.highest_score_non_lowest_money = {'money': [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25], 'score': 0, 'agent': None}
        self.players = []
        for i in range(players):
            self.players.append({
                'player_name': player_names[i],
                'hand': [],
                'money': [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25],
                'score': 0,
                'player_state': 'bidding',
                'cards_bidded': [],
                'agent': ai_agents[i],
            })
        self.round_active = False

    def start_round(self):
        self.bidding_card = self.deck_of_cards.pop(0)
        if self.bidding_card in ['double', 'half']:
            self.number_of_red_cards -= 1
        # Check for game end after drawing a red card
        if self.is_game_over():
            self.finishing_game()
            return
        self.current_bid = 0
        for player in self.players:
            player['player_state'] = 'bidding'
            player['cards_bidded'] = []
            player['money'].sort()
        self.round_active = True
        self.current_order = (
            self.players[self.current_starting_index:]
            + self.players[:self.current_starting_index]
        )
        self.current_order_index = 0
        self.current_player = self.current_order[self.current_order_index]
        logger.debug(f"\n--- New Round ---")
        logger.debug(f"Card up for bidding: {self.bidding_card}")
        logger.debug(f"Players:")
        for p in self.players:
            logger.debug(f"  {p['player_name']} | Money: {p['money']} | Hand: {p['hand']}")

    def step(self, action):
        player = self.current_player
        from agents import HumanAgent
        if isinstance(player['agent'], HumanAgent) and action is None:
            logger.debug(f"\n{player['player_name']}'s turn | Card: {self.bidding_card} | Current bid: {self.current_bid}")
            logger.debug(f"Your money: {player['money']} | Your hand: {player['hand']}")
            logger.debug(f"Your current bid: {sum(player['cards_bidded'])}")
            return 'awaiting_human', player
        # After the move, print what the player did
        result = self.apply_move(player, action)
        if not result:
            if isinstance(player['agent'], HumanAgent):
                logger.debug(f"Invalid move by {player['player_name']}: {action}. Please try again.")
                return 'invalid_human', player
            else:
                player['player_state'] = 'pass'  # Remove from bidding after illegal move
                logger.debug(f"Illegal move by {player['player_name']} (type: {type(player['agent']).__name__}): {action}")
                self._terminate_episode = True
                return 'illegal', player
        # Only print bid/pass info after the move
        if result:
            if action == 'pass':
                logger.debug(f"{player['player_name']} passes. Money returned: {player['cards_bidded']}")
        # Good card logic
        if self.bidding_card in [1,2,3,4,5,6,7,8,9,10,'double','double','double']:
            bidding_players = [p for p in self.players if p['player_state'] == 'bidding']
            if len(bidding_players) == 1:
                winner = bidding_players[0]
                winner['hand'].append(self.bidding_card)
                winner['score'] = self.calculate_scores(winner)
                self.update_poorest_player_and_highest_score()
                self.bidding_card = None
                self.current_starting_index = self.players.index(winner)
                self.round_active = False
                logger.debug(f"{winner['player_name']} wins the card {winner['hand'][-1]}! New hand: {winner['hand']}, Score: {winner['score']}")
                return 'round_end', winner
        else:
            # Bad card logic
            if action == 'pass':
                player['hand'].append(self.bidding_card)
                player['score'] = self.calculate_scores(player)
                self.current_starting_index = self.players.index(player)
                self.bidding_card = None
                self.update_poorest_player_and_highest_score()
                self.round_active = False
                logger.debug(f"{player['player_name']} is forced to take the bad card {player['hand'][-1]}. New hand: {player['hand']}, Score: {player['score']}")
                return 'round_end', player
        # Advance to next player
        self.current_order_index = (self.current_order_index + 1) % len(self.current_order)
        self.current_player = self.current_order[self.current_order_index]
        return 'continue', self.current_player

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
                logger.debug(f"Could not parse bid: {action}")
                return False
        else:
            bid_values = action
        # Validate and apply move
        if all(value in player['money'] for value in bid_values):
            if sum(player['cards_bidded']) + sum(bid_values) > self.current_bid:
                player['cards_bidded'].extend(bid_values)
                for value in bid_values:
                    player['money'].remove(value)
                self.current_bid = sum(player['cards_bidded'])
                logger.debug(f"{player['player_name']} bids {bid_values} | New bid: {self.current_bid} | Money left: {player['money']}")
                return True
            else:
                logger.debug(f"Bid not high enough. Current bid: {self.current_bid}, Attempted: {sum(player['cards_bidded']) + sum(bid_values)}")
        else:
            logger.debug(f"Bid contains invalid money cards: {bid_values} not in {player['money']}")
        return False

    def get_game_state(self, player) -> dict:
        return {
        "current_bid": self.current_bid,
        "bidding_card": self.bidding_card,

        "current_player_bid": sum(player['cards_bidded']),
        "current_player_money": sum(player['money']),
        "current_player_hand": player['hand'],
        "current_player_score": player['score'],

        "highest_score": self.player_with_highest_score['score'],
        "poorest_player_money": sum(self.poorest_player['money']),
        "score_of_winner": self.highest_score_non_lowest_money['score'],

        "number_of_red_cards": self.number_of_red_cards,
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

    def update_poorest_player_and_highest_score(self):
        for player in self.players:
            if len(player['money']) == 1: # If player has only one money card
                player_money = player['money'][0] # Then using sum doesn't work
            else:
                player_money = sum(player['money'])

            if len(self.poorest_player['money']) == 1: # Same as above
                poorest_player_money = self.poorest_player['money'][0]
            else:
                poorest_player_money = sum(self.poorest_player['money'])

            if player_money < poorest_player_money:
                self.poorest_player = player
            if player['score'] > self.player_with_highest_score['score']:
                self.player_with_highest_score = player
            if (player['score'] > self.highest_score_non_lowest_money['score']) and player_money > poorest_player_money:
                self.highest_score_non_lowest_money = player

    def is_game_over(self):
        if self._terminate_episode == True:
            logger.debug("Game ends, becouse bot made invalid move")
            return True
        return self.number_of_red_cards == 0

    def finishing_game(self):
        logger.debug("\n=== GAME OVER ===")
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
    
    def is_trainee_turn(self):
        # RL agent is always player 0 during training
        return self.current_player == self.players[0]

