from High_Society import HighSocietyGame
from agents import HumanAgent, RandomAI, RulesBasedAgent, RLagent
from Gym_Environment import HighSocietyEnv
from sb3_contrib import MaskablePPO

model = MaskablePPO.load("model_checkpoints/high_society_trained_final.zip")
env = HighSocietyEnv()
env.reset()


ai_agent_list = [
    RLagent(model=model, env=env),
    # RulesBasedAgent(),
    # HumanAgent(),
    RandomAI(),
    RandomAI(),
    ]

def get_names(list = ai_agent_list):
    list_of_player_names = []
    a = b = c = d = 0
    for agent in list:
        if isinstance(agent, RulesBasedAgent):
            a += 1
            list_of_player_names.append(f'Rules_Based_Agent_{a}')

        elif isinstance(agent, RandomAI):
            b += 1
            list_of_player_names.append(f'RandomAI_{b}')

        elif isinstance(agent, HumanAgent):
            c += 1
            list_of_player_names.append(f'Human_{c}')
        elif isinstance(agent, RLagent):
            d += 1
            list_of_player_names.append(f"Smart_Agent_{d}")
    return list_of_player_names



def check_winrate():
    original_agents = ai_agent_list.copy()
    winrates = {}
    for num_players in [3, 4, 5]:
        ai_agent_list.clear()
        ai_agent_list.extend(original_agents)
        while len(ai_agent_list) < num_players:
            ai_agent_list.append(RandomAI())
        winrate = 0
        total_games = 1000
        for _ in range(total_games):
            game = HighSocietyGame(player_names=get_names(), ai_agents=ai_agent_list, debug_output=False)
            game.start_game()
            while not game.is_game_over():
                game.start_round()
                while game.round_active:
                    current_agent = game.current_player['agent']
                    game_state = game.get_game_state(game.current_player)
                    legal_moves = game.get_legal_moves(game.current_player)
                    action = current_agent.choose_action(game_state, legal_moves)
                    result, _ = game.step(action)
            if isinstance(game.winner['agent'], RLagent):
                winrate += 1
        winrates[num_players] = winrate/total_games
        print(f"Winrate in {num_players}-player game: {winrate/total_games:.3f}")
    ai_agent_list.clear()
    ai_agent_list.extend(original_agents)
    return winrates

def play_one_round():
    game = HighSocietyGame(player_names=get_names(), ai_agents=ai_agent_list, debug_output=True)
    game.start_game()
    while not game.is_game_over():
        if not game.round_active:
            game.start_round()
            if game.is_game_over() == True:
                break
        # For human agent, first show state, then prompt, then show result
        if isinstance(game.current_player['agent'], HumanAgent):
            game.step(None)
            while True:
                action = input("Enter bid values or write pass: ")
                result, player = game.step(action)
                if result == 'invalid_human':
                    continue  # Ask again
                break
        else:
            # AI agent move
            action = game.current_player['agent'].choose_action(game.get_game_state(game.current_player), game.get_legal_moves(game.current_player))
            result, player = game.step(action)
        # If result == 'round_end', the round will be restarted in the outer loop

if __name__ == "__main__":
    play_one_round()