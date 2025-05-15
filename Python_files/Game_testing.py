from High_Society import HighSocietyGame
from agents import HumanAgent, RandomAI, RulesBasedAgent, RLagent
from Gym_Environment import HighSocietyEnv
from sb3_contrib import MaskablePPO

model = MaskablePPO.load("model_checkpoints/high_society_trained_final.zip")
env = HighSocietyEnv()
env.reset()


ai_agent_list = [
    # RulesBasedAgent(),
    RLagent(model, env),
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
        # Reset ai_agent_list to original state
        ai_agent_list.clear()
        ai_agent_list.extend(original_agents)
        
        # Fill remaining slots with RandomAI
        while len(ai_agent_list) < num_players:
            ai_agent_list.append(RandomAI())
            
        game = HighSocietyGame(player_names=get_names(), ai_agents=ai_agent_list, debug_output=False)
        winrate = 0
        total_games = 1000
        
        for _ in range(total_games):
            game.play_game()
            if isinstance(game.winner['agent'], RLagent):
                winrate += 1
                
        winrates[num_players] = winrate/total_games
        print(f"Winrate in {num_players}-player game: {winrate/total_games:.3f}")
    
    # Restore original ai_agent_list
    ai_agent_list.clear()
    ai_agent_list.extend(original_agents)
    
    return winrates

def play_one_round():
    game = HighSocietyGame(player_names=get_names(), ai_agents=ai_agent_list, debug_output=True)
    game.play_game()

if __name__ == "__main__":
    play_one_round()