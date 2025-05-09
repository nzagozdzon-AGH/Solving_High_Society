from High_Society import HighSocietyGame
from agents import HumanAgent, RandomAI, RulesBasedAgent, RLagent
from Gym_Environment import HighSocietyEnv
from sb3_contrib import MaskablePPO

model = MaskablePPO.load("../model_checkpoints/high_society_trained_final.zip")
env = HighSocietyEnv()
env.reset()


ai_agent_list = [
    RLagent(model, env),
    RandomAI(),
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
    game = HighSocietyGame(player_names=get_names(), ai_agents=ai_agent_list, debug_output=False)
    winrate = 0
    for i in range (1000):
        game.play_game()  # Start the game loop
        if isinstance(game.winner['agent'], RLagent):
            winrate +=1
    print(f"Winrate is {winrate/1000}")

def play_one_round():
    game = HighSocietyGame(player_names=get_names(), ai_agents=ai_agent_list, debug_output=True)
    game.play_game()

if __name__ == "__main__":
    check_winrate()