# File that holds the GUI for the High Society game
from High_Society import HighSocietyGame
from agents import HumanAgent, RandomAI, RulesBasedAgent

import tkinter as tk
from tkinter import ttk, messagebox
import os
from PIL import Image, ImageTk

class SetupWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Game Setup")
        self.num_players = None
        self.agent_vars = []
        self.prev_agent_values = []
        self.human_assigned = False
        self.current_human_index = None  # track which slot has the human
        
        # Initial player number selection
        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=20, pady=20)
        # Displays HS logo above the player-count buttons
        logo_path = os.path.join(os.path.dirname(__file__), '..', 'Graphics', 'logo_HS.jpg')
        logo_img = Image.open(logo_path).resize((200, 200))
        self.logo_photo = ImageTk.PhotoImage(logo_img)
        tk.Label(self.frame, image=self.logo_photo).pack(pady=(0,10))
        
        tk.Label(self.frame, text="Choose number of players:").pack()
        for n in [3, 4, 5]:
            btn = tk.Button(
                self.frame, text=f"{n} players", 
                command=lambda n=n: self.create_agent_selection(n)
            )
            btn.pack(side=tk.LEFT, padx=5)

    def create_agent_selection(self, num_players):
        self.num_players = num_players
        self.frame.destroy()
        
        # Create agent selection grid
        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=20, pady=20)
        
        tk.Label(self.frame, text="Choose your opponents:").grid(row=0, columnspan=3)
        
        # Create table headers
        tk.Label(self.frame, text="Player").grid(row=1, column=0)
        tk.Label(self.frame, text="Agent Type").grid(row=1, column=1)
        
        # Create player rows
        self.agent_vars = []
        self.prev_agent_values = []  # store previous selections for swapping
        for i in range(num_players):
            row = i + 2
            var = tk.StringVar(value="HumanAgent" if i == 0 else "RandomAI")
            self.agent_vars.append(var)
            self.prev_agent_values.append(var.get())
            
            tk.Label(self.frame, text=f"Player {i+1}").grid(row=row, column=0)
            option = ttk.Combobox(
                self.frame, textvariable=var,
                values=["HumanAgent", "RandomAI", "RulesBasedAgent"],
                state="readonly"
            )
            option.grid(row=row, column=1, pady=2)
            option.bind("<<ComboboxSelected>>", lambda e, idx=i: self.on_agent_select(idx))
        
        # Confirm button and error label
        self.confirm_btn = tk.Button(
            self.frame, text="Confirm", command=self.validate_selection
        )
        self.confirm_btn.grid(row=num_players+3, columnspan=2, pady=10)
        
        self.error_label = tk.Label(self.frame, text="", fg="red")
        self.error_label.grid(row=num_players+4, columnspan=2)

    def check_human_assignments(self, *args):
        # Update human assignment flag only
        self.human_assigned = any(var.get() == "HumanAgent" for var in self.agent_vars)
        
    def on_agent_select(self, index):
        new_val = self.agent_vars[index].get()
        # Swap existing HumanAgent if selecting a new one
        if new_val == "HumanAgent":
            for j, var in enumerate(self.agent_vars):
                if j != index and var.get() == "HumanAgent":
                    prev_val = self.prev_agent_values[index]
                    self.agent_vars[j].set(prev_val)
                    self.prev_agent_values[j] = prev_val
                    break
        # Update tracking
        self.prev_agent_values[index] = new_val
        self.human_assigned = any(v.get() == "HumanAgent" for v in self.agent_vars)

    def validate_selection(self):
        # Ensure each slot is selected
        if any(var.get() not in ["HumanAgent", "RandomAI", "RulesBasedAgent"]
               for var in self.agent_vars):
            self.error_label.config(text="Please select an agent type for all players")
            return
        # Exactly one human
        if sum(var.get() == "HumanAgent" for var in self.agent_vars) != 1:
            self.error_label.config(text="Exactly one player must be HumanAgent")
            return
        # Proceed
        self.start_game()

    def start_game(self):
        mapping = {
            "HumanAgent": HumanAgent,
            "RandomAI": RandomAI,
            "RulesBasedAgent": RulesBasedAgent
        }
        ai_agents = []
        player_names = []
        for i, var in enumerate(self.agent_vars):
            agent_key = var.get()
            ai_agents.append(mapping[agent_key]())
            player_names.append(f"Player_{i+1}_{agent_key}")

        # Close setup window
        self.master.destroy()
        
        # Launch game UI
        root = tk.Tk()
        game_gui = HighSocietyGUI(
            root,
            num_players=self.num_players,
            player_names=player_names,
            ai_agents=ai_agents
        )
        root.mainloop()

class HighSocietyGUI:
    def __init__(self, master, num_players, player_names, ai_agents):
        # Initialize
        self.master = master
        self.selected_cards = []
        self.game = HighSocietyGame(player_names=player_names, ai_agents=ai_agents, debug_output=True)
        self.game.start_game()
        # Identify human
        self.human_player_index = next((i for i, a in enumerate(ai_agents) if isinstance(a, HumanAgent)), None)
        # Load graphics
        self.graphics_path = os.path.join(os.path.dirname(__file__), '..', 'Graphics')
        self.card_images = {}
        self.load_card_images()
        # Build UI and start
        self.build_game_ui()
        self.game.start_round()
        self.process_turn()

    def load_card_images(self):
        for filename in os.listdir(self.graphics_path):
            if filename.endswith('.jpg'):
                # Derive card key by removing extension and stripping '-resized' suffixes
                base = filename[:-4]  # remove .jpg
                while base.endswith('-resized'):
                    base = base[:-8]
                key = base
                path = os.path.join(self.graphics_path, filename)
                img = Image.open(path).resize((100,150))
                self.card_images[key] = ImageTk.PhotoImage(img)
        # Map generic 'double' key to one of the specific double images for correct display
        double_keys = [k for k in self.card_images if k.startswith('double')]
        if double_keys:
            self.card_images['double'] = self.card_images[double_keys[0]]

    def build_game_ui(self):
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        # Opponents
        self.opponents_frame = tk.Frame(self.main_frame)
        self.opponents_frame.pack(fill=tk.X, pady=20)
        # Center: current card and bid
        self.center_frame = tk.Frame(self.main_frame)
        self.center_frame.pack(fill=tk.BOTH, expand=True)
        self.current_card_label = tk.Label(self.center_frame, font=('Arial',14), text="Current Card")
        self.current_card_label.pack()
        # Persistent image label for bidding cards
        self.current_card_image_label = tk.Label(self.center_frame, image=None)
        self.current_card_image_label.pack()
        self.current_bid_label = tk.Label(self.center_frame, font=('Arial',12), text="Current Bid: $0")
        self.current_bid_label.pack()
        # Controls: money, bid, buttons, cards
        self.controls_frame = tk.Frame(self.main_frame)
        self.controls_frame.pack(fill=tk.X, pady=20)
        self.bid_info_frame = tk.Frame(self.controls_frame)
        self.bid_info_frame.pack()
        self.money_var, self.bid_var = tk.StringVar(), tk.StringVar()
        tk.Label(self.bid_info_frame, text="Your money:").pack(side=tk.LEFT)
        tk.Label(self.bid_info_frame, textvariable=self.money_var).pack(side=tk.LEFT, padx=10)
        tk.Label(self.bid_info_frame, text="Your bid:").pack(side=tk.LEFT)
        self.user_bid_label = tk.Label(self.bid_info_frame, textvariable=self.bid_var, fg="red")
        self.user_bid_label.pack(side=tk.LEFT)
        # Controls: buttons
        self.button_frame = tk.Frame(self.controls_frame)
        self.button_frame.pack(pady=10)
        self.pass_button = tk.Button(self.button_frame, text="Pass", command=self.pass_action)
        self.pass_button.pack(side=tk.LEFT, padx=5)
        self.commit_button = tk.Button(self.button_frame, text="Commit Bid", command=self.commit_action)
        self.commit_button.pack(side=tk.LEFT, padx=5)
        # Create frame for money-card buttons and initialize list
        self.cards_frame = tk.Frame(self.controls_frame)
        self.cards_frame.pack()
        self.card_buttons = []
        self.error_label = tk.Label(self.controls_frame, fg="red")
        self.error_label.pack()
        # Hand display
        self.hand_frame = tk.Frame(self.main_frame)
        self.hand_frame.pack(fill=tk.X, pady=10, padx=20)
        self.score_label = tk.Label(self.hand_frame, font=('Arial',11))
        self.score_label.pack(side=tk.LEFT)
        tk.Label(self.hand_frame, text="   |   Your Hand:", font=('Arial',11)).pack(side=tk.LEFT)

    def update_interface(self):
        # Clear previous opponent info and card buttons
        for w in self.opponents_frame.winfo_children(): w.destroy()
        for b in getattr(self, 'card_buttons', []): b.destroy()
        self.card_buttons = []
        # Update current bidding card and bid text
        card = self.game.bidding_card
        img = None
        if card is not None:
            key = str(card).lower()
            if key.startswith('double'): key = 'double'
            img = self.card_images.get(key)
        # Configure image label
        if img:
            self.current_card_image_label.configure(image=img)
            self.current_card_image_label.image = img
        else:
            self.current_card_image_label.configure(image='')
            self.current_card_image_label.image = None
        # Update bid text
        self.current_bid_label.configure(text=f"Current Bid: ${self.game.current_bid or 0}")
        # Player hand
        human = self.game.players[self.human_player_index]
        self.score_label.config(text=f"Hand Score: {human['score']}")
        for w in self.hand_frame.winfo_children()[1:]: w.destroy()
        for c in human['hand']:
            img = self.card_images.get(str(c).lower())
            if img: lbl=tk.Label(self.hand_frame,image=img); lbl.image=img; lbl.pack(side=tk.LEFT,padx=5)
        # Money card buttons
        for w in self.cards_frame.winfo_children(): w.destroy()
        for v in sorted(human['money']):
            btn = tk.Button(self.cards_frame, text=str(v), command=lambda vv=v: self.toggle_card(vv))
            # Highlight selected cards
            if v in self.selected_cards:
                btn.config(bg='green')
            btn.pack(side=tk.LEFT, padx=2)
            self.card_buttons.append(btn)
        # Bid info
        rem = sum(human['money']) - sum(self.selected_cards)
        bid_amt = sum(human['cards_bidded']) + sum(self.selected_cards)
        self.money_var.set(f"${rem}")
        self.bid_var.set(f"${bid_amt}")
        # Color bid label: green if above current bid, red otherwise
        current_bid = self.game.current_bid or 0
        fg_color = "green" if bid_amt > current_bid else "red"
        self.user_bid_label.config(fg=fg_color)
        # Action buttons
        for w in self.button_frame.winfo_children(): w.destroy()
        self.pass_button=tk.Button(self.button_frame,text="Pass",command=self.pass_action); self.pass_button.pack(side=tk.LEFT,padx=5)
        self.commit_button=tk.Button(self.button_frame,text="Commit Bid",command=self.commit_action); self.commit_button.pack(side=tk.LEFT,padx=5)
        # Opponents info
        for p in self.game.players:
            if p != human:
                f=tk.Frame(self.opponents_frame,bd=1,relief=tk.RIDGE,padx=10,pady=5);f.pack(side=tk.LEFT,padx=5)
                status="BIDDING" if p['player_state']=='bidding' else 'PASSED'
                bid_info = sum(p['cards_bidded']) if status=='BIDDING' else 'PASSED'
                tk.Label(f,text=f"{p['player_name']}\nMoney:${sum(p['money'])}\nScore:{p['score']}\nStatus:{status}\nBid:{bid_info}").pack()

    def toggle_card(self,value):
        if value in self.selected_cards: self.selected_cards.remove(value)
        else: self.selected_cards.append(value)
        self.update_interface()

    def process_turn(self):
        # Start a new round if needed, then check for game over with updated card
        if not self.game.round_active:
            self.game.start_round()
            self.update_interface()
            if self.game.is_game_over():
                self.show_game_over()
                return
        # Refresh UI at turn start
        self.update_interface()
        # Skip any passed players
        while self.game.round_active and self.game.current_player['player_state'] != 'bidding':
            self.game.current_order_index = (self.game.current_order_index + 1) % len(self.game.current_order)
            self.game.current_player = self.game.current_order[self.game.current_order_index]
        # After setup, decide control state
        human = self.game.players[self.human_player_index]
        if self.game.current_player is human and human['player_state']=='bidding':
            self.enable_controls()
        else:
            self.disable_controls()
            if self.game.round_active:
                self.master.after(100, self.process_ai_turn)

    def process_ai_turn(self):
        # Let game logic play AI turn, then delegate control back to process_turn
        if not self.game.round_active or self.game.bidding_card is None:
            self.process_turn()
            return
        agent = self.game.current_player['agent']
        action = agent.choose_action(self.game.get_game_state(self.game.current_player),
                                     self.game.get_legal_moves(self.game.current_player))
        self.game.step(action)
        self.process_turn()

    def pass_action(self):
        # Clear selection and perform pass, then let process_turn handle new-round logic
        self.selected_cards.clear()
        self.error_label.config(text="")
        self.game.step('pass')
        self.process_turn()

    def commit_action(self):
        # Execute bid, then delegate flow to process_turn
        human = self.game.players[self.human_player_index]
        self.error_label.config(text="")
        if not self.selected_cards:
            self.error_label.config(text="Select cards or pass")
            return
        bid = list(self.selected_cards)
        if not all(c in human['money'] for c in bid) or sum(human['cards_bidded']) + sum(bid) <= self.game.current_bid:
            self.error_label.config(text="Invalid bid")
            return
        self.selected_cards.clear()
        self.game.step(bid)
        self.process_turn()

    def disable_controls(self):
        self.pass_button.config(state=tk.DISABLED); self.commit_button.config(state=tk.DISABLED)
        for b in self.card_buttons: b.config(state=tk.DISABLED)
    def enable_controls(self):
        self.pass_button.config(state=tk.NORMAL); self.commit_button.config(state=tk.NORMAL)
        for b in self.card_buttons: b.config(state=tk.NORMAL)

    def show_game_over(self):
        # Identify player who lost by spending most money (penalized)
        loser = next((p for p in self.game.players if p['score'] < 0), None)
        loser_text = f"{loser['player_name']} spent the most money and is out of the game!" if loser else ""
        winner = self.game.winner
        info = f"{loser_text}\nWinner: {winner['player_name']}\nScore: {winner['score']}\nCards: {winner['hand']}"
        messagebox.showinfo("Game Over", info)
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    setup = SetupWindow(root)
    root.mainloop()
