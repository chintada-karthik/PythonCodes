import random

def get_computer_move():
    moves = ['R', 'P', 'S']
    return random.choice(moves)

def determine_winner(player_move, computer_move):
    if player_move == computer_move:
        return "It's a tie!"
    elif (player_move == 'R' and computer_move == 'S') or \
         (player_move == 'P' and computer_move == 'R') or \
         (player_move == 'S' and computer_move == 'P'):
        return "You win!"
    else:
        return "Computer wins!"

def main():
    print("Welcome to Rock, Paper, Scissors!")
    while True:
        player_move = input("Enter your move (R for Rock, P for Paper, S for Scissors, Q to quit): ").upper()
        if player_move == 'Q':
            print("Thanks for playing!")
            break
        if player_move not in ['R', 'P', 'S']:
            print("Invalid input. Please enter R, P, or S.")
            continue
        computer_move = get_computer_move()
        print(f"Computer chose: {computer_move}")
        result = determine_winner(player_move, computer_move)
        print(result)

if __name__ == "__main__":
    main()