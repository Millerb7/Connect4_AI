# Your code/ answer goes here.
import numpy as np
import math

class ConnectFourHueristicDepth:

    def __init__(self, mark, rows=6, columns=7, board=None):
        self.rows = rows
        self.columns = columns
        if board is None:
            self.board = np.zeros((rows, columns), dtype=int)
        else:
            self.board = np.array(board, dtype=int)
        self.current_player = mark
        self.last_move = []

    def create_random_board(self, rows=None, columns=None):
        # Create an empty board
        if rows == None: rows = self.rows
        if columns == None: columns = self.columns
        board = np.zeros((rows, columns), dtype=int)
        player = self.get_player()

        for col in range(columns):
            # Determine a random number of pieces in this column
            pieces_in_col = random.randint(0, rows)

            for row in range(pieces_in_col):
                # Add a piece to the board, ensuring it's either a 1 or -1
                board[row, col] = player
                player = -player

        # The board has been built bottom-up, but traditionally, we visualize Connect 4 boards top-down,
        # so we need to flip it vertically.
        self.board = np.flip(board, 0)

    # will return a +1 if player 1 has played and player 2 has not, otherwise a zero if player 2 has played or no playe has
    # assuming game has started when function is called
    def get_player(self):
        num_plays = np.count_nonzero(self.board)

        # If the number of plays is even, it's player 1's turn (return 1). Otherwise, it's player 2's turn (return -1).
        return 1 if num_plays % 2 == 0 else -1

    def actions(self):
        center = self.columns // 2
        # Order actions based on their distance from the center column
        return sorted([col for col in range(self.columns) if self.board[0, col] == 0],
                    key=lambda x: abs(x - center))


    def result(self, action):
        if action >= self.columns or action < 0:
            raise ValueError("Action index out of bounds")
        new_state = self.board.copy()
        for row in range(self.rows - 1, -1, -1):
            if new_state[row, action] == 0:
                new_state[row, action] = self.current_player
                break
        return new_state

    def undo(self):
        if self.last_move:
            last_col = self.last_move.pop()
            for row in range(self.rows):
                if self.board[row, last_col] != 0:
                    self.board[row, last_col] = 0
                    break
        self.current_player *= -1  # Switch back the player

    def terminal(self):
        return self.check_board() != 2

    def utility(self):
        result = self.check_board()
        if result == 1:
            return 1  # Maximizer wins
        elif result == -1:
            return -1  # Minimizer wins
        return 0  # Draw

    def evaluate_segment(self, segment, player):
        score = 0
        opponent = -player
        player_count = np.count_nonzero(segment == player)
        opponent_count = np.count_nonzero(segment == opponent)
        empty_count = np.count_nonzero(segment == 0)

        # More nuanced scoring
        if player_count == 4:
            score += 10000  # Winning condition
        elif opponent_count == 4:
            score -= 5000   # Losing condition
        elif player_count == 3 and empty_count == 1:
            score += 100    # Potential to win
        elif opponent_count == 3 and empty_count == 1:
            score -= 50     # Need to block opponent
        elif player_count == 2 and empty_count == 2:
            score += 10     # Building opportunity
        elif opponent_count == 2 and empty_count == 2:
            score -= 5      # Opponent building opportunity

        return score

    def heuristic(self):
        score = 0
        player = self.current_player

        for row in range(self.rows):
            for col in range(self.columns - 3):
                # Evaluate horizontal segment
                horiz_seg = self.board[row, col:col+4]
                score += self.evaluate_segment(horiz_seg, player)

                if row < self.rows - 3:
                    # Evaluate vertical segment
                    vert_seg = self.board[row:row+4, col]
                    score += self.evaluate_segment(vert_seg, player)

                    # Evaluate diagonal segments
                    diag1 = self.board[row:row+4, col:col+4].diagonal()
                    score += self.evaluate_segment(diag1, player)
                    diag2 = np.fliplr(self.board[row:row+4, col:col+4]).diagonal()
                    score += self.evaluate_segment(diag2, player)

        # Consider center column control as a strategy
        center_col = self.columns // 2
        center_control = np.count_nonzero(self.board[:, center_col] == player)
        score += center_control * 3  # Additional points for center control

        return score


    def check_direction(self, start_row, start_col, dr, dc, player):
        # Check if a line of four of the player's pieces exists starting from (start_row, start_col) in direction (dr, dc)
        for i in range(4):
            row = start_row + i * dr
            col = start_col + i * dc
            if not (0 <= row < self.rows and 0 <= col < self.columns):  # Check bounds
                return False
            if self.board[row, col] != player:
                return False
        return True

    def check_board(self):
        # Define the winning sequence for players
        player_markers = [1, -1]

        # Define directions to check: vertical, horizontal, diagonal down-right, diagonal up-right
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]

        for player in player_markers:
            for row in range(self.rows):
                for col in range(self.columns):
                    # Check all four directions from the current cell
                    for dr, dc in directions:
                        if self.check_direction(row, col, dr, dc, player):
                            return player  # Current player wins

        # Check for a full board without a winner (draw)
        if np.all(self.board != 0):
            return 0  # Draw

        return 2  # No winner yet

    def minimax_search(self, depth=7, alpha=-math.inf, beta=math.inf):
        # Perform a Minimax search with alpha-beta pruning
        best_score, best_move = self.max_value(depth, alpha, beta)
        return best_move, best_score

    def max_value(self, depth, alpha, beta):
        if depth == 0 or self.terminal():
            return self.heuristic(), None
        max_score, best_action = -math.inf, None
        for action in self.actions():
            new_state = self.result(action)
            self.board = new_state
            self.current_player = -self.current_player
            score, _ = self.min_value(depth - 1, alpha, beta)
            self.board = self.board  # Reset to the original state
            self.current_player = -self.current_player
            if score > max_score:
                max_score, best_action = score, action
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return max_score, best_action

    def min_value(self, depth, alpha, beta):
        if depth == 0 or self.terminal():
            return self.heuristic(), None
        min_score, best_action = math.inf, None
        for action in self.actions():
            new_state = self.result(action)
            self.board = new_state
            self.current_player = -self.current_player
            score, _ = self.max_value(depth - 1, alpha, beta)
            self.board = self.board  # Reset to the original state
            self.current_player = -self.current_player
            if score < min_score:
                min_score, best_action = score, action
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_score, best_action

    def display_board(self):
        symbol_map = {1: 'X', -1: 'O', 0: '.'}
        for row in self.board:
            print(' '.join(symbol_map[x] for x in row))
        print()

class HeuristicMinimaxAgent:
    def __init__(self, mark, depth=12):
        self.mark = mark
        self.depth = depth

    def choose_action(self, game):
        print("AI thinking...")
        # Placeholder for Minimax logic
        for col in game.actions():
            return col  # Returns the first available move

class HumanPlayer:
    def __init__(self, mark):
        self.mark = mark

    def choose_action(self, game):
        game.display_board()
        possible_moves = game.actions()
        print("Available moves:", possible_moves)
        choice = -1
        while choice not in possible_moves:
            try:
                choice = int(input("Enter your move (column number): "))
            except ValueError:
                print("Please enter a valid number.")
            if choice not in possible_moves:
                print("Invalid move. Try again.")
        return choice

def simulate_game(human, ai_agent, game_class):
    game = game_class(mark=1)  # Human starts as Player 1
    while not game.terminal():
        current_player = game.current_player
        current_agent = human if current_player == human.mark else ai_agent
        action = current_agent.choose_action(game)
        game.board = game.result(action)
        game.current_player = -current_player  # Switch players
        print(f"Player {current_player} played column {action}")

    game.display_board()
    print("Game over. adawg a winner!") 

# Setup and run the game
human_player = HumanPlayer(mark=1)
ai_agent = HeuristicMinimaxAgent(mark=-1)
simulate_game(human_player, ai_agent, ConnectFourHueristicDepth)