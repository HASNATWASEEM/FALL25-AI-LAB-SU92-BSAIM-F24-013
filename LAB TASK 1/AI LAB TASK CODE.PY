class TicTacToe:
    def __init__(self, size):
        self.size = size
        self.board = [" "] * (size * size)
        self.current_player = "X"

    def display_board(self):
        print("\n")
        for i in range(0, len(self.board), self.size):
            row = " | ".join(self.board[i:i + self.size])
            print(row)
            if i < len(self.board) - self.size:
                print("--+" * (self.size - 1) + "--")
        print("\n")

    def make_move(self, position):
        if self.board[position] == " ":
            self.board[position] = self.current_player
            return True
        else:
            print("Nope, that square’s already snagged. Pick another one.")
            return False

    def switch_player(self):
        self.current_player = "O" if self.current_player == "X" else "X"

    def check_winner(self):
        b = self.board
        s = self.size

        # Check rows
        for r in range(0, len(b), s):
            first = b[r]
            if first != " " and all(first == b[r + c] for c in range(s)):
                return True

        # Check columns
        for c in range(s):
            first = b[c]
            if first != " " and all(first == b[c + r * s] for r in range(s)):
                return True

        # Main diagonal
        first = b[0]
        if first != " " and all(first == b[i * (s + 1)] for i in range(s)):
            return True

        # Anti-diagonal
        first = b[s - 1]
        if first != " " and all(first == b[(i + 1) * (s - 1)] for i in range(s)):
            return True

        return False

    def is_draw(self):
        return " " not in self.board

    def play(self):
        print("Yo, welcome to Tic-Tac-Toe!")
        self.display_board()

        while True:
            try:
                raw = input(f"Player {self.current_player}, where you wanna go? (1-{len(self.board)}): ")
                position = int(raw) - 1
                if position < 0 or position >= len(self.board):
                    print(f"Whoa there, {raw} is not on the board. Pick between 1 and {len(self.board)}.")
                    continue
            except ValueError:
                print("Uh, that's not a number. Try again.")
                continue

            if self.make_move(position):
                self.display_board()

                if self.check_winner():
                    print(f"And that’s game! Player {self.current_player} wins! ")
                    break
                elif self.is_draw():
                    print("Looks like a draw. Nobody wins. Womp-womp.")
                    break
                else:
                    self.switch_player()
try:
    size_input = input("How big ya want the board? (Just type a number, like 3 for 3x3): ")
    size = int(size_input)
    if size < 3:
        print("Listen, 3 is as small as we go. Setting it to 3.")
        size = 3
except ValueError:
    print("Dude. That's not even a number. Going with 3x3.")
    size = 3

game = TicTacToe(size)
game.play()