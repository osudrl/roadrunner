import termios, sys, tty, select

class Keyboard():
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.new_term = termios.tcgetattr(self.fd)
        self.old_term = termios.tcgetattr(self.fd)

        # New terminal setting unbuffered
        self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

    def data(self):
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def get_input(self):
        key = sys.stdin.read(1)
        if key == '\b':
            return 'quit'
        elif key == '\n':
            return 'menu'
        return key

    def restore(self):
       termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def __exit__(self):
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def __del__(self):
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)
