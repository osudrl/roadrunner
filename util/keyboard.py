from pynput import keyboard
import queue

class Keyboard():
    def __init__(self) -> None:
        # make a thread to listen to keyboard and register our callback functions
        self.listener = keyboard.Listener(on_press=self.on_press)

        # start listening
        self.listener.start()
        # queue to store keyboard commands
        self.command_queue = queue.Queue(maxsize=1)

    def on_press(self, key):
        """
        Callback to add the last keyboard input to the length-1 command queue
        """
        if hasattr(key, 'char'):
            print("\b \b", end = "\r")
            self.command_queue.put(key.char)
        if key == keyboard.Key.backspace:
            self.command_queue.put("quit")
        if key == keyboard.Key.enter:
            self.command_queue.put("menu")

    def get_input(self,):
        """
        Retrieves the input command, if any, from the queue.
        """
        if self.command_queue.empty():
            return None
        else:
            command = self.command_queue.get()
            return command