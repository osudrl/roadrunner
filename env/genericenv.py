from util.colors import BLUE, WHITE, ORANGE, FAIL, ENDC
class GenericEnv(object):

    """
    Define generic environment functions that are needed for RL. Should define (not implement) all
    of the functions that sampling uses.
    """

    def __init__(self):
        self.observation_size = None
        self.action_size = None
        self.input_keys_dict = {}
        self.control_commands_dict = {}
        self.num_menu_backspace_lines = None

    def reset_simulation(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step_simulation(self, action):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError

    def get_action_mirror_indices(self):
        raise NotImplementedError

    def get_observation_mirror_indices(self):
        raise NotImplementedError

    def display_controls_menu(self,):
        """
        Method to pretty print menu of available controls.
        """
        def print_command(char, info, color=ENDC):
            char += " " * (10 - len(char))
            print(f"{color}{char}\t{info}{ENDC}")

        if ((type(self.input_keys_dict) is dict) and (len(self.input_keys_dict) > 0)):
            print("")
            print_command("Key", "Function", color=BLUE)
            for key, value in self.input_keys_dict.items():
                assert isinstance(key, str) and len(key) == 1, (
                    f"{FAIL}input_keys_dict key must be a length-1 string corresponding \
                    to the desired keybind{ENDC}")
                assert isinstance(value["description"], str), (
                    f"{FAIL}control command description must be of type string{ENDC}")
                print_command(key, value["description"], color=WHITE)
            print("")

    def display_control_commands(self,
                                 erase : bool = False):
        """
        Method to pretty print menu of current commands.
        """
        def print_command(char, info, color=ENDC):
            char += " " * (10 - len(char))
            if isinstance(info, float):
                print(f"{color}{char}\t{info:.3f}{ENDC}")
            else:
                print(f"{color}{char}\t{info}{ENDC}")
        if erase:
            print(f"\033[J", end='\r')
        elif ((type(self.input_keys_dict) is dict) and (len(self.input_keys_dict)>0)):
            print("")
            print_command("Control Input", "Commanded value",color=BLUE)
            for key, value in self.control_commands_dict.items():
                assert type(key) is str, (
                    f"{FAIL}ctrl_dict key must be of type string{ENDC}")
                print_command(key, value, color=WHITE)
            print("")
            print(f"\033[{self.num_menu_backspace_lines}A\033[K", end='\r')