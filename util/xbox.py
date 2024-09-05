
import math
import numpy as np
import threading
import time

from inputs import get_gamepad, UnpluggedError
from util.colors import FAIL, ENDC

class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 10)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.Back = 0
        self.Start = 0
        self.DPadX = 0
        self.DPadY = 0

        # Add tracker class vars for all buttons
        buttons = ["A", "B", "X", "Y", "Back", "Start", "DPadX", "DPadY"]
        for button in buttons:
            setattr(self, f"{button}_pressed", False)

        self.dead_zone = 0.15    # Deadzone for joysticks

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def apply_deadzone(self, value):
        return 0 if abs(value) < self.dead_zone else value

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = self.apply_deadzone(-event.state / XboxController.MAX_JOY_VAL) # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = self.apply_deadzone(event.state / XboxController.MAX_JOY_VAL) # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = self.apply_deadzone(-event.state / XboxController.MAX_JOY_VAL) # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = self.apply_deadzone(event.state / XboxController.MAX_JOY_VAL) # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.prev_A = self.A
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.X = event.state #previously switched with X
                elif event.code == 'BTN_WEST':
                    self.Y = event.state #previously switched with Y
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'ABS_HAT0X':
                    self.DPadX = event.state
                elif event.code == 'ABS_HAT0Y':
                    self.DPadY = -event.state
            time.sleep(0.001) # Rate limit how fast the thread runs

def check_xbox_connection():
    try:
        get_gamepad()
        return True
    except (IndexError, UnpluggedError) as e:
        print(f"{FAIL}{e}{ENDC}")
        return False
    except Exception as e:
        print(f"{FAIL}Unknown xbox connection error: {e}{ENDC}")
        return False

