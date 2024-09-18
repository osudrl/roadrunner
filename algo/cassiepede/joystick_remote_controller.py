import json
import math
import sys
import threading
import time
import tty

import numpy as np
from inputs import get_gamepad

from util.topic import Topic


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
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.DPadX = 0
        self.DPadY = 0

        self.A_pressed = False
        self.B_pressed = False
        self.X_pressed = False
        self.Y_pressed = False
        self.Back_pressed = False
        self.Start_pressed = False
        self.DPadX_pressed = False
        self.DPadY_pressed = False
        self.LeftBumper_pressed = False
        self.RightBumper_pressed = False

        self.dead_zone = 0.15

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
                    self.LeftJoystickY = self.apply_deadzone(
                        -event.state / XboxController.MAX_JOY_VAL)  # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = self.apply_deadzone(
                        event.state / XboxController.MAX_JOY_VAL)  # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = self.apply_deadzone(
                        -event.state / XboxController.MAX_JOY_VAL)  # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = self.apply_deadzone(
                        event.state / XboxController.MAX_JOY_VAL)  # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = self.apply_deadzone(
                        event.state / XboxController.MAX_TRIG_VAL)  # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL  # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.X = event.state  # previously switched with X
                elif event.code == 'BTN_WEST':
                    self.Y = event.state  # previously switched with Y
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'ABS_HAT0X':
                    self.DPadX = event.state
                elif event.code == 'ABS_HAT0Y':
                    self.DPadY = -event.state


def main():
    tty.setcbreak(sys.stdin.fileno())

    topic = Topic(fetch=False)

    joy = XboxController()

    cassies_address = [
        ("192.168.2.29", 1237),  # harvard
        ("192.168.2.251", 1237),  # osu
        ("192.168.2.252", 1237),  # playground
    ]

    topic.subscribe(my_address=('192.168.2.97', 1237))

    cmd_reset = dict(
        l_stick_x=0.0,
        l_stick_y=0.0,
        r_stick_y=0.0,
        STO=0,
        operation_mode=0,
        height_offset=0.0,
    )

    cmd = cmd_reset.copy()

    last_cmd = None

    freq = 50

    # end_time = -1

    while True:
        t = time.monotonic()

        if joy.B:
            cmd['operation_mode'] = 2
            cmd["STO"] = 1

        if joy.Y:
            cmd['operation_mode'] = 2

        cmd["l_stick_x"] = joy.LeftJoystickY

        cmd["l_stick_y"] = joy.DPadX

        cmd["r_stick_y"] = joy.RightJoystickX

        cmd["height_offset"] += joy.DPadY * 0.005

        if joy.LeftBumper == 1:
            cmd = cmd_reset.copy()

        # Clip values
        for k in ["l_stick_x", "l_stick_y", "r_stick_y", "height_offset"]:
            cmd[k] = np.clip(cmd[k], -1.0, 1.0)

        if last_cmd != cmd:
            print(cmd)

        last_cmd = cmd.copy()

        for addr in cassies_address:
            topic.publish(bytes(json.dumps(cmd), encoding='utf-8'), other_address=addr)

        if freq:
            # Run process at freq
            delay = 1 / freq - (time.monotonic() - t)
            if delay > 0:
                time.sleep(delay)


if __name__ == '__main__':
    main()
