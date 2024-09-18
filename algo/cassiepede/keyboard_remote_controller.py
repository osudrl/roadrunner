import json
import time

import numpy as np
import sys
import select

import tty
import socket

from util.topic import Topic


def main():
    tty.setcbreak(sys.stdin.fileno())

    topic = Topic(fetch=False)

    # Must comment out addresses that is not in use
    cassies_address = [
        ("192.168.2.29", 1237),  # harvard
        ("192.168.2.251", 1237), # osu
        ("192.168.2.252", 1237), # playground
    ]

    topic.subscribe(my_address=('192.168.2.97', 1237))

    cmd_reset = dict(
        l_stick_x=0.0,
        l_stick_y=0.0,
        r_stick_y=0.0,
        operation_mode=0,
    )

    cmd = cmd_reset.copy()

    last_cmd = None

    while True:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            input_char = sys.stdin.read(1)
            match input_char:
                case mode if mode in "012":
                    cmd["operation_mode"] = int(mode)
                case "w":
                    cmd["l_stick_x"] += 0.05
                case "s":
                    cmd["l_stick_x"] -= 0.05
                case "q":
                    cmd["l_stick_y"] -= 0.05
                case "e":
                    cmd["l_stick_y"] += 0.05
                case "a":
                    cmd["r_stick_y"] -= 0.05
                case "d":
                    cmd["r_stick_y"] += 0.05
                case "r":
                    cmd = cmd_reset.copy()

            # Clip values
            for k in ["l_stick_x", "l_stick_y", "r_stick_y"]:
                cmd[k] = np.clip(cmd[k], -1.0, 1.0)

        if last_cmd != cmd:
            print(cmd)

        last_cmd = cmd.copy()

        for addr in cassies_address:
            topic.publish(bytes(json.dumps(cmd), encoding='utf-8'), other_address=addr)



if __name__ == '__main__':
    main()
