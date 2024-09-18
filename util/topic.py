import socket
from multiprocessing import Manager, Process
import time
import sys
import numpy as np


class Topic:
    def __init__(self, freq=None, timeout=5, fetch=True):
        self.freq = freq
        self.timeout = timeout
        self.fetch = fetch

    @staticmethod
    def _fetch(data, soc, freq, timeout, packet_size=4096):
        last_received = time.monotonic()
        while True:
            t = time.monotonic()
            try:
                inc = soc.recvfrom(packet_size)
            except socket.timeout:
                inc = None

            # print('inc', inc)

            if inc is not None:
                msg, addr = inc
                # Assume getting ndarray in bytes
                data['msg'] = msg
                last_received = time.monotonic()
            elif timeout is not None and time.monotonic() - last_received > timeout:
                    if 'msg' in data:
                        del data['msg']
                    # print(f'Warning: No data received for {timeout} second. Buffer cleared')

            if freq:
                # Run process at freq
                delay = 1 / freq - (time.monotonic() - t)
                if delay > 0:
                    time.sleep(delay)

    def subscribe(self, my_address):
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
        # self.soc.setblocking(False)
        if self.timeout:
            self.soc.settimeout(self.timeout)

        self.soc.bind(my_address)

        self.inc_data = Manager().dict()

        if self.fetch:
            remote_func = Process(target=self._fetch, args=(self.inc_data, self.soc, self.freq, self.timeout))
            remote_func.start()

    def publish(self, data, other_address):
        # publish any data to any address
        self.soc.sendto(data, other_address)

    def get_data(self):
        assert self.fetch, 'Data fetching is disabled'
        # unpack data into either ndarray in bytes or None
        data = self.inc_data.get('msg', None)
        return data

    def __del__(self):
        if self.soc is not None:
            self.soc.close()
