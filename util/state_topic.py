from multiprocessing import Manager, Process
import time

class StateTopic:
    def __init__(self, socket):
        # Initialize process to fetch state and send command with LLAPI
        self.socket = socket
        self.state = Manager().dict()
        remote_func = Process(target=self._fetch, args=(self.state, self.socket))
        remote_func.start()

    @staticmethod
    def _fetch(data, socket):
        while True:
            state = socket.recv_newest_pd()
            while state is None:
                state = socket.recv_newest_pd()
            data['state'] = state
            time.sleep(1/1e4)

    def recv(self):
        out = self.state.get('state', None)
        return out

    def __del__(self):
        if self.socket is not None:
            self.socket.__del__()
