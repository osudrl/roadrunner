from .interface_ctypes import *
from multiprocessing import Process

class DigitUdp:
    def __init__(self, robot_address='127.0.0.1',
                 remote_addr='127.0.0.1', remote_port='35000',
                 local_addr='0.0.0.0', local_port='35001'):
        # Initialize UDP connection to LLAPI
        self.sock = udp_init_client(str.encode(remote_addr),
                                    str.encode(remote_port),
                                    str.encode(local_addr),
                                    str.encode(local_port))
        self.packet_header_info = packet_header_info_t()
        self.recvlen_pd = 2 + 872
        self.sendlen_pd = 2 + 640
        self.recvbuf = (ctypes.c_ubyte * self.recvlen_pd)()
        self.sendbuf = (ctypes.c_ubyte * self.sendlen_pd)()
        self.inbuf = ctypes.cast(ctypes.byref(self.recvbuf, 2), ctypes.POINTER(ctypes.c_ubyte))
        self.outbuf = ctypes.cast(ctypes.byref(self.sendbuf, 2), ctypes.POINTER(ctypes.c_ubyte))

        # Launch UDP to LLAPI connection
        p = Process(target=self._run, args=(robot_address,))
        p.start()

    @staticmethod
    def _run(robot_address):
        llapi_run_udp(robot_address)

    def send_pd(self, u):
        pack_command_pd(self.outbuf, ctypes.byref(u))
        send_packet(self.sock, self.sendbuf, self.sendlen_pd, None, 0)

    def recv_newest_pd(self):
        nbytes = get_newest_packet(self.sock, self.recvbuf, self.recvlen_pd, None, None)
        if nbytes != self.recvlen_pd:
            return None
        process_packet_header(self.packet_header_info, self.recvbuf, self.sendbuf)
        state_out = llapi_observation_t()
        unpack_observation(self.inbuf, state_out)
        return state_out

    def __del__(self):
        udp_close(self.sock)
        llapi_free()
