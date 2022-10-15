import socket
import threading
import socketserver

RECV_BUFFER = 1024


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """
    Multi-threaded handle TCP requests
    """
    def handle(self):
        self.request.settimeout(3)
        data = bytes(self.request.recv(RECV_BUFFER), 'ascii')
        cur_thread = threading.current_thread()

        response = bytes("{}: {}".format(cur_thread.name, data), 'ascii')
        self.request.sendall(response)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass
