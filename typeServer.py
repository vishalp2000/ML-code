import messaging
#from messaging import send_distance
import socket
import json
from time import time as now
from copy import deepcopy

# Avoid a linux-based 'BrokenPipe' error by ingoring the SIGPIPE signal
from signal import signal, SIGPIPE, SIG_DFL

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("Start")
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 5966))
        print("Bind")
        s.listen()
        print("Listen")
        conn,addr = s.accept()
        print("Accepted")
        t = now()
        with conn:
            while True:
                # signal(SIGPIPE, SIG_DFL)
                # data = conn.recv(1500)
                print("Recieved")
                # data = data.decode()
                print("Decoded")
                tag_set = {}
                print("Tag Set")
                middleman = messaging.client_send('vision', tag_set, True)
                data = middleman['vision_tags']
                print("Client Send")
                # distance = middleman['node']['type']
                print("Client Recieve:")
                # print(distance)
                # ----
                # Convert this dictionary into a serialized JSON object somehow and then it should work
                s.send(json.dumps(middleman['vision_tags']).encode())
                print("Sent!")
                # data.encode()
                # s.sendall(data)
                # ----
                # if not data:
                    # break
            print("Theoretically unreachable code")
            s.close()

if __name__ == '__main__':
    main()