import socket
import pickle
import struct

def send_message(message, socket):
    data = pickle.dumps(message)
    data = struct.pack('>I', len(data)) + data
    socket.sendall(data)

# def send_distance(distance ):
#     tag_set = { 'distance': distance}
#     vision = messaging.client_send('robot', tag_set, True)

# def recv_distance(socket):
#     data = socket.recv(1024)
#     return data

def recv_msg(socket):
    # Read message length and unpack it into an integer
    packet = socket.recv(4)
    length = struct.unpack('>I', packet)[0]
    # Read the message data
    data = bytearray()
    while len(data) < length:
        packet = socket.recv(length - len(data))
        if not packet:
            return None
        data.extend(packet)
    message = pickle.loads(data)
    return message

def client_send(client, message, recieve=False):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as local:
            #local.settimeout(0.01)
            local.connect(('localhost', 8090))
            send_message( (client, message), local)
            if recieve:
                return recv_msg(local)
    except:
        print("comms error")
        if recieve:
            return None