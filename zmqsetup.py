context = zmq.Context()
socket_pub = context.socket(zmq.PUB)
socket_pub.bind("tcp://*:5555")