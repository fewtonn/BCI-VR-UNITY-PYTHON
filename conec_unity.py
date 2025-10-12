import socket; import zmq

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except:
        return "127.0.0.1"

def send_ip_UDP(ip):
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET,socket.SO_BROADCAST,1)
    s.sendto(ip.encode(),('<broadcast>',12346)); s.close()

def configurar_socket():
    ip = get_local_ip(); print(f"IP Local: {ip}"); send_ip_UDP(ip)
    ctx=zmq.Context(); pub=ctx.socket(zmq.PUB); pub.bind("tcp://*:5555"); print("ZMQ PUB na 5555"); return pub,ctx
