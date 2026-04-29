import socket
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)  # 4MB
sock.bind(('0.0.0.0', 12345))
sock.setblocking(True)  # 不用timeout，直接阻塞等

count = 0
start = time.time()
while time.time() - start < 10:
    data = sock.recv(256)
    count += 1

print(f"10 seconds: received {count} packets ({count/10:.0f} Hz)")
sock.close()