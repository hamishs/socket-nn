import io
import socket
import numpy as np


def send_array(s, A):
    """Send a numpy array A from socket s."""
    with io.BytesIO() as f:
        np.save(f, A)
        f.seek(0)
        s.sendall(f.read())


def recv_array(s):
    """Receive a numpy array from socket s."""
    with io.BytesIO() as f:
        while True:
            z = s.recv(1024)
            if not z:
                break
            f.write(z)
        f.seek(0)
        x = np.load(f)
    return x


def exchange(addr, A):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(addr)
        send_array(s, A)
        result = recv_array(s)
    return result


def main():
    """Test the server."""
    address = ("127.0.0.1", 8080)

    x = np.eye(2)
    y = exchange(address, x)
    assert np.allclose(y - x, np.ones((2, 2)))

    z = 1 - np.eye(2)
    w = exchange(address, z)
    assert np.allclose(w - z, np.ones((2, 2)))

    z = np.random.randn(2, 2)
    w = exchange(address, z)
    assert np.allclose(w - z, np.ones((2, 2)))

if __name__ == "__main__":
    main()