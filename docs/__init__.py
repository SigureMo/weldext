import os
import subprocess
import time
import webbrowser

docs_dir = 'docs/'
port = 8080
host = 'localhost'


def docs_dev():
    shell = os.name == "nt"
    p = subprocess.Popen(["python", "-m", "http.server",
                          str(port)], shell=shell, cwd=docs_dir)
    webbrowser.open(f'http://{host}:{port}')
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            p.terminate()
            break
