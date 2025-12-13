import os
import subprocess
import time
import hashlib
from io import BytesIO
from PIL import Image
from manga_ocr import MangaOcr
import signal
import sys


from transformers import logging 
logging.set_verbosity_error()


_stop = False

def _handle_sigint(sig, frame):
    global _stop
    _stop = True

signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


def env():
    return {
        "session": os.environ.get("XDG_SESSION_TYPE", "").lower(),
        "desktop": os.environ.get("XDG_CURRENT_DESKTOP", "").lower(),
    }


def has(cmd):
    return subprocess.call(
        ["which", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    ) == 0


def require(cmd, msg):
    if not has(cmd):
        raise RuntimeError(msg)


# ------------------------
# WAYLAND - GNOME (poll)
# ------------------------

def wayland_poll(mocr):
    print("Wayland GNOME → polling mode")

    last_hash = None

    while not _stop:
        types = subprocess.run(
            ["wl-paste", "-l"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        ).stdout

        if "image/png" not in types:
            time.sleep(0.5)
            continue

        data = subprocess.run(
            ["wl-paste", "--type", "image/png"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        ).stdout

        if not data:
            time.sleep(0.5)
            continue

        h = hashlib.md5(data).hexdigest()
        if h == last_hash:
            time.sleep(1.5)
            continue

        last_hash = h
        img = Image.open(BytesIO(data))
        text = mocr(img)
        print(text)

        subprocess.run(["wl-copy"], input=text.encode(), check=False)

    print("Exiting…")


# ------------------------
# WAYLAND - EVENT DRIVEN
# ------------------------

def wayland_watch(mocr):
    print("Wayland compositor → event-driven")

    proc = subprocess.Popen(
        ["wl-paste", "--watch", "--type", "image/png", "cat"],
        stdout=subprocess.PIPE
    )

    last_hash = None

    try:
        while not _stop:
            data = proc.stdout.readline()
            if not data:
                continue

            h = hashlib.md5(data).hexdigest()
            if h == last_hash:
                continue

            last_hash = h
            img = Image.open(BytesIO(data))
            text = mocr(img)
            print(text)

            subprocess.run(["wl-copy"], input=text.encode(), check=False)

    finally:
        proc.terminate()
        proc.wait()
        print("Exiting…")


# ------------------------
# X11
# ------------------------

def x11_watch(mocr):
    print("X11 → event-driven")

    if has("xclip"):
        get = ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]
        set_ = ["xclip", "-selection", "clipboard"]
    elif has("xsel"):
        get = ["xsel", "--clipboard", "--output"]
        set_ = ["xsel", "--clipboard", "--input"]
    else:
        raise RuntimeError("xclip or xsel required on X11")

    last_hash = None

    while not _stop:
        try:
            data = subprocess.check_output(get, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            time.sleep(0.5)
            continue

        if not data:
            time.sleep(0.5)
            continue

        h = hashlib.md5(data).hexdigest()
        if h == last_hash:
            time.sleep(1.5)
            continue

        last_hash = h
        img = Image.open(BytesIO(data))
        text = mocr(img)
        print(text)

        subprocess.run(set_, input=text.encode(), check=False)

    print("Exiting…")


# ------------------------
# MAIN
# ------------------------

def main():
    try:
        e = env()

        if e["session"] == "wayland":
            require("wl-paste", "wl-clipboard not available")
            require("wl-copy", "wl-clipboard not available")
        elif e["session"] == "x11":
            if not (has("xclip") or has("xsel")):
                raise RuntimeError("xclip or xsel required on X11")

        mocr = MangaOcr()
        print("CTRL+C to exit")

        if e["session"] == "wayland":
            if "gnome" in e["desktop"]:
                wayland_poll(mocr)
            else:
                wayland_watch(mocr)
        elif e["session"] == "x11":
            x11_watch(mocr)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

