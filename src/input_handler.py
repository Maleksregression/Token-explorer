import platform
import sys
import os


class InputHandler:
    """Handle keyboard input with platform support"""
    @staticmethod
    def get_key():
        if platform.system() == "Windows":
            import msvcrt
            while True:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b'\x00', b'\xe0'):
                        key2 = msvcrt.getch()
                        if key2 == b'H': return 'up'
                        elif key2 == b'P': return 'down'
                        elif key2 == b'K': return 'left'
                        elif key2 == b'M': return 'right'
                    else:
                        try:
                            return key.decode('utf-8').lower()
                        except:
                            return ''
        else:
            import tty, termios, fcntl
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    rest = ''
                    old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    try:
                        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
                        try:
                            rest = sys.stdin.read(2)
                        except:
                            rest = ''
                    finally:
                        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
                    key = ch + (rest or '')
                    if key in ('\x1b[A', '\x1bOA'): return 'up'
                    elif key in ('\x1b[B', '\x1bOB'): return 'down'
                    elif key in ('\x1b[C', '\x1bOC'): return 'right'
                    elif key in ('\x1b[D', '\x1bOD'): return 'left'
                    else:
                        return key
                else:
                    return ch.lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)