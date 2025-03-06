import logging
import os
import sys
import time
import traceback

env = os.environ.get("ENVIRONMENT", "local")

class CustomFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[38;5;246m',  # Light grey
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[31m'  # Red
    }

    def formatTime(self, record, datefmt=None):
        """
        Overriding formatTime to include milliseconds and microseconds.
        """
        ct = self.converter(record.created)
        s = time.strftime('%Y-%m-%d %H:%M:%S', ct)
        ms = int(record.msecs)
        us_part = int((record.created - int(record.created)) * 1e6)  # microseconds
        return f"{s},{ms:03d},{us_part:03d}"

    def color_text(self, text, color_code) -> str:
        return f"{color_code}{text}\033[0m"

    def format(self, record):
        formatted_log = super().format(record)

        semicolon = ":".ljust(9 - len(record.levelname))
        levelname = record.levelname
        colored_levelname = self.color_text(levelname, self.COLORS.get(record.levelname, ''))
        colored_levelname = f"{colored_levelname}{semicolon}"
        colored_time = self.color_text(self.formatTime(record), '\033[38;5;246m')  # Light grey
        colored_name = self.color_text(record.name, '\033[35m')  # Magenta
        colored_fileinfo = self.color_text(f"[{record.filename}:{record.funcName}:{record.lineno}]",
                                           '\033[38;5;246m')  # Light grey

        formatted_log = formatted_log.replace(record.levelname.ljust(8), colored_levelname)
        formatted_log = formatted_log.replace(record.asctime, colored_time)
        formatted_log = formatted_log.replace(record.name, colored_name)
        formatted_log = formatted_log.replace(f"[{record.filename}:{record.funcName}:{record.lineno}]",
                                              colored_fileinfo)

        return f"{colored_levelname} {formatted_log}"


def get_logger(name: str, level=logging.DEBUG):
    logger = logging.getLogger("hai_"+name)

    logger.setLevel(level)

    formatter = CustomFormatter('%(asctime)s - %(name)s - [%(filename)s:%(funcName)s:%(lineno)d] -> %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger

def disable_logger(name: str):
    """Disables the logger with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL)  # Suppress all log messages for this logger
    logger.handlers.clear()            # Remove all attached handlers
    logger.propagate = False

def silence_loggers():
    uvicorn_loggers = ['uvicorn', 'uvicorn.error', 'uvicorn.access']
    for logger_name in uvicorn_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)


def print_pretty_traceback(exc: Exception = None):
    """
    Prints a colorized Python traceback to stdout,
    including a stylized header and short error summary before the details.

    :param exc: Exception object to print. If None, tries sys.exc_info().
    """

    # Grab exception info from sys.exc_info() if not provided explicitly
    exc_type, exc_value, exc_tb = (
        sys.exc_info() if exc is None else (type(exc), exc, exc.__traceback__)
    )

    # If there's no active exception and none was passed in, just return.
    if exc_type is None:
        print("No exception found to print.")
        return

    # Basic ANSI color codes
    RED = "\033[31m"
    BLACK_BG = "\033[40m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

    # Print the stylized header
    header_lines = [
        f"{RED}{BLACK_BG}##############################################{RESET}",
        f"{RED}{BLACK_BG}##   Hippocraticum AI Traceback            ##{RESET}",
        f"{RED}{BLACK_BG}##############################################{RESET}",
    ]
    for line in header_lines:
        print(line)

    # Print short summary of error (type and message) in red
    error_summary = f"{RED}ERROR TYPE: {exc_type.__name__}\nERROR MSG : {str(exc_value)}{RESET}"
    print(error_summary)
    print()  # blank line

    # Build the standard traceback text
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    tb_str = "".join(tb_lines)

    # Split into lines so we can colorize selectively
    lines = tb_str.splitlines()
    colorized_lines = []

    for line in lines:
        # Highlight lines containing file/line info in blue
        if 'File "' in line and 'line' in line:
            colorized_lines.append(f"{BLUE}{line}{RESET}")
        # Highlight caret lines in yellow
        elif line.strip().startswith("^"):
            colorized_lines.append(f"{YELLOW}{line}{RESET}")
        # Highlight the standard "Traceback (most recent call last)" line in yellow
        elif line.startswith("Traceback (most recent call last)"):
            colorized_lines.append(f"{YELLOW}{line}{RESET}")
        # For everything else, use red
        else:
            colorized_lines.append(f"{RED}{line}{RESET}")

    # Print a short explanatory heading for the detailed traceback
    print(f"{RED}-------  FULL TRACEBACK  -------{RESET}")

    # Print each colorized line
    for colorized_line in colorized_lines:
        print(colorized_line)

# Mute predefined loggers
disable_logger("speechbrain.utils.quirks")
disable_logger("httpx")
disable_logger("faster_whisper")