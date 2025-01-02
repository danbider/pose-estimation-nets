from typing import List
import argparse
import sys
import shutil


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(
            formatter_class=_SmartFormatter,
            epilog="documentation: https://lightning-pose.readthedocs.io/en/latest/source/user_guide/index.html",
            **kwargs,
        )
        self.is_sub_parser = False

    def print_help(self, with_welcome=True, **kwargs):
        if with_welcome and not self.is_sub_parser:
            print("Welcome to the lightning-pose CLI!\n")
        super().print_help(**kwargs)

    def error(self, message):
        red = "\033[91m"
        end = "\033[0m"
        sys.stderr.write(red + f"error:\n{message}\n\n" + end)

        width = shutil.get_terminal_size().columns
        sys.stderr.write("-" * width + "\n")
        self.print_help(with_welcome=False)
        sys.exit(2)


class ArgumentSubParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_sub_parser = True


# Source: https://gist.github.com/panzi/b4a51b3968f67b9ff4c99459fb9c5b3d
class _SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text: str, width: int) -> List[str]:
        lines: List[str] = []
        for line_str in text.split("\n"):
            line: List[str] = []
            line_len = 0
            for word in line_str.split(" "):
                word_len = len(word)
                next_len = line_len + word_len
                if line:
                    next_len += 1
                if next_len > width:
                    lines.append(" ".join(line))
                    line.clear()
                    line_len = 0
                elif line:
                    line_len += 1

                line.append(word)
                line_len += word_len

            lines.append(" ".join(line))
        return lines

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        return "\n".join(
            indent + line for line in self._split_lines(text, width - len(indent))
        )
