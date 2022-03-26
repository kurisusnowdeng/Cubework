import argparse


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    return parser.parse_args()
