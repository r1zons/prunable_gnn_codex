"""Test-first V2 adaptive API; implemented after the Red commit."""
class LocalV2Env:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

def update_q(*args, **kwargs):
    raise NotImplementedError

def select_action(*args, **kwargs):
    raise NotImplementedError

def run_greedy(*args, **kwargs):
    raise NotImplementedError

def run_q_learning(*args, **kwargs):
    raise NotImplementedError
