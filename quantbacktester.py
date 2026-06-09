"""Backward-compatible entry point. Use finpulse.py instead."""

import runpy

if __name__ == "__main__":
    print("Note: quantbacktester.py has been renamed to finpulse.py\n")
    runpy.run_module("finpulse", run_name="__main__")
