"""Discrete-event simulator for the frontier training program.

Everything in this package is pure-Python. No torch, no GPUs.
We model:
  • virtual time (days/hours/seconds)
  • cluster of GPUs with realistic failure rates
  • Chinchilla-style scaling laws for loss prediction
  • MFU → throughput → wall time
  • eval-score curves as a function of compute
  • rolling $ accounting
"""
