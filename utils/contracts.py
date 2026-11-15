"""Shared input and graph contracts."""

class ContractError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise ContractError(message)
