"""Minimal offline-only subset of gin-config used by BinocMesher decorators."""
def configurable(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    def decorate(obj):
        return obj
    return decorate
