"""
Hierarchical clustering of data
"""

try:
    import fastcluster
    _no_fastcluster = False
except ImportError:
    _no_fastcluster = True

