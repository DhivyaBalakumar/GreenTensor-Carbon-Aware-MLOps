from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """All optimizers must implement apply() and revert()."""

    @abstractmethod
    def apply(self):
        """Apply the optimization."""

    @abstractmethod
    def revert(self):
        """Revert the optimization to its original state."""

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, *args):
        self.revert()
