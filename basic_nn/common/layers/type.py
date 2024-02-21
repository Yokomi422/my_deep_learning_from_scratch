from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class Layer(BaseModel, ABC):
    """
    Base Class for neural network node
    """

    model_config = ConfigDict(extra="allow")

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        forward propagation
        """
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        """
        backward propagation
        """
        pass
