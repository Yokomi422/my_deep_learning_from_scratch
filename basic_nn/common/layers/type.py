from abc import abstractmethod,ABC
from pydantic import BaseModel


class Layer(BaseModel,ABC):
    """
    Base Class for neural network node
    """
    @abstractmethod
    def forward(self,*args,**kwargs):
        """
        forward propagation
        """
        pass

    @abstractmethod
    def backward(self,*args,**kwargs):
        """
        backward propagation
        """
        pass
