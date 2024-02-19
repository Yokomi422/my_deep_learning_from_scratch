import os,sys
from pydantic import BaseModel,Field
import numpy as np
# OrderedDictは追加した順番を保持する辞書
from collections import OrderedDict


class TwoLyNNParams(BaseModel):
    input_size: int = Field(...,gt=1)
    hidden_size: int = Field(...,gt=1)
    output_size: int = Field(...,gt=1)
    weight_init_std: float = Field(default=0.01,gt=0)


sys.path.append(os.pardir)


class TwoLayerNet:
    def __init__(self, params: TwoLyNNParams):
        self.params = {
            "W1": params.weight_init_std * np.random.randn(params.input_size,params.hidden_size),
            "b1": params.weight_init_std * np.zeros(params.hidden_size),
            "W2": params.weight_init_std * np.random.randn(params.hidden_size,params.output_size),
            "b2": params.weight_init_std * np.zeros(params.output_size)
        }
        # レイヤーの作成
        self.layers = OrderedDict()

