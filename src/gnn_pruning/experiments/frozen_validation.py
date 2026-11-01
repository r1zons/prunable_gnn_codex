"""Exact-model validation memoization, scoped to one frozen input and callback."""
from .flickr_dense import tensors_hash


class FrozenAccuracy:
    """Reuse a scalar only for identical architecture and exact tensor bytes.

    The input graph/labels and callback must be frozen and deterministic. This is
    per-run execution reuse, never a learned value or an aggregate-state cache.
    Every request and every real callback invocation is accounted separately.
    """
    def __init__(self, evaluate, input_identity):
        self.evaluate=evaluate
        self.input_identity=input_identity
        self.values={}
        self.requests=0
        self.forward_calls=0
        self.events=[]

    def __call__(self, model):
        model.eval()
        tensor_identity=tensors_hash(model.state_dict())
        architecture=model.export_architecture_config()
        key=(self.input_identity,type(model).__qualname__,str(architecture),tensor_identity)
        self.requests+=1
        reused=key in self.values
        if not reused:
            self.values[key]=float(self.evaluate(model))
            self.forward_calls+=1
        value=self.values[key]
        self.events.append(dict(request=self.requests,tensor_sha256=tensor_identity,
                                reused=reused,validation_accuracy=value))
        return value
