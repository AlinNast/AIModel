import math


class Value:
    """ stores a single scalar value and its gradient """
    # Scalar its a one dimension/value array/tensor
    
    def __init__(self, data, _children=(), _op=''):
        
        self.data = data                # This stores the actual value of the obj
        self.prev = set(_children)      # This stores the objects that took part in creating this object
        self._op = _op                  # This stores the operation that was used to craete this object
        self.grad = 0                   # This is the stored grad
        self._backward = lambda: None
    
    def __add__(self,other):
        """ Direct addition between objects of type value is not
        supported by default so it has to be defined.
        It returns a object of type Value with the added data between the two opernads"""
        
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        ### on this operation, the construction of the new Value object is made
        ### while also passing in the tuple of the initial Value objects
        ### this way the resulting objects know the roots of the objets that
        ### created it and also the operation that created it
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        ### after building the object and storing his derivatives, the funtion that
        ### calculates their gradients is defined and stored in the object
        
        return out
    
    def __mul__(self,other):
        """ It returns a object of type Value with the data
        field containing the multiplication of the operands"""
        
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        ### on this operation, the construction of the new Value object is made
        ### while also passing in the tuple of the initial Value objects
        ### this way the resulting objects know the roots of the objets that
        ### created it and also the operation that created it
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad 
        out._backward = _backward
        ### after building the object and storing his derivatives, the funtion that
        ### calculates their gradients is defined and stored in the object
        
        return out
    
    def __pow__(self, other):
        """It returns a Object of type Value with the data field containing the 
        Power of the operands
        """
        
        assert isinstance(other, (int, float))
        out = Value(self ** other, (self,), 'pow')
        
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        
        return out
    
    def __tanh__(self):
        x = self.data
        t += (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), "tanh")
        
        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward
        
    def __radd__(self, other):
        return self + other
    
    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
        
    
    def backward(self):
        """ This stores all the values that are in the graph related to this Value object
        in a array, marks them as visited or not (in order to not duplicate an entry),
        goes to the parent nodes and redoes the calculation, and stores them all in a array"""

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
        
    def __repr__(self):
        """It transforms the criptic output into a readable print"""
        
        return f"Value(data={self.data}, grad={self.grad})"