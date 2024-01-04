

class Value:
    """ stores a single scalar value and its gradient """
    
    def __init__(self, data, _children=(), _op=''):
        
        self.data = data                # This stores the actual value of the obj
        self.prev = set(_children)      # This stores the objects that took part in creating this object
        self._op = _op                  # This stores the operation that was used to craete this object
        self.grad = 0
    
    def __add__(self,other):
        """ Direct addition between objects of type value is not
        supported by default so it has to be defined.
        It returns a object of type Value with the added data between the two opernads"""
        
        out = Value(self.data + other.data, (self, other), '+')
        ### on this operation, the construction of the new Value object is made
        ### while also passing in the tuple of the initial Value objects
        ### this way the resulting objects know the roots of the objets that
        ### created it and also the operation that created it
        return out
    
    def __mul__(self,other):
        """ It returns a object of type Value with the data
        field containing the multiplication of the operands"""
        
        out = Value(self.data * other.data, (self, other), '*')
        ### on this operation, the construction of the new Value object is made
        ### while also passing in the tuple of the initial Value objects
        ### this way the resulting objects know the roots of the objets that
        ### created it and also the operation that created it
        return out
        
    def __repr__(self):
        """It transforms the criptic output into a readable print"""
        return f"Value(data={self.data}, grad={self.grad})"