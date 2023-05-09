import random
import math

class Value:
    def __init__(self,data= 0.0, label='',operator='',came_from=()):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.label = label
        self.operator = operator
        self.came_from = set(came_from)
    
    def __add__(self,other):
        if isinstance(other,(int,float)):
            other = Value(other)
        ret = Value(self.data + other.data)
        ret.came_from = (self,other)
        ret.operator = '+'
        def backward():
            self.grad += ret.grad
            other.grad += ret.grad
        ret._backward = backward
        return ret
    
    
    def __mul__(self,other):
        if isinstance(other,(int,float)):
            other = Value(other)
        ret = Value(self.data * other.data)
        ret.came_from = (self,other)
        ret.operator = '*'
        def backward():
            self.grad += other.data * ret.grad
            other.grad += self.data * ret.grad
        ret._backward = backward
        return ret
    
    def __pow__(self, other):
        if not isinstance(other,(float,int)):
            raise Exception('only float and int are supported')
        ret = Value(self.data ** other)
        ret.came_from = (self,)
        ret.operator = '**'
        def backward():
            self.grad += (other  * self.data**(other -1)) * ret.grad
        ret._backward = backward
        return ret
    
    def __sub__(self,other):
        return self + (-other)
    
    def __rsub__(self, other): 
        return other + (-self)
    
    def __radd__(self,other):
        return self + other
    def __rmul__(self,other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __truediv__(self, other): 
        return self * other**-1

    def __rtruediv__(self, other): 
        return other * self**-1

    def __repr__(self) -> str:
        return f'{self.label}|{self.data}|{self.grad}|{self.operator}'
    
    def __div__(self,other):
        return self * (other ** -1)
    
    def tanh(self):
        e = math.exp(self.data*2)
        th = (e-1)/(e+1)
        ret = Value(th,label = 'tanh',came_from=(self,))
        def backward():
            self.grad += (1 - th**2) * ret.grad
        ret._backward = backward
        return ret
    
    def backward(self):
        visited = set()
        topo = []
        def toposort(root):
            if root not in visited:
                visited.add(root)
                for child in root.came_from:
                    toposort(child)
                topo.append(root)
        toposort(self)
        self.grad = 1.0
        for node in reversed(topo):
            #print(f'calling backward of {node}')
            node._backward()

class Neuron:
    def __init__(self, numInputs = 0):
        #random.seed(1)
        self.numInputs = numInputs
        self.weights =[Value(random.uniform(-1,1)) for _ in range(numInputs)]
        self.bias = Value(random.uniform(-1,1))

    def __call__(self, x):
        sum = Value(0)
        for wi,xi in zip(self.weights, x):
            sum+=wi*xi
        return (sum+self.bias).tanh()
    
    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self,numInputs=0,numOutputs=0):
        self.neurons = [Neuron(numInputs) for _ in range(numOutputs)]
    
    def __call__(self,x):
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        p = []
        for neuron in self.neurons:
            p.extend(neuron.parameters())
        return p
    
class MLP:
    def __init__(self,numInputs=0, numOutputs=[]):
        sz = [numInputs] + numOutputs
        self.layers = []
        for i in range(len(numOutputs)):
            self.layers.append(Layer(sz[i],sz[i+1]))
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x)==1 else p   
    def parameters(self):
        p = []
        for layer in self.layers:
            p.extend(layer.parameters())
        return p

def main():
    x  = Value(10,label = 'x')
    y = Value(20, label = 'y')
    z = 2*x+ 3 *y ; z.label ='z'
    z.backward()
    print(x,y,z)
    
    xs = [[2.0,3.0,-1.0],
        [3.0,-1.0,0.5],
        [0.5,1.0,1.0],
        [1.0,1.0,-1.0]];

    ys = [1.0, -1.0, -1.0, 1.0]

    mlp = MLP(3,[4,4,1])
    print(f'parameter count: {len(mlp.parameters())}')
    yPred = []
    for epoch in range(20):
        yPred = [mlp(x) for x in xs]
        loss = sum([(yP-y)**2 for yP,y in zip(yPred,ys)])

        #zero grad
        for p in mlp.parameters():
            p.grad = 0.0

        #backward
        loss.backward()

        #update params
        for p in mlp.parameters():
            p.data += -0.1 * p.grad

        print(f'epoch:{epoch}: loss:{loss.data}')


if __name__ == '__main__':
    main()