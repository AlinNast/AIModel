from nn import MLP

# sample data
x = [2.0, 4.0, -5.0] 

#
y = 1.0



def main():
    print("\nProgram Initialized\n")
    
    model = MLP(3 ,[4,4,1])
    print (model)
    
    print("\n Now to call with test data\n")
    ypred = model(x)
    print(ypred)

if __name__ == "__main__":
    main()