from sys import argv

def f(x,y):
    return not (x and y)

print(f(*map(int,argv[1:])))
