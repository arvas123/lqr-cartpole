from matplotlib import pyplot as plt
x1=[]
y1=[]
with open('low.txt') as f:
    for line in f:
        x,y = map(float,line.split(' '))
        x1.append(x)
        y1.append(y)
x2=[]
y2=[]
with open('high.txt') as f:
    for line in f:
        x,y = map(float,line.split(' '))
        x2.append(x)
        y2.append(y)
plt.scatter(x1,y1, label='Low R')
plt.scatter(x2,y2, label='Equal R')
plt.legend(loc='best')
plt.show()
