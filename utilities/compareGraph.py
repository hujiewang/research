import matplotlib.pyplot as plt

def getData(fname):
    x_data=[]
    y_data=[]
    cnt=0
    with open(fname,'r') as f:
        for line in f:
            y=float(line)
            x_data.append(cnt)
            y_data.append(y)
            cnt+=1
    return x_data, y_data

x_0,y_0 = getData('/root/source/research/exp/semeval2016_task3/train_loss.txt')
x_1,y_1 = getData('/root/source/research/exp/semeval2016_task3/valid_loss.txt')
plt.plot(x_0, y_0, 'r-', x_1, y_1, 'b-',)
plt.show()