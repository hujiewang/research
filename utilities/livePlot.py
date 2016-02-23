__author__ = 'hujie'
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class LivePlot:

    def __init__(self,fname):
        self.fname=fname

        fig = plt.figure()
        self.subplot = fig.add_subplot(1,1,1)
        ani = animation.FuncAnimation(fig, self.animate, interval=1000)
        plt.show()



    def animate(self,i):
        x_data=[]
        y_data=[]
        cnt=0
        with open(self.fname,'r') as f:
            for line in f:
                y=float(line)
                x_data.append(cnt)
                y_data.append(y)
                cnt+=1

        self.subplot.clear()
        self.subplot.plot(x_data,y_data)

lp=LivePlot('/home/hehe/projects/research/exp/h_rnn/valid_acc.txt')
lp=LivePlot('/home/hehe/projects/research/exp/h_rnn/train_loss.txt')
