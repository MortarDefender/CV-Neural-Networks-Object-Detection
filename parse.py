import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_row(row):
    cls_loss = float(row[-7:-1])
    reg_loss = float(row[-44: -38])
    loss = float(row[-76:-70])
    return loss, reg_loss, cls_loss


with open('models/dogs_exp1/loss_lines.txt', 'r') as f:
    iterations, cls, reg, loss = [], [], [], []
    curr_iter = 0
    for line in f.readlines():
        try:
            res = parse_row(line)
            loss.append(res[0])
            reg.append(res[1])
            cls.append(res[2])
            iterations.append(curr_iter)
            curr_iter += 100
        except:
            print(line)


plt.plot(iterations, loss)
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.savefig('loss.png')
plt.clf()

plt.plot(iterations, cls)
plt.ylabel('Classification Loss')
plt.xlabel('Iterations')
plt.savefig('cls_loss.png')
plt.clf()

plt.plot(iterations, reg)
plt.ylabel('Regression Loss')
plt.xlabel('Iterations')
plt.savefig('reg_loss.png')
plt.clf()

