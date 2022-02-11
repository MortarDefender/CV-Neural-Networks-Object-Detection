import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_row(row):
    cls_loss = float(row[-7:-1])
    reg_loss = float(row[-47: -41])
    loss = float(row[-82:-76])
    return loss, reg_loss, cls_loss


with open('models/dogs_exp1/val_loss.txt', 'r') as f:
    iterations, cls, reg, loss = [], [], [], []
    curr_iter = 0
    for line in f.readlines():
        try:
            res = parse_row(line)
            loss.append(res[0])
            reg.append(res[1])
            cls.append(res[2])
            iterations.append(curr_iter)
            curr_iter += 5
        except:
            print(line)


plt.plot(iterations, loss)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.savefig('val_loss.png')
plt.clf()

plt.plot(iterations, cls)
plt.ylabel('Classification Loss')
plt.xlabel('Epochs')
plt.savefig('cls_val_loss.png')
plt.clf()

plt.plot(iterations, reg)
plt.ylabel('Regression Loss')
plt.xlabel('Epochs')
plt.savefig('reg_val_loss.png')
plt.clf()

