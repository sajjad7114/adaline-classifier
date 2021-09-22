import matplotlib.pyplot as plt
from random import shuffle


def net_input(xin, w):
    yin = 0
    for i in range(len(w)):
        yin += w[i] * xin[i]
    return yin


def predict(xin, w, theta):
    yin = net_input(xin, w)
    if yin > theta:
        return 1
    if yin < -theta:
        return -1
    return 0


def train(s_in, y_out, w, alpha):
    count = 0
    for i in range(len(s_in)):
        xin = s_in[i]
        y_out_h = net_input(xin, w)
        d = y_out[i] - y_out_h
        count += d ** 2
        for j in range(len(w)):
            w[j] += alpha * d * xin[j]
    return count


def score(s_in, y_out, w, t):
    count_error = 0
    count_correct = 0
    for i in range(len(s_in)):
        xin = s_in[i]
        y_out_h = predict(xin, w, t)
        if y_out_h != y_out[i]:
            count_error += 1
        else:
            count_correct += 1
    return count_correct / (count_error + count_correct)


if __name__ == "__main__":
    EPOCHS = 25
    ALPHA = 0.01
    THETA = 0
    s = []
    y_out = []
    w = [0, 0, 0]
    with open("iris.data", 'r') as file:
        total = file.read().split()
        for sample in total:
            data = sample.split(',')
            if data[4] != "Iris-setosa":
                s.append([1, float(data[1]), float(data[3])])
                if data[4] == 'Iris-virginica':
                    y_out.append(1)
                if data[4] == 'Iris-versicolor':
                    y_out.append(-1)
    train_xx = s[:40] + s[50: 90]
    train_yy = y_out[:40] + y_out[50: 90]
    test_x = s[40:50] + s[90:]
    test_y = y_out[40:50] + y_out[90:]
    li = list(range(len(train_xx)))
    shuffle(li)
    train_x = train_xx.copy()
    train_y = train_yy.copy()
    for i in range(len(train_xx)):
        train_x[li[i]] = train_xx[i]
        train_y[li[i]] = train_yy[i]
    error_count = []
    for i in range(EPOCHS):
        errors = train(train_x, train_y, w, ALPHA)
        error_count.append(errors)
    print("W:", w)
    plt.figure()
    for i in range(len(train_x)):
        tx = train_x[i]
        ty = train_y[i]
        color = "green" if ty == 1 else "red"
        x = tx[1]
        y = tx[2]
        plt.scatter(x, y, c=color)
    X = [0, 10]
    Y = []
    for x in X:
        y = (-w[0] - w[1]*x) / w[2]
        Y.append(y)
    plt.plot(X, Y)
    plt.title("Train data")
    plt.figure()
    plt.plot(list(range(EPOCHS)), error_count)
    plt.title("Errors")
    plt.xlabel("Epochs")
    print("accuracy:", int(score(test_x, test_y, w, THETA) * 100), '%')
    plt.show()

