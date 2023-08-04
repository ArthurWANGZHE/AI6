import random


def calculate_avg_corners(corner_list):
    max_x, max_y = max(corner_list, key=lambda c: c[0])
    min_x, min_y = min(corner_list, key=lambda c: c[0])
    max_x1, max_y1 = max(corner_list, key=lambda c: c[1])
    min_x1, min_y1 = min(corner_list, key=lambda c: c[1])

    x_l, y_l = [], []


    for x, y in corner_list:
        x_l.append(x)
        y_l.append(y)

    y_l.remove(max_y1)
    y_l.remove(min_y1)
    x_l.remove(max_x)
    x_l.remove(min_x)

    x_sum = sum(x_l)
    y_sum = sum(y_l)
    num_points = len(x_l)

    avg_x = x_sum / num_points if num_points > 0 else 0
    avg_y = y_sum / num_points if num_points > 0 else 0

    return avg_x, avg_y

tes=[]
for i in range(1,10):
    a=random.randint(1,100)
    b=random.randint(1,100)
    c=(a,b)
    tes.append(c)
    i = i+1

print(tes)
te = calculate_avg_corners(tes)
print(te)

