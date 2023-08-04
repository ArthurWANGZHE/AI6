import random


def calculate_avg_corners(corner_list):
    max_x, max_y = max(corner_list, key=lambda c: c[0])
    min_x, min_y = min(corner_list, key=lambda c: c[0])
    print(max_x, max_y, min_x, min_y)
    max_x1, max_y1 = max(corner_list, key=lambda c: c[1])
    min_x1, min_y1 = min(corner_list, key=lambda c: c[1])
    print(max_x1, max_y1, min_x1, min_y1)

    x_l, y_l = [], []
    max_x_found, min_x_found = False, False
    max_y_found, min_y_found = False, False

    for x, y in corner_list:
        if not max_x_found and x == max_x:
            max_x_found = True
        elif not min_x_found and x == min_x:
            min_x_found = True
        elif not max_y_found and y == max_y1:
            max_y_found = True
        elif not min_y_found and y == min_y1:
            min_y_found = True
        else:
            x_l.append(x)
            y_l.append(y)
    print("x_l:", x_l)
    print("y_l:", y_l)


    y_l.append(max_y)
    y_l.append(min_y)
    x_l.append(max_x1)
    x_l.append(min_x1)

    x_sum = sum(x_l)
    y_sum = sum(y_l)
    num_points = len(x_l)

    avg_x = x_sum / num_points if num_points > 0 else 0
    avg_y = y_sum / num_points if num_points > 0 else 0

    print("x_l:", x_l)
    print("y_l:", y_l)

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

