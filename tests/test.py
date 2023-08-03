import random


def calculate_avg_corners(corner_list):
    max_x, max_y = max(corner_list, key=lambda c: c[0])
    min_x, min_y = min(corner_list, key=lambda c: c[0])

    cleaned_corners = [(x, y) for x, y in corner_list if (x, y) != (max_x, max_y) and (x, y) != (min_x, min_y)]

    x_sum = sum(x for x, _ in cleaned_corners)
    y_sum = sum(y for _, y in cleaned_corners)
    num_points = len(cleaned_corners)

    avg_x = x_sum / num_points if num_points > 0 else 0
    avg_y = y_sum / num_points if num_points > 0 else 0

    avg_corners = (avg_x, avg_y)
    return avg_corners


"""
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
"""
