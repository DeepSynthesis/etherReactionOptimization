import torch

def pareto_front_2_dim(points):
    sort_points = points.copy()
    sort_points = sorted(sort_points, key=lambda x: (x[0], x[1]), reverse=True)
    max_y = -1e9
    ans_points = []
    for x in sort_points:
        if x[1] > max_y:
            max_y = x[1]
            ans_points.append(x)
    ans_points.reverse()
    return torch.Tensor(ans_points)