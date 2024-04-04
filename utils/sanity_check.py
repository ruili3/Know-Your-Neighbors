import numpy as np
import torch

import matplotlib.pyplot as plt
import os



def divide_into_groups(points, axis='z', group_num=4):
    """
    Divides the points into 4 groups based on the given axis.

    Parameters:
    points (Tensor): Tensor of shape (B, n_pts, 3) representing 3D points.
    axis (str): Either 'x' or 'z' indicating which axis to sort and divide along.

    Returns:
    List[Tensor]: A list of 4 tensors each of shape (B, n_pts//4, 3), representing the 4 groups.
    List[Tuple]: A list of 4 tuples each containing two floats representing the min and max distance for each group.
    """
    B, n_pts, _ = points.shape

    # Determine which dimension to sort based on
    dim_to_sort = 0 if axis == 'x' else 2

    # Sort the points along the specified axis
    _, indices = torch.sort(points[:, :, dim_to_sort], dim=1)
    sorted_points = torch.gather(points, 1, indices.unsqueeze(-1).expand(B, n_pts, 3))

    # Divide into 4 groups
    group_size = n_pts // group_num
    groups = []
    ranges = []
    print("group size:{}".format(group_size))
    for i in range(group_num):
        start = i * group_size
        end = (i + 1) * group_size if i < 3 else n_pts  # Handle the last group
        group = sorted_points[:, start:end, :]
        groups.append(group)

        # Find the distance range for each group
        min_distance = group[:, :, dim_to_sort].min().item()
        max_distance = group[:, :, dim_to_sort].max().item()
        ranges.append((min_distance, max_distance))
        print("group:{}, z-min:{}, z-max:{}".format(i, min_distance, max_distance))

    return groups, ranges


def divide_into_evenly_spaced_groups(points, axis='z'):
    """
    Divides the points into 4 groups based on evenly spaced distances along the specified axis.

    Parameters:
    points (Tensor): Tensor of shape (B, n_pts, 3) representing 3D points.
    axis (str): Either 'x' or 'z' indicating which axis to consider.

    Returns:
    List[Tensor]: A list of 4 tensors representing the 4 groups. The size of each group may vary.
    """
    B, n_pts, _ = points.shape

    # Determine which dimension to consider based on the axis
    dim_to_consider = 0 if axis == 'x' else 2

    # Find the minimum and maximum distance along the specified axis
    min_distance = points[:, :, dim_to_consider].min().item()
    max_distance = points[:, :, dim_to_consider].max().item()

    # Create 4 evenly spaced intervals within the distance range
    intervals = torch.linspace(min_distance, max_distance, steps=5)

    # print("intervals:{}".format(intervals))

    groups = []
    sizes = []

    # Divide points into 4 groups based on which interval they fall into
    for i in range(4):
        lower_bound = intervals[i]
        upper_bound = intervals[i + 1]

        mask = (points[:, :, dim_to_consider] >= lower_bound) & (points[:, :, dim_to_consider] < upper_bound)
        group = points[mask]
        
        # Reshape to original batch shape, but the number of points may differ now
        group = group.view(B, -1, 3)

        groups.append(group)
        print(f"Group {lower_bound}-{upper_bound}: Size {group.shape[1]}")


    return groups