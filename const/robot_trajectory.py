"""Robot trajectory constants for direct trajectory prediction"""
NUM_ROBOT_COORDS = 12  # 4 arm points x 3 coordinates

"""
Robot arm trajectory points
"""
robot_point_indices = {
    0: "Left_L1",
    1: "Left_L2",
    2: "Right_R1",
    3: "Right_R2"
}

"""
Robot arm trajectory point names for CSV output
"""
robot_point_names = ["Left_L1", "Left_L2", "Right_R1", "Right_R2"]

"""
Robot arm coordinate field names in H5 files
"""
robot_coord_fields = [
    "left_arm_L1", "left_arm_L2", "right_arm_R1", "right_arm_R2"
]

"""
Robot arm side indices for mirroring: (left_index, right_index)
"""
robot_side_idx = [
    (0, 2),  # Left_L1 <-> Right_R1
    (1, 3),  # Left_L2 <-> Right_R2
]

def flip_robot_sides(robot_coords):
    """
    Flip the positions of left-side robot points for right-side robot points and vice-versa.

    Args:
        robot_coords (torch.Tensor): Tensor containing robot coordinates [4, 3].

    Returns:
        torch.Tensor: Tensor with flipped robot coordinates.
    """
    flipped_coords = robot_coords.clone()
    for l_idx, r_idx in robot_side_idx:
        flipped_coords[l_idx, :], flipped_coords[r_idx, :] = robot_coords[r_idx, :].clone(), robot_coords[l_idx, :].clone()
    return flipped_coords