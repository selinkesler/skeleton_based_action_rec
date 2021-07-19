from typing import TypeVar

from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.data_structures.skeletons.skeleton_stickman_3d import SkeletonStickman3D

T = TypeVar('T', SkeletonStickman, SkeletonStickman3D)

def get_joints_default(skeleton: T):
    return [
        # Head region
        skeleton.joints.nose,
        skeleton.joints.neck,
        skeleton.joints.right_eye,

        # Torso
        skeleton.joints.neck,
        skeleton.joints.left_hip,
        skeleton.joints.right_hip,

        # Left shoulder
        skeleton.joints.left_shoulder,
        skeleton.joints.left_elbow,
        skeleton.joints.left_wrist,

        # Right shoulder
        skeleton.joints.right_shoulder,
        skeleton.joints.right_elbow,
        skeleton.joints.right_wrist,

        # Left leg
        skeleton.joints.left_hip,
        skeleton.joints.left_knee,
        skeleton.joints.left_ankle,

        # Right leg
        skeleton.joints.right_hip,
        skeleton.joints.right_knee,
        skeleton.joints.right_ankle,
    ]


def get_joints_full(skeleton: T):
    return list(skeleton.joints)


def get_joints_jhmdb(skeleton: SkeletonStickman):
    """
    JHMDB -> No annotated eyes / ears
    """
    return [
        # Head region
        skeleton.joints.nose,
        skeleton.joints.neck,
        skeleton.joints.hip_center,

        # Left shoulder
        skeleton.joints.left_shoulder,
        skeleton.joints.left_elbow,
        skeleton.joints.left_wrist,

        # Right shoulder
        skeleton.joints.right_shoulder,
        skeleton.joints.right_elbow,
        skeleton.joints.right_wrist,

        # Left leg
        skeleton.joints.left_hip,
        skeleton.joints.left_knee,
        skeleton.joints.left_ankle,

        # Right leg
        skeleton.joints.right_hip,
        skeleton.joints.right_knee,
        skeleton.joints.right_ankle,
    ]
