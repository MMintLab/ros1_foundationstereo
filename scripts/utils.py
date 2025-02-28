from arm_robots.panda import Panda


def setup_panda(panda_id: int = 1, has_gripper: bool=False):
    # Panda robot interface.
    # panda = Panda(arms_controller_name=f"/combined_panda/effort_joint_trajectory_controller_panda_{panda_id}",
    #               controller_name=f"effort_joint_trajectory_controller_panda_{panda_id}",
    #               robot_namespace=f'combined_panda',
    #               panda_name=f'panda_{panda_id}',
    #               has_gripper=has_gripper)
    panda = Panda(arms_controller_name=f"/combined_panda/effort_joint_trajectory_controller",
                  controller_name=f"effort_joint_trajectory_controller",
                  robot_namespace=f'combined_panda',
                  panda_name=f'panda_{panda_id}',
                  has_gripper=has_gripper)
    panda.connect()

    return panda
