from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms, combine_frame_transforms
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
import math

@configclass
class HansCubeEnvCfg(DirectRLEnvCfg):
    episode_length_s = 5.0  # 500 timesteps at 100Hz
    decimation = 2
    action_space = 7
    observation_space = 22
    state_space = 0
    seed = 42
    
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        ),
    )
    
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    
    gripper_width = 55.0 * (math.pi / 180.0)
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/njt/Workspace/multi_arm_rl/source/isaaclab_assets/robot/elfin5_lebai.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, 
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "elfin_joint1": 0.0,
                "elfin_joint2": -0.383,
                "elfin_joint3": 1.221,
                "elfin_joint4": 0,
                "elfin_joint5": 0.523,
                "elfin_joint6": 2.094,
                "gripper_[lr]_joint1": gripper_width,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            # "elfin_joint1": ImplicitActuatorCfg(
            #     joint_names_expr=["elfin_joint[1]"],
            #     effort_limit_sim=200.0,
            #     stiffness=9.09995,
            #     damping=0.00364,
            # ),
            # "elfin_joint2": ImplicitActuatorCfg(
            #     joint_names_expr=["elfin_joint[2]"],
            #     effort_limit_sim=200.0,
            #     stiffness=145.00891,
            #     damping=0.058,
            # ),
            # "elfin_joint3": ImplicitActuatorCfg(
            #     joint_names_expr=["elfin_joint[3]"],
            #     effort_limit_sim=200.0,
            #     stiffness=1056.85144,
            #     damping=0.42274,
            # ),
            # "elfin_joint4": ImplicitActuatorCfg(
            #     joint_names_expr=["elfin_joint[4]"],
            #     effort_limit_sim=104.0,
            #     stiffness=466.87366,
            #     damping=0.18675,
            # ),
            # "elfin_joint5": ImplicitActuatorCfg(
            #     joint_names_expr=["elfin_joint[5]"],
            #     effort_limit_sim=34.0,
            #     stiffness=280.00146,
            #     damping=0.112,
            # ),
            # "elfin_joint6": ImplicitActuatorCfg(
            #     joint_names_expr=["elfin_joint[6]"],
            #     effort_limit_sim=34.0,
            #     stiffness=8.81607,
            #     damping=0.00353,
            # ),
            "elfin_shoulder1": ImplicitActuatorCfg(
                joint_names_expr=["elfin_joint[1-3]"],
                effort_limit_sim=200.0,
                stiffness=1000.0,
                damping=50.0,
            ),
            "elfin_shoulder2": ImplicitActuatorCfg(
                joint_names_expr=["elfin_joint[4]"],
                effort_limit_sim=104.0,
                stiffness=600.0,
                damping=30.0,
            ),
            "elfin_forearm": ImplicitActuatorCfg(
                joint_names_expr=["elfin_joint[5-6]"],
                effort_limit_sim=34.0,
                stiffness=200.0,
                damping=15.0,
            ),
            "elfin_hand": ImplicitActuatorCfg(
                joint_names_expr=["gripper_[lr]_joint1"],
                effort_limit_sim=40.0,
                stiffness=1000.0, 
                damping=20.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0
    )
    
    table_height = 0.055
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, table_height), 
                                                  rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
        ),
    )
    
    table = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    )
    
    marker_cfg = FRAME_MARKER_CFG.copy().replace(
        prim_path="/Visuals/TargetPosition",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1)
        )}
    )
    
    action_scale = 0.5
    
    reward_weights = {
        "reaching_object": 1.0,
        "lifting_object": 15.0,
        "object_goal_tracking": 16.0,
        "object_goal_tracking_fine_grained": 5.0,
        "initial_action_rate_penalty": -1e-4,
        "initial_joint_vel_penalty": -1e-4,
        "action_rate_penalty": -1e-1,
        "joint_vel_penalty": -1e-1,
    }
    
    desired_joint_pattern = "elfin_joint[1-6]|gripper_[lr]_joint1"
    curriculum_steps = 10000
    
    dof_pos_noise: float = 0.001 
    dof_vel_noise: float = 0.005
    object_pos_noise: float = 0.005
    
    ee_offset = 0.165
    reaching_std = 0.1
    goal_tracking_std = 0.3
    goal_tracking_fine_std = 0.05
    minimal_lift_height = 0.04
    
    
@configclass
class HansCubeEnvCfg_PLAY(HansCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.dof_pos_noise = 0.0
        self.dof_vel_noise = 0.0
        self.object_pos_noise = 0.0
        
class HansCubeEnv(DirectRLEnv):
    cfg: HansCubeEnvCfg
    
    def __init__(self, cfg: HansCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.target_poses = torch.zeros(self.num_envs, 7, device=self.device)
        self.previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.observed_joint_indices, _ = self._robot.find_joints(self.cfg.desired_joint_pattern)
        self.ee_link_index = self._robot.find_bodies("gripper_base_link")[0][0]
        
        self.gripper_width = self.cfg.gripper_width
        self.curriculum_steps = self.cfg.curriculum_steps
        self.reward_weights = self.cfg.reward_weights
        self.global_step_counter = 0
        self.action_scale = self.cfg.action_scale
        self.dof_pos_noise = self.cfg.dof_pos_noise
        self.dof_vel_noise = self.cfg.dof_vel_noise
        self.object_pos_noise = self.cfg.object_pos_noise
        
        self.arm_lower_limits = self._robot.data.soft_joint_pos_limits[0, :6, 0]
        self.arm_upper_limits = self._robot.data.soft_joint_pos_limits[0, :6, 1]
        
        self.ee_offset = self.cfg.ee_offset
        self.reaching_std = self.cfg.reaching_std
        self.goal_tracking_std = self.cfg.goal_tracking_std
        self.goal_tracking_fine_std = self.cfg.goal_tracking_fine_std
        self.minimal_height = self.cfg.minimal_lift_height
        
        self.table_height = self.cfg.table_height
        
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object        
        self.cfg.table.func("/World/envs/env_.*/Table", self.cfg.table, translation=(0.6, 0.0, 0.0), orientation=(0.707, 0.0, 0.0, 0.707))
        
        self._ee_marker = VisualizationMarkers(self.cfg.marker_cfg)
        self.scene.extras["ee_marker"] = self._ee_marker
        self._target_marker = VisualizationMarkers(self.cfg.marker_cfg)
        self.scene.extras["target_marker"] = self._target_marker
        
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()
        
        sim_utils.spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.global_step_counter += 1
        self.actions = actions.clone().clamp(-1.0, 1.0)
        
        arm_action = actions[:, :6] * self.action_scale
        arm_target = self._robot.data.default_joint_pos[:, :6] + arm_action
        arm_target = torch.clamp(arm_target, self.arm_lower_limits, self.arm_upper_limits)
        
        gripper_action_cmd = actions[:, 6]
        gripper_target_val = torch.where(
            gripper_action_cmd >= 0, 
            torch.tensor(0.0, device=self.device), 
            torch.tensor(self.gripper_width, device=self.device)
        )
        gripper_target = gripper_target_val.unsqueeze(1).repeat(1, 2)
        
        self.target_pos = torch.cat([arm_target, gripper_target], dim=1)
        
    def _apply_action(self):
        self._robot.set_joint_position_target(self.target_pos, joint_ids=self.observed_joint_indices)
        
    def _get_observations(self) -> dict:
        dof_pos = self._robot.data.joint_pos[:, self.observed_joint_indices]
        dof_vel = self._robot.data.joint_vel[:, self.observed_joint_indices]        
        robot_pos = self._robot.data.root_pos_w
        object_pos = self._object.data.root_pos_w
        object_pos_rel, _ = subtract_frame_transforms(robot_pos, self._robot.data.root_quat_w, object_pos)

        if self.dof_pos_noise > 0.0:
            dof_pos += torch.randn_like(dof_pos) * self.dof_pos_noise
        if self.dof_vel_noise > 0.0:
            dof_vel += torch.randn_like(dof_vel) * self.dof_vel_noise        
        if self.object_pos_noise > 0.0:
            object_pos_rel += torch.randn_like(object_pos_rel) * self.object_pos_noise
            
        default_pos = self._robot.data.default_joint_pos[:, self.observed_joint_indices]
        dof_pos_scaled = (dof_pos - default_pos)

        obs = torch.cat([
            dof_pos_scaled,         
            dof_vel,         
            object_pos_rel,         
            self.target_poses[:, :3], 
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        ee_link_pos_w = self._robot.data.body_pos_w[:, self.ee_link_index]
        ee_link_quat_w = self._robot.data.body_quat_w[:, self.ee_link_index]
        offset_pos = torch.tensor([0.0, 0.0, self.ee_offset], device=self.device).repeat(self.num_envs, 1)
        offset_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        ee_pos, _ = combine_frame_transforms(
            ee_link_pos_w, ee_link_quat_w,
            offset_pos, offset_quat
        )
        object_pos = self._object.data.root_pos_w
        distance_ee_obj = torch.norm(object_pos - ee_pos, dim=1)
        # is_closing = torch.where(self.actions[:, 6] > 0.0, 1.0, 0.0) 
        # is_near_obj = torch.where(distance_ee_obj < 0.1, 1.0, 0.0)

        # rew_grasping = is_closing * is_near_obj
        rew_reaching = 1 - torch.tanh(distance_ee_obj / self.reaching_std)
        
        # lift_goal = self.target_poses[:, 2]
        
        obj_height = object_pos[:, 2]
        # rew_lift_height = torch.clamp(torch.clamp(obj_height - self.minimal_height, min=0.0)
        #                               / (self.target_poses[:, 2] - self.minimal_height), max=1.0)
        
        # rew_lifting = rew_lift_height  * is_near_obj
        
        
        rew_action_rate = torch.sum(torch.square(self.actions - self.previous_actions), dim=1)
        rew_joint_vel = torch.sum(torch.square(self._robot.data.joint_vel[:, self.observed_joint_indices]), dim=1)
        
        target_pos_w = self._robot.data.root_pos_w + self.target_poses[:, :3]

        distance_obj_goal = torch.norm(target_pos_w - object_pos, dim=1)
        is_lifted = torch.where(obj_height >  self.minimal_height, 1.0, 0.0)
        rew_lifting = is_lifted
        rew_goal = is_lifted * (1 - torch.tanh(distance_obj_goal / self.goal_tracking_std))
        rew_goal_fine = is_lifted * (1 - torch.tanh(distance_obj_goal / self.goal_tracking_fine_std))
        
        if self.global_step_counter >= self.curriculum_steps:
            action_rate_penalty = self.reward_weights["action_rate_penalty"]
            joint_vel_penalty = self.reward_weights["joint_vel_penalty"]
        else:
            action_rate_penalty = self.reward_weights["initial_action_rate_penalty"]
            joint_vel_penalty = self.reward_weights["initial_joint_vel_penalty"]
            
        self._ee_marker.visualize(ee_pos, _)
        self._target_marker.visualize(target_pos_w)
        
        total_reward = (
            # rew_grasping +
            rew_reaching * self.reward_weights["reaching_object"] +
            rew_lifting * self.reward_weights["lifting_object"] +
            rew_goal * self.reward_weights["object_goal_tracking"] +
            rew_goal_fine * self.reward_weights["object_goal_tracking_fine_grained"] +
            rew_action_rate * action_rate_penalty +
            rew_joint_vel * joint_vel_penalty
        )
        
        log_dict = {
            # "grasping": (rew_grasping),
            "reaching_object": (rew_reaching * self.reward_weights["reaching_object"]),
            "lifting_object": (rew_lifting * self.reward_weights["lifting_object"]),
            "object_goal_tracking": (rew_goal * self.reward_weights["object_goal_tracking"]),
            "object_goal_tracking_fine_grained": (rew_goal_fine * self.reward_weights["object_goal_tracking_fine_grained"]),
            "action_rate": (rew_action_rate * action_rate_penalty),
            "joint_vel": (rew_joint_vel * joint_vel_penalty),
        }

        self.extras["log"] = log_dict
        for key, value in log_dict.items():
            self.extras[key] = value
        
        self.previous_actions = self.actions.clone()
        return total_reward
    
    
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算终止条件。"""
        # 1. 超时
        time_out = self.episode_length_buf >= self.max_episode_length
        # 2. 掉落 (Object dropping)
        died = self._object.data.root_pos_w[:, 2] < -0.05
        # 3. object_reached_goal
        # finshed = torch.norm(self._object.data.root_pos_w - self._object.data.root_pos_w[:, :3], dim=1) < 0.02
        return died, time_out #, finshed
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 2. 重置物体位置 (在桌面上随机)
        sampled_pos_x = sample_uniform(0.4, 0.8, (len(env_ids), 1), device=self.device)
        sampled_pos_y = sample_uniform(-0.25, 0.25, (len(env_ids), 1), device=self.device)
        object_pos = torch.cat([sampled_pos_x, sampled_pos_y, torch.full_like(sampled_pos_x, self.table_height)], dim=1)
        # 添加基座位置偏移
        object_pos += self.scene.env_origins[env_ids]
        
        object_rot = torch.zeros(len(env_ids), 4, device=self.device)
        object_rot[:, 0] = 1.0 
        object_vel = torch.zeros(len(env_ids), 6, device=self.device)
        root_state = torch.cat([object_pos, object_rot, object_vel], dim=1)
        self._object.write_root_state_to_sim(root_state, env_ids=env_ids)

        # 3. 重置目标 (target_poses)
        self.target_poses[env_ids, 0] = sample_uniform(0.4, 0.6, (len(env_ids),), device=self.device) # X
        self.target_poses[env_ids, 1] = sample_uniform(-0.25, 0.25, (len(env_ids),), device=self.device) # Y
        self.target_poses[env_ids, 2] = sample_uniform(0.25, 0.35, (len(env_ids),), device=self.device) # Z
        self.target_poses[:, 3] = 1.0
        # 清空历史动作
        self.previous_actions[env_ids] = 0.0