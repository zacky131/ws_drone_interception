"""PX4/Gazebo supervisor for GPN and native velocity/yaw-rate MPC."""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
import numpy as np
import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint
from drone_interception_px4 import experiment_supervisor as base
from drone_interception_px4.frames import enu_velocity_to_ned
from drone_interception_px4.px4_contract import INTERCEPTOR
from .controller import GPN, SRIVASTAVA, METHODS, V2ControllerAdapter

base.METHODS=METHODS; base.ExistingControllerAdapter=V2ControllerAdapter


def scalar(value, default=math.nan):
    try: return float(value)
    except (TypeError,ValueError): return default


class V2Supervisor(base.ExperimentSupervisor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.controller.reset(self.trial_seed,self.trajectory_id,self.condition.name)
        self._native_velocity=np.zeros(3); self._native_yaw_rate=0.0

    def _publish_acceleration(self, command):
        if self.method==GPN: return super()._publish_acceleration(command)
        info=self.controller.get_diagnostics(); velocity=np.asarray(info['native_velocity_command_enu'],float); yaw=float(info['native_yaw_rate_radps'])
        self._native_velocity=velocity.copy(); self._native_yaw_rate=yaw
        timestamp=int(self.get_clock().now().nanoseconds//1000); mode=OffboardControlMode(); mode.timestamp=timestamp
        mode.position=False; mode.velocity=True; mode.acceleration=False; mode.attitude=False; mode.body_rate=False; mode.thrust_and_torque=False; mode.direct_actuator=False
        self.mode_pubs[INTERCEPTOR.role].publish(mode); point=TrajectorySetpoint(); point.timestamp=timestamp
        point.position=[math.nan]*3; point.velocity=enu_velocity_to_ned(velocity).tolist(); point.acceleration=[math.nan]*3; point.jerk=[math.nan]*3; point.yaw=math.nan; point.yawspeed=-yaw
        self.setpoint_pubs[INTERCEPTOR.role].publish(point); return np.full(3,math.nan)

    def _run_step(self,tick_start_ns):
        if self.method==SRIVASTAVA:
            heading_ned=float(self.positions[INTERCEPTOR.role].heading)
            if math.isfinite(heading_ned):
                self.controller.interceptor_yaw_enu=math.atan2(math.sin(math.pi/2-heading_ned),math.cos(math.pi/2-heading_ned))
        previous=len(self.rows); super()._run_step(tick_start_ns)
        if len(self.rows)==previous:return
        info=self.controller.get_diagnostics(); row=self.rows[-1]
        for key in ('external_packet_source_timestamp_s','external_packet_arrival_timestamp_s','external_measurement_update_timestamp_s','external_posterior_timestamp_s'):
            source=key.removeprefix('external_'); row[key]=scalar(info.get(source))
        row['external_packet_accepted']=int(info.get('packet_accepted',0)); row['external_future_truth_access']=int(info.get('future_truth_access',0)); row['external_executed_truth_controller_access']=int(info.get('executed_truth_controller_access',0)); row['external_commanded_reference_access']=int(info.get('commanded_target_reference_access',0))
        for prefix,key in (('gpn_raw','raw_command'),('gpn_post_acceleration_limit','post_acceleration_limit'),('gpn_post_rate_limit','post_rate_limit')):
            values=np.asarray(info.get(key,np.full(3,np.nan)),float); row.update(self._flatten(prefix,values,'mps2'))
        row['gpn_range_guard_activated']=int(info.get('gpn_range_guard_activated',0)); row['external_acceleration_limited']=int(not np.allclose(info.get('raw_command',[0,0,0]),info.get('post_acceleration_limit',[0,0,0]),atol=1e-12)); row['external_rate_limited']=int(info.get('external_rate_limited',0))
        row.update(self._flatten('native_velocity_command',self._native_velocity,'mps')); row['native_yaw_rate_radps']=self._native_yaw_rate
        for key in ('srivastava_replan_attempted','srivastava_solve_attempts_total','srivastava_solve_successes_total','srivastava_solve_failures_total'):
            row[key]=int(info.get(key,0))
        row['srivastava_body_forward_mps']=scalar(info.get('srivastava_body_forward_mps')); row['srivastava_body_vertical_mps']=scalar(info.get('srivastava_body_vertical_mps'))
        if self.method==SRIVASTAVA:
            for axis in 'enu': row[f'acceleration_command_enu_{axis}_mps2']=math.nan

    def _write_outputs(self):
        super()._write_outputs(); path=self.output_dir/'summary.json'; summary=json.loads(path.read_text()); summary.update({'method_manuscript':self.method,'capture_distance_m':1.0,'controller_information_source':'corrected_M1_exact_source_time_CA_posterior','future_truth_controller_access':False,'executed_truth_controller_access':False,'native_command_interface':'acceleration' if self.method==GPN else 'PX4_velocity_plus_yawspeed'}); path.write_text(json.dumps(summary,indent=2)+'\n')


def main(argv=None):
    source_root=Path(os.environ.get('DRONE_INTERCEPTION_V3','.')); parser=argparse.ArgumentParser()
    parser.add_argument('--trajectory',type=Path,required=True); parser.add_argument('--trajectory-id',required=True); parser.add_argument('--family',required=True); parser.add_argument('--condition',choices=sorted(base.CONDITIONS),required=True); parser.add_argument('--method',choices=METHODS,required=True); parser.add_argument('--trial-seed',type=int,required=True); parser.add_argument('--config',type=Path,default=source_root/'configs/q2_revision_pilot.yaml'); parser.add_argument('--output-dir',type=Path,required=True); parser.add_argument('--timeout-s',type=float,default=90)
    args,ros_args=parser.parse_known_args(argv); rclpy.init(args=ros_args); node=None
    try:
        node=V2Supervisor(args.trajectory,args.trajectory_id,args.family,args.condition,args.method,args.trial_seed,args.config,args.output_dir,args.timeout_s)
        while rclpy.ok() and not node.done:rclpy.spin_once(node,timeout_sec=.1)
    finally:
        success=bool(node is not None and node.success)
        if node is not None:node.destroy_node()
        rclpy.shutdown()
    return 0 if success else 1

if __name__=='__main__': raise SystemExit(main())
