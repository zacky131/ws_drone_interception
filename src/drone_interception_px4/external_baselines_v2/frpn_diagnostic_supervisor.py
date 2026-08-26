"""Enhanced-logging-only FRPN diagnostics; never evaluation evidence."""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import rclpy
from external_baselines.supervisor import ExternalBaselineSupervisor, base


class FRPNDiagnosticSupervisor(ExternalBaselineSupervisor):
    def _run_step(self,tick_start_ns):
        previous=len(self.rows); super()._run_step(tick_start_ns)
        if len(self.rows)==previous:return
        info=self.controller.get_diagnostics(); raw=np.asarray(info['external_raw_command'],float)
        axis=np.clip(raw,-self.controller.safety.max_acceleration_axis,self.controller.safety.max_acceleration_axis); norm=np.linalg.norm(axis)
        if norm>self.controller.safety.max_acceleration:axis*=self.controller.safety.max_acceleration/norm
        applied=np.array([self.rows[-1][f'acceleration_command_enu_{a}_mps2'] for a in 'enu'])
        self.rows[-1].update(self._flatten('frpn_raw',raw,'mps2')); self.rows[-1].update(self._flatten('frpn_post_acceleration_limit',axis,'mps2')); self.rows[-1].update(self._flatten('frpn_post_rate_limit',applied,'mps2'))
        self.rows[-1]['frpn_G']=float(info['frpn_G']); self.rows[-1]['frpn_W']=float(info['frpn_W']); self.rows[-1]['diagnostic_only']=1


def main(argv=None):
    source_root=Path(os.environ.get('DRONE_INTERCEPTION_V3','.')); p=argparse.ArgumentParser()
    p.add_argument('--trajectory',type=Path,required=True); p.add_argument('--trajectory-id',required=True); p.add_argument('--family',required=True); p.add_argument('--condition',choices=sorted(base.CONDITIONS),required=True); p.add_argument('--method',choices=('FRPN',),default='FRPN'); p.add_argument('--trial-seed',type=int,required=True); p.add_argument('--config',type=Path,default=source_root/'configs/q2_revision_pilot.yaml'); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--timeout-s',type=float,default=90)
    args,ros_args=p.parse_known_args(argv); rclpy.init(args=ros_args); node=None
    try:
        node=FRPNDiagnosticSupervisor(args.trajectory,args.trajectory_id,args.family,args.condition,args.method,args.trial_seed,args.config,args.output_dir,args.timeout_s)
        while rclpy.ok() and not node.done:rclpy.spin_once(node,timeout_sec=.1)
    finally:
        success=bool(node is not None and node.success)
        if node is not None:node.destroy_node()
        rclpy.shutdown()
    return 0 if success else 1
if __name__=='__main__':raise SystemExit(main())
