"""Per-run CSV/metadata logger; rosbag remains optional and external."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import AccelStamped
from nav_msgs.msg import Odometry, Path as NavPath
from std_msgs.msg import String
import yaml

from .config_utils import WORKSPACE_ROOT, default_config, default_field, load_mirror_config, load_yaml
from .geometry_utils import distance
from .ros_utils import diagnostic_values, odom_vectors, stamp_seconds


STEP_FIELDS = [
    "ros_time_s", "phase", "method", "trajectory", "condition", "seed",
    "target_truth_e_m", "target_truth_n_m", "target_truth_u_m", "target_truth_ve_mps", "target_truth_vn_mps", "target_truth_vu_mps", "target_truth_ae_mps2", "target_truth_an_mps2", "target_truth_au_mps2",
    "target_estimate_e_m", "target_estimate_n_m", "target_estimate_u_m", "target_estimate_ve_mps", "target_estimate_vn_mps", "target_estimate_vu_mps",
    "target_estimate_ae_mps2", "target_estimate_an_mps2", "target_estimate_au_mps2",
    "interceptor_px4_e_m", "interceptor_px4_n_m", "interceptor_px4_u_m", "interceptor_px4_ve_mps", "interceptor_px4_vn_mps", "interceptor_px4_vu_mps",
    "interceptor_px4_ae_mps2", "interceptor_px4_an_mps2", "interceptor_px4_au_mps2",
    "interceptor_gazebo_e_m", "interceptor_gazebo_n_m", "interceptor_gazebo_u_m",
    "command_raw_e_mps2", "command_raw_n_mps2", "command_raw_u_mps2", "command_applied_e_mps2", "command_applied_n_mps2", "command_applied_u_mps2",
    "separation_m", "minimum_separation_m", "captured", "capture_time_s", "px4", "safety_abort", "safety_reason", "prediction_horizon_json", "complete_loop_ms",
]


class LoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("mirror_logger")
        self.declare_parameter("config", str(default_config()))
        self.declare_parameter("field", str(default_field()))
        self.declare_parameter("method", "M1")
        self.declare_parameter("trajectory", "HT1")
        self.declare_parameter("condition", "HC1")
        self.declare_parameter("seed", 1)
        self.declare_parameter("repetition", 1)
        self.declare_parameter("output_root", "")
        self.config = load_mirror_config(self.get_parameter("config").value)
        self.field = load_yaml(self.get_parameter("field").value)
        self.method = str(self.get_parameter("method").value)
        self.trajectory = str(self.get_parameter("trajectory").value)
        self.condition = str(self.get_parameter("condition").value)
        self.seed = int(self.get_parameter("seed").value)
        self.repetition = int(self.get_parameter("repetition").value)
        self.run_id = f"PENDING_keyboard_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        configured_root = str(self.get_parameter("output_root").value)
        self.root = (Path(configured_root) if configured_root else WORKSPACE_ROOT / self.config["logging"]["root"]).resolve()
        self.output = self._allocate(self.root / self.run_id)
        self.output.mkdir(parents=True)
        self.get_logger().info(f"logging pending session to {self.output}")
        (self.output / "config_snapshot.yaml").write_text(yaml.safe_dump(self.config, sort_keys=True), encoding="utf-8")
        (self.output / "field_snapshot.yaml").write_text(yaml.safe_dump(self.field, sort_keys=True), encoding="utf-8")
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.metadata = {"run_id": self.output.name, "deterministic_run_id": self.run_id, "method": self.method, "trajectory": self.trajectory, "condition": self.condition, "seed": self.seed, "repetition": self.repetition, "started_utc": self.started_utc, "complete": False, "hardware_validation": False, "framework_role": "simulation_rehearsal"}
        self._write_metadata()
        self.step_file = (self.output / "steps.csv").open("w", newline="", encoding="utf-8", buffering=1)
        self.step_writer = csv.DictWriter(self.step_file, fieldnames=STEP_FIELDS)
        self.step_writer.writeheader()
        self.packet_file = (self.output / "telemetry_packets.csv").open("w", newline="", encoding="utf-8", buffering=1)
        self.packet_fields = ["packet_id", "condition", "source_time_s", "arrival_time_s", "actual_age_ms", "requested_age_ms", "dropped"]
        self.packet_writer = csv.DictWriter(self.packet_file, fieldnames=self.packet_fields)
        self.packet_writer.writeheader()
        self.phase_file = (self.output / "experiment_events.csv").open("w", newline="", encoding="utf-8", buffering=1)
        self.phase_writer = csv.DictWriter(self.phase_file, fieldnames=["ros_time_s", "phase", "captured", "capture_time_s", "minimum_separation_m", "safety_abort"])
        self.phase_writer.writeheader()
        self.latest: dict[str, object] = {}
        self.minimum_separation = math.inf
        self.last_phase = ""
        self.closed = False
        self.create_subscription(Odometry, "/ras_hw_mirror/target/truth", lambda msg: self.latest.__setitem__("truth", msg), 50)
        self.create_subscription(Odometry, "/ras_hw_mirror/target/estimate", lambda msg: self.latest.__setitem__("estimate", msg), 50)
        self.create_subscription(Odometry, "/ras_hw_mirror/interceptor/state/px4", lambda msg: self.latest.__setitem__("px4_state", msg), 50)
        self.create_subscription(Odometry, "/ras_hw_mirror/interceptor/state/ground_truth", lambda msg: self.latest.__setitem__("gz_state", msg), 50)
        self.create_subscription(AccelStamped, "/ras_hw_mirror/controller/command_raw", lambda msg: self.latest.__setitem__("raw_command", msg), 50)
        self.create_subscription(AccelStamped, "/ras_hw_mirror/controller/command_applied", lambda msg: self.latest.__setitem__("applied_command", msg), 50)
        self.create_subscription(NavPath, "/ras_hw_mirror/target/prediction_path", lambda msg: self.latest.__setitem__("prediction", msg), 10)
        self.create_subscription(DiagnosticArray, "/ras_hw_mirror/telemetry/status", self._packet, 50)
        self.create_subscription(DiagnosticArray, "/ras_hw_mirror/controller/status", self._controller_status, 50)
        self.create_subscription(DiagnosticArray, "/ras_hw_mirror/safety/status", self._safety, 20)
        self.create_subscription(DiagnosticArray, "/ras_hw_mirror/experiment/status", self._experiment, 20)
        self.create_subscription(String, "/ras_hw_mirror/scenario/selection", self._scenario, 10)
        self.timer = self.create_timer(1.0 / float(self.config["experiment"]["control_rate_hz"]), self._sample)

    @staticmethod
    def _allocate(path: Path) -> Path:
        if not path.exists():
            return path
        metadata = path / "metadata.json"
        if metadata.exists():
            try:
                if json.loads(metadata.read_text(encoding="utf-8")).get("complete"):
                    raise FileExistsError(f"refusing to overwrite completed run: {path}")
            except json.JSONDecodeError:
                pass
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return path.with_name(f"{path.name}_{suffix}")

    def _write_metadata(self) -> None:
        (self.output / "metadata.json").write_text(json.dumps(self.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _packet(self, msg: DiagnosticArray) -> None:
        values = diagnostic_values(msg)
        if all(key in values for key in self.packet_fields):
            self.packet_writer.writerow({key: values[key] for key in self.packet_fields})

    def _controller_status(self, msg: DiagnosticArray) -> None:
        self.latest["controller_status"] = diagnostic_values(msg)

    def _scenario(self, msg: String) -> None:
        if self.closed:
            return
        try:
            request = json.loads(msg.data)
            if request.get("stage") == "GS6":
                return
            method = str(request["method"])
            trajectory = str(request["trajectory"])
            condition = str(request["condition"])
            seed = int(request["seed"])
            repetition = int(request.get("repetition", self.repetition))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.get_logger().warning("ignored malformed scenario selection")
            return
        run_id = f"{trajectory}_{condition}_{method}_rep{repetition:02d}"
        requested_run_id = request.get("run_id")
        if requested_run_id is not None and str(requested_run_id) != run_id:
            self.get_logger().warning(
                f"ignored inconsistent requested run_id {requested_run_id!r}; using {run_id}"
            )
        if run_id != self.run_id:
            destination = self.root / run_id
            try:
                destination = self._allocate(destination)
            except FileExistsError:
                suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                destination = destination.with_name(f"{destination.name}_{suffix}")
            self.output.rename(destination)
            self.output = destination
        self.run_id = run_id
        self.method, self.trajectory, self.condition, self.seed = method, trajectory, condition, seed
        self.repetition = repetition
        self.metadata.update({
            "run_id": self.output.name,
            "deterministic_run_id": self.run_id,
            "stage": request.get("stage"),
            "method": method,
            "trajectory": trajectory,
            "condition": condition,
            "seed": seed,
            "repetition": repetition,
            "campaign": request.get("campaign"),
            "manifest_row": request.get("manifest_row"),
        })
        self._write_metadata()
        self.get_logger().info(f"logging selected scenario to {self.output}")

    def _safety(self, msg: DiagnosticArray) -> None:
        values = diagnostic_values(msg)
        if msg.status:
            values["message"] = msg.status[0].message
        self.latest["safety"] = values

    def _experiment(self, msg: DiagnosticArray) -> None:
        values = diagnostic_values(msg)
        self.latest["experiment"] = values
        phase = values.get("phase", "")
        if phase and phase != self.last_phase:
            self.phase_writer.writerow({"ros_time_s": f"{self._now():.9f}", "phase": phase, "captured": values.get("captured", "0"), "capture_time_s": values.get("capture_time_s", ""), "minimum_separation_m": values.get("minimum_separation_m", ""), "safety_abort": values.get("safety_abort", "0")})
            self.last_phase = phase
        if phase == "DONE":
            self._finish(values)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _state_values(msg, prefix: str, velocity: bool = True) -> dict:
        if msg is None:
            keys = [f"{prefix}_{axis}_m" for axis in "enu"]
            if velocity:
                keys += [f"{prefix}_v{axis}_mps" for axis in "enu"]
            return {key: math.nan for key in keys}
        p, v = odom_vectors(msg)
        result = {f"{prefix}_{axis}_m": float(value) for axis, value in zip("enu", p)}
        if velocity:
            result.update({f"{prefix}_v{axis}_mps": float(value) for axis, value in zip("enu", v)})
        return result

    @staticmethod
    def _command_values(msg, prefix: str) -> dict:
        values = [math.nan] * 3 if msg is None else [msg.accel.linear.x, msg.accel.linear.y, msg.accel.linear.z]
        return {f"{prefix}_{axis}_mps2": float(value) for axis, value in zip("enu", values)}

    def _sample(self) -> None:
        if self.closed:
            return
        truth = self.latest.get("truth")
        px4 = self.latest.get("px4_state")
        estimate = self.latest.get("estimate")
        experiment = self.latest.get("experiment", {})
        safety = self.latest.get("safety", {})
        separation = math.nan
        if truth is not None and px4 is not None:
            separation = distance(odom_vectors(truth)[0], odom_vectors(px4)[0])
            self.minimum_separation = min(self.minimum_separation, separation)
        row = {key: math.nan for key in STEP_FIELDS}
        row.update({"ros_time_s": self._now(), "phase": experiment.get("phase", ""), "method": self.method, "trajectory": self.trajectory, "condition": self.condition, "seed": self.seed})
        row.update(self._state_values(truth, "target_truth"))
        if truth is not None:
            row.update({f"target_truth_a{axis}_mps2": value for axis, value in zip("enu", [truth.twist.twist.angular.x, truth.twist.twist.angular.y, truth.twist.twist.angular.z])})
        row.update(self._state_values(estimate, "target_estimate"))
        if estimate is not None:
            row.update({f"target_estimate_a{axis}_mps2": value for axis, value in zip("enu", [estimate.twist.twist.angular.x, estimate.twist.twist.angular.y, estimate.twist.twist.angular.z])})
        row.update(self._state_values(px4, "interceptor_px4"))
        if px4 is not None:
            row.update({f"interceptor_px4_a{axis}_mps2": value for axis, value in zip("enu", [px4.twist.twist.angular.x, px4.twist.twist.angular.y, px4.twist.twist.angular.z])})
        row.update(self._state_values(self.latest.get("gz_state"), "interceptor_gazebo", velocity=False))
        row.update(self._command_values(self.latest.get("raw_command"), "command_raw"))
        row.update(self._command_values(self.latest.get("applied_command"), "command_applied"))
        prediction = self.latest.get("prediction")
        row["prediction_horizon_json"] = json.dumps([] if prediction is None else [[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z] for pose in prediction.poses], separators=(",", ":"))
        scientific_minimum = experiment.get("minimum_separation_m", self.minimum_separation)
        row.update({"separation_m": separation, "minimum_separation_m": scientific_minimum, "captured": experiment.get("captured", "0"), "capture_time_s": experiment.get("capture_time_s", ""), "px4": experiment.get("px4", ""), "safety_abort": safety.get("abort", "0"), "safety_reason": safety.get("reason", ""), "complete_loop_ms": self.latest.get("controller_status", {}).get("complete_loop_ms", math.nan)})
        self.step_writer.writerow(row)

    def _finish(self, experiment: dict) -> None:
        if self.closed:
            return
        capture_time = experiment.get("capture_time_s")
        self.metadata.update({"complete": True, "finished_utc": datetime.now(timezone.utc).isoformat(), "captured": experiment.get("captured") == "1", "capture_time_s": None if not capture_time else float(capture_time), "minimum_separation_m": float(experiment.get("minimum_separation_m", self.minimum_separation))})
        self._write_metadata()
        for stream in (self.step_file, self.packet_file, self.phase_file):
            stream.flush()
            stream.close()
        self.closed = True

    def destroy_node(self):
        if not self.closed:
            self.metadata.update({"complete": False, "finished_utc": datetime.now(timezone.utc).isoformat(), "termination": "node shutdown before DONE"})
            self._write_metadata()
            for stream in (self.step_file, self.packet_file, self.phase_file):
                stream.close()
            self.closed = True
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
