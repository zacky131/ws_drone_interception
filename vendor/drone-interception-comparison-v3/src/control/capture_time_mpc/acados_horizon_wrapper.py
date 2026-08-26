"""
src/control/capture_time_mpc/acados_horizon_wrapper.py

Phase 2 & Phase 4: ACADOS Horizon Solver Builder supporting stage-dependent dt and cost weights.
"""

import atexit
import os
import shutil
import tempfile
import uuid
import numpy as np
import casadi as ca

try:
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
    _HAS_ACADOS = True
except ImportError:
    _HAS_ACADOS = False

from src.utils.config_schema import ControllerConfig, PursuerConfig, SimulationConfig
from src.control.capture_time_mpc.horizon_config import HorizonSpecification

def build_acados_horizon_mpc(
    ctrl_cfg: ControllerConfig,
    pursuer_cfg: PursuerConfig,
    sim_cfg: SimulationConfig,
    horizon_spec: HorizonSpecification,
    Q_pos: float = 50.0,
    Q_T_pos: float = 500.0,
) -> tuple[AcadosOcpSolver, str]:
    """Build and compile ACADOS OCP Solver for arbitrary horizon grid and dynamic stage parameters."""
    if not _HAS_ACADOS:
        raise ImportError("Acados is not installed in the current environment.")

    ocp = AcadosOcp()
    N = horizon_spec.N
    node_dts = horizon_spec.node_dts

    tau = pursuer_cfg.actuator_time_constant
    a_max = pursuer_cfg.max_acceleration
    a_max_axis = pursuer_cfg.max_acceleration_per_axis
    v_max = pursuer_cfg.max_velocity
    jerk_max = pursuer_cfg.max_jerk

    Q_vel = ctrl_cfg.Q_vel
    R = ctrl_cfg.R_control
    R_rate = ctrl_cfg.R_rate
    Q_T_vel = ctrl_cfg.Q_terminal_vel

    # ── Model Definition ──────────────────────────────────────────────────
    model = AcadosModel()
    model.name = f'drone_horizon_mpc_{horizon_spec.name}'

    # States: p(3), v(3), a_app(3), u_prev(3)
    p_p = ca.SX.sym('p_p', 3)
    v_p = ca.SX.sym('v_p', 3)
    a_app = ca.SX.sym('a_app', 3)
    u_prev = ca.SX.sym('u_prev', 3)
    x = ca.vertcat(p_p, v_p, a_app, u_prev)
    model.x = x

    # Controls: a_cmd(3)
    a_cmd = ca.SX.sym('a_cmd', 3)
    model.u = a_cmd

    # Parameters: p_t(3), v_t(3), wind(3), dt_k(1), w_pos(1) -> 11 params
    p_t = ca.SX.sym('p_t', 3)
    v_t = ca.SX.sym('v_t', 3)
    wind = ca.SX.sym('wind', 3)
    dt_sym = ca.SX.sym('dt_k', 1)
    w_pos_sym = ca.SX.sym('w_pos', 1)
    p = ca.vertcat(p_t, v_t, wind, dt_sym, w_pos_sym)
    model.p = p

    # Discrete Dynamics with per-stage dt_sym
    alpha = ca.fmin(dt_sym / max(tau, 1e-6), 1.0)
    a_app_new = a_app + alpha * (a_cmd - a_app)
    total_acc = a_app_new + wind
    v_new = v_p + total_acc * dt_sym
    p_new = p_p + v_p * dt_sym + 0.5 * total_acc * dt_sym**2
    u_prev_new = a_cmd

    model.disc_dyn_expr = ca.vertcat(p_new, v_new, a_app_new, u_prev_new)

    # ── Cost Function ─────────────────────────────────────────────────────
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    dp = p_new - p_t
    dv = v_new - v_t
    du = (a_cmd - u_prev) / ca.fmax(dt_sym, 1e-4)

    # Stage Cost using parameter w_pos_sym
    cost_expr = w_pos_sym * ca.dot(dp, dp) + Q_vel * ca.dot(dv, dv) + \
                R * ca.dot(a_cmd, a_cmd) + R_rate * ca.dot(du, du)
    model.cost_expr_ext_cost = cost_expr

    # Terminal Cost
    dp_e = p_p - p_t
    dv_e = v_p - v_t
    cost_expr_e = Q_T_pos * ca.dot(dp_e, dp_e) + Q_T_vel * ca.dot(dv_e, dv_e)
    model.cost_expr_ext_cost_e = cost_expr_e

    # ── Constraints ───────────────────────────────────────────────────────
    ocp.constraints.lbu = np.array([-a_max_axis, -a_max_axis, -a_max_axis])
    ocp.constraints.ubu = np.array([a_max_axis, a_max_axis, a_max_axis])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    h_expr = ca.vertcat(
        ca.dot(a_cmd, a_cmd),
        ca.dot(v_new, v_new),
        ca.dot(du, du)
    )
    model.con_h_expr = h_expr
    ocp.constraints.lh = np.array([0.0, 0.0, 0.0])
    ocp.constraints.uh = np.array([a_max**2, v_max**2, jerk_max**2])

    h_expr_e = ca.vertcat(
        ca.dot(v_p, v_p)
    )
    model.con_h_expr_e = h_expr_e
    ocp.constraints.lh_e = np.array([0.0])
    ocp.constraints.uh_e = np.array([v_max**2])

    ocp.model = model

    # ── Solver Options ────────────────────────────────────────────────────
    ocp.dims.N = N
    ocp.solver_options.tf = horizon_spec.total_duration
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = ctrl_cfg.solver_max_iter
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.print_level = ctrl_cfg.solver_print_level

    # Set per-stage default parameters (p_t, v_t, wind, dt_k, w_pos)
    ocp.parameter_values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, node_dts[0], Q_pos])

    ocp.constraints.lbx_0 = np.zeros(12)
    ocp.constraints.ubx_0 = np.zeros(12)
    ocp.constraints.idxbx_0 = np.arange(12)

    unique_id = uuid.uuid4().hex[:8]
    base_dir = (
        getattr(ctrl_cfg, "acados_export_dir", "")
        or os.environ.get("DRONE_ACADOS_EXPORT_DIR", "")
        or os.path.join(tempfile.gettempdir(), "drone_interception_acados")
    )
    export_dir = os.path.abspath(os.path.join(base_dir, f"acados_horizon_{horizon_spec.name}_{unique_id}"))
    os.makedirs(export_dir, exist_ok=True)
    ocp.code_export_directory = export_dir

    solver = AcadosOcpSolver(
        ocp,
        json_file=os.path.join(export_dir, f"acados_ocp_{unique_id}.json"),
        build=True,
        generate=True,
    )

    for k in range(N):
        p_init = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, node_dts[k], Q_pos])
        solver.set(k, "p", p_init)

    if not getattr(ctrl_cfg, "acados_keep_export", True):
        atexit.register(shutil.rmtree, export_dir, ignore_errors=True)

    return solver, export_dir
