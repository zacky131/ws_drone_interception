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

def build_acados_mpc(
    ctrl_cfg: ControllerConfig,
    pursuer_cfg: PursuerConfig,
    sim_cfg: SimulationConfig,
    Q_pos: float,
    Q_T_pos: float,
) -> AcadosOcpSolver:
    """Build and compile the ACADOS OCP Solver for the MPC tracking problem."""
    
    if not _HAS_ACADOS:
        raise ImportError("Acados is not installed in the current environment.")
        
    ocp = AcadosOcp()
    
    # ── Parameters ────────────────────────────────────────────────────────
    N = ctrl_cfg.horizon
    dt = sim_cfg.dt
    tau = pursuer_cfg.actuator_time_constant
    alpha = min(dt / tau, 1.0) if tau > 1e-9 else 1.0
    
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
    model.name = 'drone_interception_mpc'
    
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
    
    # Parameters: p_t(3), v_t(3), wind(3)
    p_t = ca.SX.sym('p_t', 3)
    v_t = ca.SX.sym('v_t', 3)
    wind = ca.SX.sym('wind', 3)
    p = ca.vertcat(p_t, v_t, wind)
    model.p = p
    
    # Discrete Dynamics (x_{k+1} = f(x_k, u_k, p_k))
    a_app_new = a_app + alpha * (a_cmd - a_app)
    total_acc = a_app_new + wind
    v_new = v_p + total_acc * dt
    p_new = p_p + v_p * dt + 0.5 * total_acc * dt**2
    u_prev_new = a_cmd
    
    model.disc_dyn_expr = ca.vertcat(p_new, v_new, a_app_new, u_prev_new)
    
    # ── Cost Function ─────────────────────────────────────────────────────
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    dp = p_new - p_t
    dv = v_new - v_t
    du = (a_cmd - u_prev) / dt
    
    # Stage Cost
    cost_expr = Q_pos * ca.dot(dp, dp) + Q_vel * ca.dot(dv, dv) + \
                R * ca.dot(a_cmd, a_cmd) + R_rate * ca.dot(du, du)
    model.cost_expr_ext_cost = cost_expr
    
    # Terminal Cost
    dp_e = p_p - p_t
    dv_e = v_p - v_t
    cost_expr_e = Q_T_pos * ca.dot(dp_e, dp_e) + Q_T_vel * ca.dot(dv_e, dv_e)
    model.cost_expr_ext_cost_e = cost_expr_e
    
    # ── Constraints ───────────────────────────────────────────────────────
    # Bounds on u: a_cmd in [-a_max_axis, a_max_axis]
    ocp.constraints.lbu = np.array([-a_max_axis, -a_max_axis, -a_max_axis])
    ocp.constraints.ubu = np.array([a_max_axis, a_max_axis, a_max_axis])
    ocp.constraints.idxbu = np.array([0, 1, 2])
    
    # Nonlinear constraints: ‖a_cmd‖² ≤ a_max², ‖v_new‖² ≤ v_max², ‖du‖² ≤ jerk_max²
    h_expr = ca.vertcat(
        ca.dot(a_cmd, a_cmd),
        ca.dot(v_new, v_new),
        ca.dot(du, du)
    )
    model.con_h_expr = h_expr
    ocp.constraints.lh = np.array([0.0, 0.0, 0.0])
    ocp.constraints.uh = np.array([a_max**2, v_max**2, jerk_max**2])
    
    # Terminal nonlinear constraints: ‖v_p‖² ≤ v_max²
    h_expr_e = ca.vertcat(
        ca.dot(v_p, v_p)
    )
    model.con_h_expr_e = h_expr_e
    ocp.constraints.lh_e = np.array([0.0])
    ocp.constraints.uh_e = np.array([v_max**2])
    
    ocp.model = model
    
    # ── Solver Options ────────────────────────────────────────────────────
    ocp.dims.N = N
    ocp.solver_options.tf = N * dt  # Only used for consistency, model is DISCRETE
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = getattr(ctrl_cfg, 'acados_nlp_solver_type', 'SQP_RTI')
    ocp.solver_options.nlp_solver_max_iter = ctrl_cfg.solver_max_iter
    ocp.solver_options.qp_solver = getattr(
        ctrl_cfg, 'acados_qp_solver', 'PARTIAL_CONDENSING_HPIPM'
    )
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.print_level = ctrl_cfg.solver_print_level
    
    # Provide default values for parameters to prevent uninitialized memory issues
    ocp.parameter_values = np.zeros(9)
    
    # Provide default values for initial state
    ocp.constraints.lbx_0 = np.zeros(12)
    ocp.constraints.ubx_0 = np.zeros(12)
    ocp.constraints.idxbx_0 = np.arange(12)
    
    # ── Multiprocessing Compilation Safety ────────────────────────────────
    # ACADOS compiles C-code into `c_generated_code`. When running in parallel,
    # multiple workers will try to compile simultaneously which crashes.
    # We solve this by compiling in a unique temporary directory per worker.
    
    # Create a unique export directory for this solver instantiation. ACADOS
    # needs generated C code and a compiled shared library while the solver is
    # alive; keeping it out of the project root avoids clutter during MC runs.
    unique_id = uuid.uuid4().hex[:8]
    base_dir = (
        getattr(ctrl_cfg, "acados_export_dir", "")
        or os.environ.get("DRONE_ACADOS_EXPORT_DIR", "")
        or os.path.join(tempfile.gettempdir(), "drone_interception_acados")
    )
    export_dir = os.path.abspath(os.path.join(base_dir, f"acados_export_{unique_id}"))
    os.makedirs(export_dir, exist_ok=True)

    ocp.code_export_directory = export_dir

    # Compile and load solver
    solver = AcadosOcpSolver(
        ocp,
        json_file=os.path.join(export_dir, f"acados_ocp_{unique_id}.json"),
        build=True,
        generate=True,
    )

    if not getattr(ctrl_cfg, "acados_keep_export", True):
        atexit.register(shutil.rmtree, export_dir, ignore_errors=True)

    return solver, export_dir
