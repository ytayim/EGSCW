

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from base_case import (
    load_data,
    default_parameters,
    build_policy_params,
    run_simulation,
    run_optimisation,
    policy_self_consumption,
    policy_tariff_aware,
)

STEPS_PER_DAY = 48  # 30-min resolution


# PV forecast models


def forecast_pv(pv_history, t, H, method, ma_days=3,
                steps_per_day=STEPS_PER_DAY):
    """Forecast PV over the next H steps from time t.

    Note: only PV is forecast. Load and tariffs are assumed to be known
    (load is deterministic in this dataset; tariffs are published day-ahead).
    """
    n = len(pv_history)
    H_eff = min(H, n - t)
    fc = np.zeros(H_eff)

    if method == 'perfect':
        fc = pv_history[t:t + H_eff].copy()

    elif method == 'persistence':
        fc[:] = pv_history[t]

    elif method == 'moving_average':
        # For the first few days history is thin; fall back to the current
        # observed value, which makes early-horizon MA behave like persistence.
        for k in range(H_eff):
            samples = [
                pv_history[(t + k) - d * steps_per_day]
                for d in range(1, ma_days + 1)
                if (t + k) - d * steps_per_day >= 0
            ]
            fc[k] = np.mean(samples) if samples else pv_history[t]

    else:
        raise ValueError(f"Unknown forecast method: {method}")

    return fc


# Single-window LP solver


def solve_lp_window(pv_fc, load_fc, tariff_imp, tariff_exp,
                    soc0, params,
                    soc_terminal=None, terminal_value=0.0):
    H = len(pv_fc)
    dt = params['dt_hours']
    cap = params['capacity_kwh']
    etac = params['eta_ch']
    etad = params['eta_dis']
    pmax_c = params['max_charge_kw']
    pmax_d = params['max_discharge_kw']
    reserve = params.get('soc_min_reserve', 0.0)

    p_ch = cp.Variable(H, nonneg=True)
    p_dis = cp.Variable(H, nonneg=True)
    p_imp = cp.Variable(H, nonneg=True)
    p_exp = cp.Variable(H, nonneg=True)
    soc = cp.Variable(H + 1)

    constraints = [
        soc[0] == soc0,
        p_ch <= pmax_c,
        p_dis <= pmax_d,
        soc >= reserve,
        soc <= cap,
        pv_fc + p_imp + p_dis == load_fc + p_exp + p_ch,
        soc[1:] == soc[:-1] + etac * p_ch * dt - p_dis * dt / etad,
    ]

    if soc_terminal is not None:
        constraints.append(soc[H] >= soc_terminal)

    grid_cost = cp.sum(
        cp.multiply(tariff_imp, p_imp) * dt
        - cp.multiply(tariff_exp, p_exp) * dt
    )

    objective = grid_cost - terminal_value * soc[H] if terminal_value != 0.0 else grid_cost

    problem = cp.Problem(cp.Minimize(objective), constraints)

    installed = cp.installed_solvers()
    solver_order = ["CLARABEL", "OSQP", "SCS", "ECOS"]
    solved = False

    for solver_name in solver_order:
        if solver_name in installed:
            try:
                problem.solve(solver=solver_name)
                if problem.status in ("optimal", "optimal_inaccurate"):
                    solved = True
                    break
            except Exception:
                continue

    if not solved:
        try:
            problem.solve()
            if problem.status in ("optimal", "optimal_inaccurate"):
                solved = True
        except Exception:
            pass

    if not solved:
        raise RuntimeError(
            f"LP window failed: status={problem.status}, installed_solvers={installed}"
        )

    return {
        'p_ch': np.asarray(p_ch.value).flatten(),
        'p_dis': np.asarray(p_dis.value).flatten(),
        'p_imp': np.asarray(p_imp.value).flatten(),
        'p_exp': np.asarray(p_exp.value).flatten(),
        'soc': np.asarray(soc.value).flatten(),
        'cost': float(problem.value),
    }


# Receding-horizon controller


def run_receding_horizon(df, params, H, forecast_method, ma_days=3):
    n = len(df)

    if forecast_method == 'perfect':
        H = n

    dt = params['dt_hours']
    cap = params['capacity_kwh']
    etac = params['eta_ch']
    etad = params['eta_dis']
    reserve = params.get('soc_min_reserve', 0.0)

    pv_actual = df['pv'].to_numpy()
    load_actual = df['load'].to_numpy()
    tariff_i = df['tariff_import'].to_numpy()
    tariff_e = df['tariff_export'].to_numpy()

    soc = np.zeros(n + 1)
    soc[0] = params['soc_init']

    p_charge = np.zeros(n)
    p_discharge = np.zeros(n)
    p_import = np.zeros(n)
    p_export = np.zeros(n)
    pv_forecast_next_step = np.full(n, np.nan)

    energy_balance_residual = np.zeros(n)
    soc_equation_residual = np.zeros(n)

    avg_buy_tariff = float(np.mean(tariff_i))

    err_abs_sum = 0.0
    err_sq_sum = 0.0
    err_sum = 0.0
    err_count = 0

    for t in range(n):
        H_eff = min(H, n - t)

        pv_fc = forecast_pv(pv_actual, t, H_eff, forecast_method, ma_days=ma_days)
        pv_fc[0] = pv_actual[t]  # current PV is observed

        load_fc = load_actual[t:t + H_eff]
        ti_fc = tariff_i[t:t + H_eff]
        te_fc = tariff_e[t:t + H_eff]

        if forecast_method == 'perfect':
            terminal_value = 0.0
            soc_terminal = params['soc_init']
        else:
            # Forward-looking shadow price: mean tariff over the upcoming window.
            window_mean_tariff = float(np.mean(ti_fc))
            terminal_value = window_mean_tariff * etad
            soc_terminal = None

        sol = solve_lp_window(
            pv_fc, load_fc, ti_fc, te_fc,
            soc[t], params,
            soc_terminal=soc_terminal,
            terminal_value=terminal_value,
        )

        p_ch = float(sol['p_ch'][0])
        p_dis = float(sol['p_dis'][0])

        net = load_actual[t] + p_ch - pv_actual[t] - p_dis
        p_imp, p_exp = (net, 0.0) if net >= 0 else (0.0, -net)

        soc_next_raw = soc[t] + etac * p_ch * dt - p_dis * dt / etad
        soc_next = min(max(soc_next_raw, reserve), cap)
        soc[t + 1] = soc_next

        p_charge[t] = p_ch
        p_discharge[t] = p_dis
        p_import[t] = p_imp
        p_export[t] = p_exp

        if H_eff >= 2:
            pv_forecast_next_step[t] = pv_fc[1]

        for k in range(1, H_eff):
            if t + k < n:
                e = pv_fc[k] - pv_actual[t + k]
                err_abs_sum += abs(e)
                err_sq_sum += e * e
                err_sum += e
                err_count += 1

        energy_balance_residual[t] = (
            pv_actual[t] + p_imp + p_dis
            - load_actual[t] - p_exp - p_ch
        )
        soc_equation_residual[t] = soc_next - soc_next_raw

    delta_soc = soc[-1] - params['soc_init']
    terminal_soc_adjustment = -delta_soc * avg_buy_tariff
    electricity_cost = float(np.sum(tariff_i * p_import - tariff_e * p_export) * dt)
    adjusted_cost = electricity_cost + terminal_soc_adjustment

    if err_count > 0:
        forecast_error = {
            'MAE': err_abs_sum / err_count,
            'RMSE': float(np.sqrt(err_sq_sum / err_count)),
            'bias': err_sum / err_count,
        }
    else:
        forecast_error = {'MAE': 0.0, 'RMSE': 0.0, 'bias': 0.0}

    return {
        'soc': soc,
        'p_charge': p_charge,
        'p_discharge': p_discharge,
        'p_import': p_import,
        'p_export': p_export,
        'pv_forecast_next_step': pv_forecast_next_step,
        'energy_balance_residual': energy_balance_residual,
        'soc_equation_residual': soc_equation_residual,
        'electricity_cost': electricity_cost,
        'adjusted_cost': adjusted_cost,
        'terminal_soc_adjustment': terminal_soc_adjustment,
        'total_throughput_kwh': float(np.sum(p_charge + p_discharge) * dt),
        'forecast_error': forecast_error,
        'H': H,
        'forecast_method': forecast_method,
    }


# Base-case reference costs


def compute_reference_costs(df, params):
    """Run P1/P2/P3 from base_case and return their adjusted costs."""
    print("Computing reference costs from base case...")
    p1 = run_simulation(df, params, policy_self_consumption)["total_cost_adjusted"]
    print(f"  P1 Self-Consumption : £{p1:.4f}")
    p2 = run_simulation(df, params, policy_tariff_aware)["total_cost_adjusted"]
    print(f"  P2 Tariff-Aware     : £{p2:.4f}")
    p3 = run_optimisation(df, params)["total_cost_adjusted"]
    print(f"  P3 LP Optimisation  : £{p3:.4f}")
    return p1, p2, p3


# Experiment Runner


def run_forecast_experiments(df, params,
                             horizons=(2, 4, 8, 12, 24, 48),
                             methods=('persistence', 'moving_average')):
    results = {}
    for method in methods:
        for H in horizons:
            print(f"  P4  method={method:<18s}  H={H:>3d} ...", end=' ', flush=True)
            res = run_receding_horizon(df, params, H, method)
            results[(method, H)] = res
            print(f"cost = £{res['adjusted_cost']:.2f}  (MAE={res['forecast_error']['MAE']:.3f} kW)")
    return results


# Verification


def verify_p4(df, params, cost_p3_reference, tol_kwh=1e-4, tol_gbp=1e-2):
    print("=" * 60)
    print("P4 VERIFICATION")
    print("=" * 60)

    res_perfect = run_receding_horizon(
        df, params, H=STEPS_PER_DAY, forecast_method='perfect'
    )
    diff = abs(res_perfect['adjusted_cost'] - cost_p3_reference)

    print(f"(iii) Perfect-forecast MPC cost : £{res_perfect['adjusted_cost']:.4f}")
    print(f"      P3 global LP cost         : £{cost_p3_reference:.4f}")
    print(f"      |diff|                    : £{diff:.6f}   {'PASS' if diff < tol_gbp else 'FAIL'}")

    max_eb = float(np.max(np.abs(res_perfect['energy_balance_residual'])))
    max_soc = float(np.max(np.abs(res_perfect['soc_equation_residual'])))
    print(f"(i)   max |energy balance residual| : {max_eb:.2e}   {'PASS' if max_eb < tol_kwh else 'FAIL'}")
    print(f"(ii)  max |SOC dynamics residual|   : {max_soc:.2e}   {'PASS' if max_soc < tol_kwh else 'FAIL'}")

    res_h1 = run_receding_horizon(df, params, H=1, forecast_method='persistence')
    print(f"(iv)  H=1 persistence cost   : £{res_h1['adjusted_cost']:.2f}  "
          f"(sanity check only; 1-step horizon uses observed current PV)")
    print("=" * 60)

    return {'perfect': res_perfect, 'H1': res_h1}


# Plots


def _best_horizon_per_method(results_p4):
    """Return {method: H*} where H* minimises adjusted cost for that method."""
    methods = sorted({m for (m, _) in results_p4})
    best = {}
    for m in methods:
        entries = [(H, results_p4[(m, H)]['adjusted_cost'])
                   for (mm, H) in results_p4 if mm == m]
        best[m] = min(entries, key=lambda x: x[1])[0]
    return best


def plot_p4_results(results_p4, cost_p1, cost_p2, cost_p3):
    # Each method's best-performing horizon — shown in the cost chart
    best_H = _best_horizon_per_method(results_p4)
    H_pers = best_H.get('persistence')
    H_ma = best_H.get('moving_average')

    # ---------- Figure 1: cost comparison using best-per-method ----------
    labels = [
        'P1\nSelf-cons.',
        'P2\nTariff-aware',
        f'P4-persist\n(H={H_pers}, best)',
        f'P4-MA\n(H={H_ma}, best)',
        'P3\nLP (oracle)'
    ]
    costs = [
        cost_p1,
        cost_p2,
        results_p4[('persistence', H_pers)]['adjusted_cost'],
        results_p4[('moving_average', H_ma)]['adjusted_cost'],
        cost_p3
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ['grey', 'orange', 'steelblue', 'teal', 'green']
    bars = ax.bar(labels, costs, color=colours)
    ax.axhline(cost_p3, ls='--', color='green', alpha=0.6,
               label=f'Optimal = £{cost_p3:.2f}')
    ax.axhline(cost_p1, ls=':', color='grey', alpha=0.6,
               label=f'P1 baseline = £{cost_p1:.2f}')
    for b, c in zip(bars, costs):
        ax.annotate(f'£{c:.2f}', (b.get_x() + b.get_width() / 2, c),
                    ha='center', va='bottom')
    ax.set_ylabel("Terminal-SOC-Adjusted Cost (£)")
    ax.set_title("Policy Cost Comparison (P4 at best horizon per method)")
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(costs) * 1.15)
    plt.tight_layout()
    plt.savefig("fig_p4_cost_comparison.png", dpi=300)
    plt.close()

    # ---------- Figure 2: horizon sensitivity ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, marker, colour in [('persistence', 'o', 'tab:blue'),
                                   ('moving_average', 's', 'tab:orange')]:
        Hs = sorted({H for (m, H) in results_p4 if m == method})
        cs = [results_p4[(method, H)]['adjusted_cost'] for H in Hs]
        ax.plot(Hs, cs, marker=marker, color=colour, label=f'P4 — {method}')
    ax.axhline(cost_p3, ls='--', color='green', label='P3 (perfect foresight)')
    ax.axhline(cost_p1, ls=':', color='grey', label='P1 (no foresight)')
    ax.set_xlabel("Horizon H (steps, 30 min each)")
    ax.set_ylabel("Adjusted Cost (£)")
    ax.set_title("Diminishing Returns of Forecast Horizon")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig_p4_horizon_sensitivity.png", dpi=300)
    plt.close()

    # ---------- Figure 3: forecast error vs cost penalty ----------

    p1_penalty = cost_p1 - cost_p3
    Hs_all = sorted({H for (_, H) in results_p4})
    cmap = plt.cm.viridis
    H_min, H_max = min(Hs_all), max(Hs_all)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, marker in [('persistence', 'o'), ('moving_average', 's')]:
        for (m, H), res in results_p4.items():
            if m != method:
                continue
            # Normalise H for colour: log scale spreads the small Hs better
            t_norm = (np.log(H) - np.log(H_min)) / (np.log(H_max) - np.log(H_min))
            colour = cmap(t_norm)
            ax.scatter(
                res['forecast_error']['MAE'],
                res['adjusted_cost'] - cost_p3,
                marker=marker, s=90, c=[colour],
                edgecolor='black', linewidth=0.5,
            )
            # Label each point with its horizon for easy reading
            ax.annotate(f'H={H}',
                        (res['forecast_error']['MAE'],
                         res['adjusted_cost'] - cost_p3),
                        textcoords='offset points', xytext=(6, 3), fontsize=8)

    ax.axhline(p1_penalty, ls=':', color='grey',
               label=f'P1 reference (£{p1_penalty:.2f} above P3)')

    import matplotlib.lines as mlines
    persist_handle = mlines.Line2D([], [], color='w', marker='o',
                                   markeredgecolor='black',
                                   markerfacecolor='grey',
                                   markersize=9, label='persistence')
    ma_handle = mlines.Line2D([], [], color='w', marker='s',
                              markeredgecolor='black',
                              markerfacecolor='grey',
                              markersize=9, label='moving_average')
    p1_handle = mlines.Line2D([], [], color='grey', ls=':',
                              label=f'P1 reference (£{p1_penalty:.2f})')
    ax.legend(handles=[persist_handle, ma_handle, p1_handle], loc='upper left')

    
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.matplotlib.colors.LogNorm(vmin=H_min, vmax=H_max)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Horizon H (log scale)")

    ax.set_xlabel("PV forecast MAE (kW, leads ≥ 1)")
    ax.set_ylabel("Cost penalty vs P3 (£)")
    ax.set_title("Forecast Accuracy vs Cost Penalty (colour = horizon H)")
    plt.tight_layout()
    plt.savefig("fig_p4_error_vs_cost.png", dpi=300)
    plt.close()

    # ---------- Console summary ----------
    gap = cost_p1 - cost_p3
    print("\n% of foresight gap closed (baseline = P1):")
    print(f"  Total gap P1 - P3 = £{gap:.2f}")
    print(f"  Best persistence  H={H_pers:>3d}   £{results_p4[('persistence', H_pers)]['adjusted_cost']:>6.2f}"
          f"   {(cost_p1 - results_p4[('persistence', H_pers)]['adjusted_cost']) / gap * 100:>6.1f}%")
    print(f"  Best MA           H={H_ma:>3d}   £{results_p4[('moving_average', H_ma)]['adjusted_cost']:>6.2f}"
          f"   {(cost_p1 - results_p4[('moving_average', H_ma)]['adjusted_cost']) / gap * 100:>6.1f}%")
    print("\nFull sweep:")
    for (m, H), res in sorted(results_p4.items()):
        closed = (cost_p1 - res['adjusted_cost']) / gap * 100 if gap != 0 else 0.0
        print(f"  {m:<16s}  H={H:>3d}   £{res['adjusted_cost']:>6.2f}   {closed:>6.1f}% of gap closed")


# Driver


def main(csv_path, quick=False):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            f"Pass --csv <path> or place the file next to this script."
        )

    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows = {len(df) / STEPS_PER_DAY:.1f} days\n")

    params = build_policy_params(df)

    cost_p1, cost_p2, cost_p3 = compute_reference_costs(df, params)
    print()

    verify_p4(df, params, cost_p3_reference=cost_p3)

    if quick:
        horizons = (4, 12, 24)
    else:
        horizons = (2, 4, 8, 12, 24, 48)

    print("\nRunning forecast sweep...")
    results_p4 = run_forecast_experiments(
        df, params,
        horizons=horizons,
        methods=('persistence', 'moving_average'),
    )

    plot_p4_results(results_p4, cost_p1, cost_p2, cost_p3)


if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="P4: Receding-horizon MPC with PV forecasting")
    parser.add_argument(
        "--csv",
        type=Path,
        default=BASE_DIR / "caseA_smart_home_30min_summer.csv",
        help="Path to the input CSV file (default: ./caseA_smart_home_30min_summer.csv)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller horizon sweep (faster; good for iteration)"
    )
    args = parser.parse_args()
    main(args.csv, quick=args.quick)