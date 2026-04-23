import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# LOAD DATA


def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "pv_kw": "pv",
        "base_load_kw": "load",
        "import_tariff_gbp_per_kwh": "tariff_import",
        "export_price_gbp_per_kwh": "tariff_export"
    })

    return df


# PARAMETERS


def default_parameters():
    return {
        "capacity_kwh": 5.0,
        "max_charge_kw": 2.5,
        "max_discharge_kw": 2.5,
        "eta_ch": 0.95,
        "eta_dis": 0.95,
        "soc_init": 2.5,
        "soc_min_reserve": 0.5,
        "dt_hours": 0.5,
        "target_soc_low_price": 3.5,
    }


def build_policy_params(df, base_params=None):
    """Return a params dict with tariff quantiles filled in for P2."""
    params = dict(base_params) if base_params is not None else default_parameters()
    params.setdefault("high_price", float(df["tariff_import"].quantile(0.75)))
    params.setdefault("low_price", float(df["tariff_import"].quantile(0.25)))
    return params


# POLICIES (P1 & P2)


def policy_self_consumption(t, soc, pv, load, tariff_import, tariff_export, params):
    surplus = pv - load
    if surplus > 0:
        return {"p_charge_request": surplus, "p_discharge_request": 0.0}
    else:
        return {"p_charge_request": 0.0, "p_discharge_request": -surplus}


def policy_tariff_aware(t, soc, pv, load, tariff_import, tariff_export, params):
    surplus = pv - load
    p_charge_request = 0.0
    p_discharge_request = 0.0

    # Fall back gracefully if quantile thresholds weren't pre-computed
    high_price = params.get("high_price", float("inf"))
    low_price = params.get("low_price", float("-inf"))
    reserve = params["soc_min_reserve"]
    target_soc_low_price = params["target_soc_low_price"]

    if surplus > 0:
        p_charge_request = surplus
    else:
        deficit = -surplus
        if tariff_import >= high_price and soc > reserve:
            p_discharge_request = deficit
        elif tariff_import <= low_price and soc < target_soc_low_price:
            p_charge_request = params["max_charge_kw"]

    return {"p_charge_request": p_charge_request,
            "p_discharge_request": p_discharge_request}


# SIMULATOR


def run_simulation(df, params, policy_function):
    n = len(df)
    dt = params["dt_hours"]
    eta_ch = params["eta_ch"]
    eta_dis = params["eta_dis"]
    cap = params["capacity_kwh"]

    # Pull columns into numpy arrays once
    pv_arr = df["pv"].to_numpy(dtype=float)
    load_arr = df["load"].to_numpy(dtype=float)
    tar_imp_arr = df["tariff_import"].to_numpy(dtype=float)
    tar_exp_arr = df["tariff_export"].to_numpy(dtype=float)

    soc = np.zeros(n + 1)
    soc[0] = params["soc_init"]

    p_charge = np.zeros(n)
    p_discharge = np.zeros(n)
    p_import = np.zeros(n)
    p_export = np.zeros(n)

    energy_balance_residual = np.zeros(n)
    soc_equation_residual = np.zeros(n)

    for t in range(n):
        pv = pv_arr[t]
        load = load_arr[t]
        tariff_import = tar_imp_arr[t]
        tariff_export = tar_exp_arr[t]

        decision = policy_function(
            t, soc[t], pv, load, tariff_import, tariff_export, params
        )

        p_ch_req = float(decision["p_charge_request"])
        p_dis_req = float(decision["p_discharge_request"])

        if p_ch_req > 0 and p_dis_req > 0:
            raise ValueError(f"Policy requested simultaneous charge/discharge at t={t}")
        
        reserve = params["soc_min_reserve"]
        max_charge_from_space = max(0.0, (cap - soc[t]) / (eta_ch * dt))
        max_discharge_from_soc = max(0.0, (soc[t] - reserve) * eta_dis / dt)


        p_ch = min(max(p_ch_req, 0.0), params["max_charge_kw"], max_charge_from_space)
        p_dis = min(max(p_dis_req, 0.0), params["max_discharge_kw"], max_discharge_from_soc)

        if p_ch > 0 and p_dis > 0:
            raise ValueError(f"Simultaneous charge/discharge after limiting at t={t}")

        net_grid = load + p_ch - pv - p_dis
        if net_grid >= 0:
            p_imp, p_exp = net_grid, 0.0
        else:
            p_imp, p_exp = 0.0, -net_grid

        # Unclipped SOC from the dynamics equation
        soc_next_raw = soc[t] + eta_ch * p_ch * dt - (p_dis * dt / eta_dis)
        # Clip to physical bounds
        soc_next = min(max(soc_next_raw, 0.0), cap)
        # Residual captures any clipping (should be ~0 if pre-limits are correct)
        soc_equation_residual[t] = soc_next - soc_next_raw
        soc[t + 1] = soc_next

        p_charge[t] = p_ch
        p_discharge[t] = p_dis
        p_import[t] = p_imp
        p_export[t] = p_exp

        energy_balance_residual[t] = pv + p_imp + p_dis - load - p_exp - p_ch

    return _build_results_dict(
        df, params, soc, p_charge, p_discharge, p_import, p_export,
        energy_balance_residual, soc_equation_residual
    )


# P3: OPTIMISATION


def run_optimisation(df, params):
    n = len(df)
    dt = params["dt_hours"]
    cap = params["capacity_kwh"]
    p_ch_max = params["max_charge_kw"]
    p_dis_max = params["max_discharge_kw"]
    eta_ch = params["eta_ch"]
    eta_dis = params["eta_dis"]
    soc_init = params["soc_init"]
    reserve = params["soc_min_reserve"]

    pv = df["pv"].to_numpy()
    load = df["load"].to_numpy()
    pi_imp = df["tariff_import"].to_numpy()
    pi_exp = df["tariff_export"].to_numpy()

    p_ch_var = cp.Variable(n, nonneg=True)
    p_dis_var = cp.Variable(n, nonneg=True)
    p_imp_var = cp.Variable(n, nonneg=True)
    p_exp_var = cp.Variable(n, nonneg=True)
    soc_var = cp.Variable(n + 1)

    # simultaneous charge/discharge is not explicitly forbidden here.
    # With positive tariffs and eta < 1 it is strictly suboptimal, so the LP
    # will not do it; the verification step checks this holds in practice.
    constraints = [
        soc_var[0] == soc_init,
        soc_var[n] == soc_init,          
        soc_var >= reserve,           
        soc_var <= cap,
        p_ch_var <= p_ch_max,
        p_dis_var <= p_dis_max,
    ]

    for t in range(n):
        constraints += [
            pv[t] + p_imp_var[t] + p_dis_var[t]
                == load[t] + p_exp_var[t] + p_ch_var[t],
            soc_var[t + 1]
                == soc_var[t] + eta_ch * p_ch_var[t] * dt - (p_dis_var[t] * dt / eta_dis),
        ]

    cost_expr = cp.sum(cp.multiply(pi_imp, p_imp_var)
                       - cp.multiply(pi_exp, p_exp_var)) * dt
    problem = cp.Problem(cp.Minimize(cost_expr), constraints)
    problem.solve()

    print(f"LP solver status: {problem.status}")

    p_charge = p_ch_var.value
    p_discharge = p_dis_var.value
    p_import = p_imp_var.value
    p_export = p_exp_var.value
    soc = soc_var.value

    energy_balance_residual = (
        pv + p_import + p_discharge - load - p_export - p_charge
    )
    soc_equation_residual = (
        soc[1:] - (soc[:-1] + eta_ch * p_charge * dt - (p_discharge * dt / eta_dis))
    )

    return _build_results_dict(
        df, params, soc, p_charge, p_discharge, p_import, p_export,
        energy_balance_residual, soc_equation_residual
    )


# SHARED RESULTS BUILDER


def _build_results_dict(df, params, soc, p_charge, p_discharge, p_import, p_export,
                        energy_balance_residual, soc_equation_residual):
    dt = params["dt_hours"]
    eta_ch = params["eta_ch"]
    eta_dis = params["eta_dis"]
    cap = params["capacity_kwh"]

    total_import_cost = np.sum(p_import * df["tariff_import"].to_numpy() * dt)
    total_export_revenue = np.sum(p_export * df["tariff_export"].to_numpy() * dt)
    total_cost_raw = total_import_cost - total_export_revenue

    soc_difference = soc[-1] - soc[0]
    avg_import_tariff = float(df["tariff_import"].mean())
    terminal_soc_adjustment = -soc_difference * avg_import_tariff
    total_cost_adjusted = total_cost_raw + terminal_soc_adjustment

    total_import_energy = np.sum(p_import) * dt
    total_export_energy = np.sum(p_export) * dt
    total_charge_energy = np.sum(p_charge) * dt
    total_discharge_energy = np.sum(p_discharge) * dt
    total_pv_energy = np.sum(df["pv"]) * dt
    total_load_energy = np.sum(df["load"]) * dt

    lhs = soc[-1] - soc[0]
    rhs = eta_ch * total_charge_energy - (total_discharge_energy / eta_dis)
    battery_state_residual = lhs - rhs

    pv_self_consumption_ratio = (
        1.0 - (total_export_energy / total_pv_energy) if total_pv_energy > 0 else np.nan
    )
    grid_dependency_ratio = (
        total_import_energy / total_load_energy if total_load_energy > 0 else np.nan
    )
    battery_round_trip_ratio = (
        total_discharge_energy / total_charge_energy if total_charge_energy > 0 else np.nan
    )

    verification = {
        "max_energy_balance_residual": float(np.max(np.abs(energy_balance_residual))),
        "max_soc_equation_residual": float(np.max(np.abs(soc_equation_residual))),
        "soc_min": float(np.min(soc)),
        "soc_max": float(np.max(soc)),
        "end_soc_difference": float(soc_difference),
        "terminal_soc_adjustment_gbp": float(terminal_soc_adjustment),
        "simultaneous_charge_discharge_count": int(np.sum((p_charge > 1e-6) & (p_discharge > 1e-6))),
        "simultaneous_import_export_count": int(np.sum((p_import > 1e-6) & (p_export > 1e-6))),
        "battery_state_residual_kwh": float(battery_state_residual),
        "charge_limit_violations": int(np.sum(p_charge > params["max_charge_kw"] + 1e-6)),
        "discharge_limit_violations": int(np.sum(p_discharge > params["max_discharge_kw"] + 1e-6)),
        "soc_lower_bound_violations": int(np.sum(soc < -1e-6)),
        "soc_upper_bound_violations": int(np.sum(soc > cap + 1e-6))
    }

    return {
        "total_cost_raw": float(total_cost_raw),
        "total_cost_adjusted": float(total_cost_adjusted),
        "soc": soc,
        "p_charge": p_charge,
        "p_discharge": p_discharge,
        "p_import": p_import,
        "p_export": p_export,
        "verification": verification,
        "energy_summary": {
            "pv_kwh": float(total_pv_energy),
            "load_kwh": float(total_load_energy),
            "import_kwh": float(total_import_energy),
            "export_kwh": float(total_export_energy),
            "battery_charge_kwh": float(total_charge_energy),
            "battery_discharge_kwh": float(total_discharge_energy)
        },
        "kpi_summary": {
            "pv_self_consumption_ratio": float(pv_self_consumption_ratio),
            "grid_dependency_ratio": float(grid_dependency_ratio),
            "battery_round_trip_ratio": float(battery_round_trip_ratio)
        }
    }


# PLOTTING FUNCTIONS


def save_table_as_image(df_table, title, filename, fig_width=12, fig_height=2.5, font_size=8):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# F1: Combined input plot
def plot_inputs(df, start_idx, end_idx, filename, show=False):
    hours = np.arange(end_idx - start_idx) * 0.5
    pv = df["pv"].iloc[start_idx:end_idx].to_numpy()
    load = df["load"].iloc[start_idx:end_idx].to_numpy()
    tar_i = df["tariff_import"].iloc[start_idx:end_idx].to_numpy()
    tar_e = df["tariff_export"].iloc[start_idx:end_idx].to_numpy()

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(hours, pv, label="PV (kW)", color="tab:orange")
    ax1.plot(hours, load, label="Load (kW)", color="tab:blue")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Power (kW)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(hours, tar_i, "--", color="tab:red", label="Import Tariff (£/kWh)")
    ax2.plot(hours, tar_e, ":", color="tab:green", label="Export Price (£/kWh)")
    ax2.set_ylabel("Price (£/kWh)")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=8)
    plt.title("Case Inputs: PV, Load, and Tariffs (First Day)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close()


# F2: SOC comparison across policies (stacked panels + tariff overlay)
def plot_soc_comparison(all_results, df, params, start_idx, end_idx, filename,
                        title_suffix="First Day", show=False):
    hours = np.arange(end_idx - start_idx + 1) * 0.5
    tariff = df["tariff_import"].iloc[start_idx:end_idx].to_numpy()
    tariff_hours = np.arange(end_idx - start_idx) * 0.5
    cap = params["capacity_kwh"]

    fig, axes = plt.subplots(len(all_results), 1, figsize=(10, 2.3 * len(all_results)),
                             sharex=True)
    if len(all_results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        soc = res["soc"][start_idx:end_idx + 1]
        ax.plot(hours, soc, "k-", linewidth=1.8, label="SOC (kWh)")
        ax.fill_between(hours, 0, soc, alpha=0.1, color="black")
        ax.set_ylabel("SOC (kWh)")
        ax.set_ylim(0, cap * 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(name, loc="left", fontsize=10)

        ax2 = ax.twinx()
        ax2.plot(tariff_hours, tariff, color="tab:red", alpha=0.6,
                 linewidth=1, label="Import Tariff (£/kWh)")
        ax2.set_ylabel("Tariff (£/kWh)", color="tab:red")
        ax2.tick_params(axis='y', labelcolor="tab:red")

    axes[-1].set_xlabel("Time (hours)")
    plt.suptitle(f"Battery SOC Across Policies ({title_suffix})", y=1.00)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# F3: Grid flows comparison 
def plot_grid_comparison(all_results, start_idx, end_idx, filename, show=False):
    hours = np.arange(end_idx - start_idx) * 0.5
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for name, res in all_results.items():
        ax1.plot(hours, res["p_import"][start_idx:end_idx], label=name)
        ax2.plot(hours, res["p_export"][start_idx:end_idx], label=name)

    ax1.set_ylabel("Grid Import (kW)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.set_ylabel("Grid Export (kW)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Time (hours)")

    plt.suptitle("Grid Flows by Policy (First Day)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close()


# F4: Cost comparison with optimality gap
def plot_cost_with_gap(all_results, filename, show=False):
    names = list(all_results.keys())
    costs = [r["total_cost_adjusted"] for r in all_results.values()]
    opt = min(costs)

    colors = ["tab:gray", "tab:orange", "tab:green"][:len(names)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, costs, color=colors)
    ax.set_ylabel("Terminal-SOC-Adjusted Cost (£)", fontsize=11)
    ax.set_title("Policy Cost Comparison", fontsize=12)
    ax.axhline(opt, linestyle="--", color="tab:green", alpha=0.6,
               label=f"Optimal = £{opt:.2f}")
    ax.legend(loc="upper right", fontsize=10)

    for bar, cost in zip(bars, costs):
        gap_pct = (cost - opt) / opt * 100 if opt != 0 else 0
        label = f"£{cost:.2f}" if abs(cost - opt) < 1e-6 else f"£{cost:.2f}\n(+{gap_pct:.1f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() - bar.get_height() * 0.08,
                label, ha="center", va="top",
                color="white", fontweight="bold", fontsize=11)

    ax.set_ylim(0, max(costs) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

# F5: Energy flow grouped bars
def plot_energy_flow(all_results, filename, show=False):
    labels = ["Import", "Export", "Battery Charge", "Battery Discharge"]
    keys = ["import_kwh", "export_kwh", "battery_charge_kwh", "battery_discharge_kwh"]

    x = np.arange(len(labels))
    n = len(all_results)
    width = 0.8 / n

    plt.figure(figsize=(9, 5))
    for i, (name, res) in enumerate(all_results.items()):
        vals = [res["energy_summary"][k] for k in keys]
        plt.bar(x + (i - (n - 1) / 2) * width, vals, width=width, label=name)

    plt.xticks(x, labels)
    plt.ylabel("Energy (kWh)")
    plt.title("Energy Flow Comparison Across Policies")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close()


# F6: tariff vs battery actions, stacked across policies
def plot_tariff_vs_actions_stacked(df, all_results, start_idx, end_idx, filename, show=False):
    hours = np.arange(end_idx - start_idx) * 0.5
    tariff = df["tariff_import"].iloc[start_idx:end_idx].to_numpy()

    fig, axes = plt.subplots(len(all_results), 1, figsize=(10, 2.5 * len(all_results)),
                             sharex=True)
    if len(all_results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        p_ch = res["p_charge"][start_idx:end_idx]
        p_dis = res["p_discharge"][start_idx:end_idx]

        ax.plot(hours, p_ch, color="tab:green", label="Charge (kW)")
        ax.plot(hours, p_dis, color="tab:purple", label="Discharge (kW)")
        ax.set_ylabel("Power (kW)")
        ax.grid(True, alpha=0.3)
        ax.set_title(name, loc="left", fontsize=10)

        ax2 = ax.twinx()
        ax2.plot(hours, tariff, color="tab:red", alpha=0.6, linewidth=1,
                 label="Import Tariff (£/kWh)")
        ax2.set_ylabel("Tariff (£/kWh)", color="tab:red")
        ax2.tick_params(axis='y', labelcolor="tab:red")

        if ax is axes[0]:
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time (hours)")
    plt.suptitle("Tariff vs Battery Actions Across Policies", y=1.00)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# MAIN


def main(csv_path, show_plots=False):
    df = load_data(csv_path)

    params = build_policy_params(df)

    print("Running P1: Self-Consumption Policy...")
    results_self = run_simulation(df, params, policy_self_consumption)
    print(f"  Adjusted Cost (£): {results_self['total_cost_adjusted']:.4f}")

    print("\nRunning P2: Tariff-Aware Policy...")
    results_tariff = run_simulation(df, params, policy_tariff_aware)
    print(f"  Adjusted Cost (£): {results_tariff['total_cost_adjusted']:.4f}")

    print("\nRunning P3: LP Optimisation Policy...")
    results_opt = run_optimisation(df, params)
    print(f"  Adjusted Cost (£): {results_opt['total_cost_adjusted']:.4f}")

    all_results = {
        "P1: Self-Consumption":  results_self,
        "P2: Tariff-Aware":      results_tariff,
        "P3: Optimisation (LP)": results_opt,
    }

    # TABLES
    def _row(res):
        return [
            res["total_cost_raw"],
            res["total_cost_adjusted"],
            res["energy_summary"]["import_kwh"],
            res["energy_summary"]["export_kwh"],
            res["energy_summary"]["battery_charge_kwh"],
            res["energy_summary"]["battery_discharge_kwh"],
            res["verification"]["end_soc_difference"],
            res["kpi_summary"]["pv_self_consumption_ratio"],
            res["kpi_summary"]["grid_dependency_ratio"],
        ]

    summary_table = pd.DataFrame(
        {name: _row(res) for name, res in all_results.items()},
        index=[
            "Raw Cost (£)", "Adjusted Cost (£)",
            "Import (kWh)", "Export (kWh)",
            "Battery Charge (kWh)", "Battery Discharge (kWh)",
            "End SOC Change (kWh)",
            "PV Self-Consumption Ratio", "Grid Dependency Ratio"
        ]
    ).T.reset_index().rename(columns={"index": "Policy"})

    def _ver_row(res):
        v = res["verification"]
        return [
            v["max_energy_balance_residual"],
            v["max_soc_equation_residual"],
            v["soc_min"],
            v["soc_max"],
            v["end_soc_difference"],
            v["terminal_soc_adjustment_gbp"],
            v["simultaneous_charge_discharge_count"],
            v["simultaneous_import_export_count"],
            v["battery_state_residual_kwh"],
            v["charge_limit_violations"],
            v["discharge_limit_violations"],
            v["soc_lower_bound_violations"],
            v["soc_upper_bound_violations"],
        ]

    verification_table = pd.DataFrame({
        "Check": [
            "Max energy balance residual",
            "Max SOC equation residual",
            "SOC minimum",
            "SOC maximum",
            "End SOC difference (kWh)",
            "Terminal SOC adjustment (£)",
            "Simultaneous charge/discharge count",
            "Simultaneous import/export count",
            "Battery state residual (kWh)",
            "Charge limit violations",
            "Discharge limit violations",
            "SOC lower bound violations",
            "SOC upper bound violations"
        ],
        **{name: _ver_row(res) for name, res in all_results.items()}
    })

    print("\n=== Policy Summary Table ===")
    print(summary_table.round(4))
    print("\n=== Verification Table ===")
    print(verification_table.round(8))

    save_table_as_image(summary_table.round(3), "Policy Summary Table",
                        "summary_table.png", fig_width=15, fig_height=2.6, font_size=8)
    save_table_as_image(verification_table.round(6), "Verification Table",
                        "verification_table.png", fig_width=13, fig_height=4.5, font_size=8)

    summary_table.to_csv("summary_table.csv", index=False)
    verification_table.to_csv("verification_table.csv", index=False)
    with pd.ExcelWriter("results_tables.xlsx") as writer:
        summary_table.to_excel(writer, sheet_name="Summary", index=False)
        verification_table.to_excel(writer, sheet_name="Verification", index=False)

    # FIGURES
    start_idx = 0
    end_idx = 48   # first 24 hours at 30-min resolution

    plot_inputs(df, start_idx, end_idx, "fig1_inputs.png", show=show_plots)
    plot_soc_comparison(all_results, df, params, start_idx, end_idx,
                        "fig2_soc_comparison.png", show=show_plots)
    plot_grid_comparison(all_results, start_idx, end_idx,
                         "fig3_grid_comparison.png", show=show_plots)
    plot_cost_with_gap(all_results, "fig4_cost_comparison.png", show=show_plots)
    plot_energy_flow(all_results, "fig5_energy_flow.png", show=show_plots)

    plot_tariff_vs_actions_stacked(
        df, all_results, start_idx, end_idx,
        "figA1_tariff_vs_actions_stacked.png", show=show_plots
    )

    # Full-horizon SOC comparison 
    plot_soc_comparison(
        all_results, df, params, 0, len(df),
        "figA2_soc_full_horizon.png",
        title_suffix="Full Horizon", show=show_plots
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart-home battery policy simulator")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("caseA_smart_home_30min_summer.csv"),
        help="Path to the input CSV file (default: ./caseA_smart_home_30min_summer.csv)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them"
    )
    args = parser.parse_args()
    main(args.csv, show_plots=args.show)