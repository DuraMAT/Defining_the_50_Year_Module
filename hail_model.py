# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:06:49 2026

@author: sayala
"""

import numpy as np
import pandas as pd


# -----------------------------
# Weibull reliability
# -----------------------------


def weibull_params_from_t50_t90(t50, t90):
    """
    Calculate Weibull shape (alpha) and scale (beta) parameters
    from t50 and t90.

    t50 = year when 50% of the original cohort has failed.
    t90 = year when 90% of the original cohort has failed.
    """
    cdf1, cdf2 = 0.50, 0.90

    alpha = (
        np.log(-np.log(1 - cdf1)) - np.log(-np.log(1 - cdf2))
    ) / (
        np.log(t50) - np.log(t90)
    )

    beta = t50 / ((-np.log(1 - cdf1)) ** (1 / alpha))

    return alpha, beta


def weibull_survival(t, alpha, beta):
    """
    Calculate Weibull survival probability at each age.

    Returns the fraction of the original cohort still surviving
    at time t from intrinsic reliability failures alone.
    """
    return np.exp(-((np.array(t) / beta) ** alpha))


def weibull_annual_failure_fraction(years, alpha, beta):
    """
    Calculate the intrinsic reliability failure fraction for each year.

    Converts Weibull survival into the fraction of modules that fail
    during each year, relative to the modules surviving at the start
    of that year.

    Returns:
        S_rel : Weibull survival fraction by year.
        f_rel : Annual failure fraction of the surviving cohort.
    """
    S_rel = weibull_survival(years, alpha, beta)

    f_rel = np.zeros_like(S_rel, dtype=float)

    for i in range(1, len(years)):
        f_rel[i] = 1.0 - S_rel[i] / S_rel[i-1]

    return S_rel, f_rel


# -----------------------------
# Hail occurrence / hazard
# -----------------------------


def get_return_period(location, hail_return):
    """
    Get the median hail return period for a location
    from the hail return-period input table.
    """
    row = hail_return[hail_return["Location"] == location]

    if row.empty:
        raise ValueError(f"No return-period data found for {location}")

    return float(row["return_period"].iloc[0])


def get_gamma_params(location, hail_dist):
    """
    Get Gamma shape and scale parameters for the hail-size
    distribution at a given location.
    """
    gamma = hail_dist[
        (hail_dist["Location"] == location) &
        (hail_dist["Distribution"] == "Gamma")
    ]

    shape = float(
        gamma.loc[gamma["Parameter"] == "Shape", "Estimate"].iloc[0]
    )

    scale = float(
        gamma.loc[gamma["Parameter"] == "Scale", "Estimate"].iloc[0]
    )

    return shape, scale


def get_coverage_fraction(location, hail_coverage):
    """
    Return the spatial hail coverage fraction for a location.

    Interpretation
    --------------
    This approximates the probability that the PV site lies inside the
    hail footprint, GIVEN that the county experiences the hail event.

    Example
    -------
    Austin coverage_fraction ≈ 0.203

    This means that, under the current simplified spatial assumption,
    approximately 20.3% of Travis County is covered by the representative
    hail footprint.

    Notes
    -----
    This is NOT the annual probability of hail.
    Annual hail occurrence is handled separately using the return period.
    """

    row = hail_coverage[hail_coverage["Location"] == location]

    if row.empty:
        raise ValueError(f"No hail coverage data found for {location}")

    return float(row["coverage_fraction"].iloc[0])


def get_annual_site_hail_probability(location, hail_return, hail_coverage):
    """
    Calculate the approximate annual probability that the PV site
    is directly affected by the modeled hail event.

    Step 1
    ------
    Convert the location's median return period into an annual
    county-level occurrence probability:

        P(county event) = 1 / return_period

    Step 2
    ------
    Apply the spatial coverage modifier:

        P(site hit | county event) = coverage_fraction

    Step 3
    ------
    Combine them:

        P(site hit) =
            P(county event) * P(site hit | county event)

    This is currently a simplified approximation using the median
    2-inch hail footprint.
    """

    rp = get_return_period(location, hail_return)
    coverage = get_coverage_fraction(location, hail_coverage)

    p_county_event = 1 / rp
    p_site_hit = p_county_event * coverage

    return p_site_hit


def hail_event(location, hail_return, hail_coverage, rng=None):
    """
    Randomly determine whether the PV site is hit by hail this year.

    Returns
    -------
    True
        The simulated site experiences the modeled hail event this year.

    False
        The simulated site is not affected by hail this year.

    Important
    ---------
    This function ONLY determines whether the site is hit.

    It does NOT determine:
        - hail size
        - broken-glass fraction
        - post-hail degradation

    Those are handled later in the simulation.
    """

    if rng is None:
        rng = np.random.default_rng()
        
    p_site_hit = get_annual_site_hail_probability(location, hail_return, hail_coverage)

    return np.random.rand() < p_site_hit


# Sample Hail-
def sample_hail(location, hail_dist, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    shape, scale = get_gamma_params(location, hail_dist)
    return np.random.gamma(shape=shape, scale=scale)


# -----------------------------
# Damage
# -----------------------------


def hail_bin_label(hail_size_in):
    if hail_size_in < 1:
        return "0_1"
    elif hail_size_in < 2:
        return "1_2"
    elif hail_size_in < 3:
        return "2_3"
    elif hail_size_in < 4:
        return "3_4"
    else:
        return "4_5"


def sample_from_empirical_cdf_arrays(x_vals, cdf_vals, rng=None):
    """
    Inverse-CDF sampler from an empirical CDF.
    Returns one sampled PLR increase value (%/year).
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform()
    return np.interp(u, cdf_vals, x_vals)


def sample_from_empirical_cdf(cdf_df, x_col, p_col, rng=None):
    """
    Samples a value from an empirical CDF using DataFrame columns for values and cumulative probabilities.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = cdf_df[x_col].to_numpy(dtype=float)
    p = cdf_df[p_col].to_numpy(dtype=float)

    order = np.argsort(p)
    p = p[order]
    x = x[order]

    u = rng.random()
    return np.interp(u, p, x)


def sample_broken_glass_fraction(hail_size_in, damage_cdfs, rng=None):
    label = hail_bin_label(hail_size_in)
    cdf_df = damage_cdfs[label]

    broken_percent = sample_from_empirical_cdf(
        cdf_df,
        x_col="broken_glass_percent",
        p_col="cumulative_probability",
        rng=rng
    )

    return broken_percent / 100.0


def sample_plr_increase(plr_cdf, rng=None):
    """
    Samples additional PLR after hail, in fraction/year.
    Input CSV is percent/year, so convert to fraction/year.
    """
    if rng is None:
        rng = np.random.default_rng()

    plr_percent = sample_from_empirical_cdf(
        plr_cdf,
        x_col="plr_percent_increase_peryear",
        p_col="cumulative_probability",
        rng=rng
    )

    return plr_percent / 100.0



# -----------------------------
# Simulation
# -----------------------------


def simulate_project(
    P0_MW, location, years, damage_cdfs, hail_return, hail_dist,
    hail_coverage, plr_cdf=None, t50=39, t90=42, seed=0,
    use_post_hail_plr=True, track_hail_buckets=True,
    apply_plr_once=True, debug=False,
):
    """
    Simulate project capacity loss from intrinsic reliability failures
    and hail damage.

    Parameters
    ----------
    use_post_hail_plr : bool
        If False:
            Reliability failures + broken-glass hail failures only.
            No additional post-hail PLR.

        If True:
            Hail survivors may receive an additional PLR penalty
            sampled from plr_cdf.

    track_hail_buckets : bool
        Only relevant when use_post_hail_plr=True.

        If True:
            Track undamaged and hail-damaged surviving capacity separately.
            Post-hail PLR applies only to hail-damaged survivors.

        If False:
            Apply post-hail PLR globally to all surviving project capacity.

    apply_plr_once : bool
        If True:
            A PLR penalty is assigned only once.

        If False:
            PLR penalties may accumulate after subsequent hail events.
    """

    rng = np.random.default_rng(seed)

    alpha, beta = weibull_params_from_t50_t90(t50, t90)
    S_rel_only, f_rel = weibull_annual_failure_fraction(years, alpha, beta)

    # ------------------------------------------------------------------
    # Capacity buckets
    # ------------------------------------------------------------------

    undamaged_MW = P0_MW
    hail_damaged_MW = 0.0

    # Post-hail degradation
    extra_plr_per_year = 0.0

    # Used for global-PLR mode
    global_performance_factor = 1.0

    # Used for bucketed-PLR mode
    hail_damaged_perf_factor = 1.0

    rows = []

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    for i, year in enumerate(years):

        # --------------------------------------------------------------
        # Initial year
        # --------------------------------------------------------------

        # Start-of-year physical capacity
        available_start = undamaged_MW + hail_damaged_MW

        if i == 0:
            rows.append({
                "year": year,
                "hail_event": False,
                "hail_size_in": 0.0,
                "f_rel": 0.0,
                "f_cat": 0.0,

                "plr_added_this_year": 0.0,
                "cumulative_extra_plr_per_year": 0.0,

                "undamaged_MW_end": undamaged_MW,
                "hail_damaged_MW_end": hail_damaged_MW,

                "available_capacity_MW_end": P0_MW,
                "effective_capacity_MW_end": P0_MW,

                "reliability_failures_MW": 0.0,
                "hail_failures_MW": 0.0,
                "total_failures_MW": 0.0,

                "S_total": 1.0,
                "available_capacity_MW_start": P0_MW,
            })
            continue

        # --------------------------------------------------------------
        # 0. Apply existing post-hail PLR
        # --------------------------------------------------------------

        if use_post_hail_plr:

            if track_hail_buckets:

                # Only previously hail-damaged survivors receive
                # the post-hail degradation penalty.
                hail_damaged_perf_factor *= 1.0 - extra_plr_per_year

            else:

                # Global PLR mode:
                # degradation applies to all surviving project capacity.
                global_performance_factor *= 1.0 - extra_plr_per_year

        # --------------------------------------------------------------
        # 1. Intrinsic Weibull reliability failures
        # --------------------------------------------------------------

        rel_fail_undamaged = undamaged_MW * f_rel[i]
        rel_fail_damaged = hail_damaged_MW * f_rel[i]
        rel_fail_MW = rel_fail_undamaged + rel_fail_damaged

        undamaged_MW -= rel_fail_undamaged
        hail_damaged_MW -= rel_fail_damaged

        # --------------------------------------------------------------
        # 2. Hail event
        # --------------------------------------------------------------

        hail_bool = False
        hail_size = 0.0
        f_cat = 0.0

        hail_fail_MW = 0.0
        plr_added = 0.0

        if hail_event(location, hail_return, hail_coverage, rng):

            hail_bool = True
            hail_size = sample_hail(location, hail_dist, rng=rng)

            f_cat = sample_broken_glass_fraction(
                hail_size, damage_cdfs, rng=rng
            )

            # Broken-glass failures affect all physically surviving MW
            hail_fail_undamaged = undamaged_MW * f_cat
            hail_fail_damaged = hail_damaged_MW * f_cat
            hail_fail_MW = hail_fail_undamaged + hail_fail_damaged

            undamaged_survivors_after_hail = undamaged_MW - hail_fail_undamaged
            damaged_survivors_after_hail = hail_damaged_MW - hail_fail_damaged

            # ----------------------------------------------------------
            # 3. Post-hail PLR
            # ----------------------------------------------------------

            if use_post_hail_plr and plr_cdf is not None:

                sampled_plr = sample_plr_increase(plr_cdf, rng=rng)

                if debug:
                    print("PLR branch entered at year", year)
                    print("use_post_hail_plr =", use_post_hail_plr)
                    print("plr_cdf is None =", plr_cdf is None)
                    print("sampled_plr =", sampled_plr)

                # ------------------------------------------------------
                # A. TRACK HAIL-DAMAGED CAPACITY SEPARATELY
                # ------------------------------------------------------

                if track_hail_buckets:

                    # Capacity experiencing hail and surviving it
                    # becomes hail-damaged capacity.
                    newly_damaged_MW = undamaged_survivors_after_hail

                    undamaged_MW = 0.0
                    hail_damaged_MW = (
                        damaged_survivors_after_hail + newly_damaged_MW
                    )

                    if apply_plr_once:

                        # Add a PLR penalty only if one has not
                        # previously been assigned.
                        if extra_plr_per_year == 0.0:
                            plr_added = sampled_plr
                            extra_plr_per_year = sampled_plr
                        else:
                            plr_added = 0.0

                    else:

                        # Allow penalties from multiple hail events
                        # to accumulate.
                        plr_added = sampled_plr
                        extra_plr_per_year += sampled_plr

                # ------------------------------------------------------
                # B. GLOBAL POST-HAIL PLR
                # ------------------------------------------------------

                else:

                    # No need to distinguish hail-damaged survivors
                    # for performance purposes.
                    undamaged_MW = (
                        undamaged_survivors_after_hail
                        + damaged_survivors_after_hail
                    )

                    hail_damaged_MW = 0.0

                    if apply_plr_once:
                        if extra_plr_per_year == 0.0:
                            plr_added = sampled_plr
                            extra_plr_per_year = sampled_plr
                        else:
                            plr_added = 0.0
                    else:
                        plr_added = sampled_plr
                        extra_plr_per_year += sampled_plr

            # ----------------------------------------------------------
            # No post-hail PLR
            # ----------------------------------------------------------

            else:

                undamaged_MW = (
                    undamaged_survivors_after_hail
                    + damaged_survivors_after_hail
                )

                hail_damaged_MW = 0.0

        # --------------------------------------------------------------
        # End-of-year capacities
        # --------------------------------------------------------------

        total_fail_MW = rel_fail_MW + hail_fail_MW
        available_end = undamaged_MW + hail_damaged_MW

        # --------------------------------------------------------------
        # Effective capacity
        # --------------------------------------------------------------

        if not use_post_hail_plr:

            # Physical surviving capacity only
            effective_capacity_MW = available_end

        elif track_hail_buckets:

            # Undamaged MW remain at full performance.
            # Hail-damaged MW receive the post-hail penalty.
            effective_capacity_MW = (
                undamaged_MW
                + hail_damaged_MW * hail_damaged_perf_factor
            )

        else:

            # Post-hail degradation applies globally.
            effective_capacity_MW = available_end * global_performance_factor

        # --------------------------------------------------------------
        # Save results
        # --------------------------------------------------------------

        rows.append({
            "year": year,

            "hail_event": hail_bool,
            "hail_size_in": hail_size,

            "f_rel": f_rel[i],
            "f_cat": f_cat,

            "plr_added_this_year": plr_added,
            "cumulative_extra_plr_per_year": extra_plr_per_year,

            "undamaged_MW_end": undamaged_MW,
            "hail_damaged_MW_end": hail_damaged_MW,

            "available_capacity_MW_end": available_end,
            "effective_capacity_MW_end": effective_capacity_MW,

            "reliability_failures_MW": rel_fail_MW,
            "hail_failures_MW": hail_fail_MW,
            "total_failures_MW": total_fail_MW,

            "S_total": available_end / P0_MW,
            "available_capacity_MW_start": available_start,
        })

    return pd.DataFrame(rows)


def run_monte_carlo(
    n_sims, P0_MW, location, years, damage_cdfs, hail_return, hail_dist,
    hail_coverage, plr_cdf=None, t50=39, t90=42, base_seed=0,
    use_post_hail_plr=True, track_hail_buckets=True, apply_plr_once=True,):
    sims = []

    for s in range(n_sims):
        res = simulate_project(
            P0_MW=P0_MW,
            location=location,
            years=years,
            damage_cdfs=damage_cdfs,
            hail_return=hail_return,
            hail_dist=hail_dist,
            hail_coverage=hail_coverage,
            plr_cdf=plr_cdf,
            t50=t50,
            t90=t90,
            seed=base_seed + s,
            use_post_hail_plr=use_post_hail_plr,
            track_hail_buckets=track_hail_buckets,
            apply_plr_once=apply_plr_once,
        )
        res["sim"] = s
        sims.append(res)

    return pd.concat(sims, ignore_index=True)

