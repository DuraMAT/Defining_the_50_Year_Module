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
# Hail hazard input helpers
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


# -----------------------------
# Generic empirical CDF sampling
# -----------------------------

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


# -----------------------------
# Post-hail degradation
# -----------------------------

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
# Fragility-curve damage model
# -----------------------------

def fragility_damage_percent(hail_size, A, B, C, D, E):
    """
    Calculate visible system damage [%] from hail size using the
    five-parameter fragility curve:

        f(x) = D + (A - D) / (1 + (x / C)**B)**E

    where x is hail size.
    """
    damage_percent = D + (A - D) / (1 + (hail_size / C)**B)**E

    # Keep damage physically bounded between 0 and 100%.
    return np.clip(damage_percent, 0.0, 100.0)


def get_fragility_params(module_area_m2, fragility_curves):
    """
    Select fragility-curve parameters based on module area [m²].

    Bins:
        < 1.5 m²
        1.5–2 m²
        >= 2 m²
    """

    if module_area_m2 < 1.5:
        area_bin = "<1.5"
    elif module_area_m2 < 2.0:
        area_bin = "1.5-2"
    else:
        area_bin = "2+"

    row = fragility_curves[
        fragility_curves["module_area_bin"] == area_bin
    ]

    if row.empty:
        raise ValueError(f"No fragility parameters found for {area_bin}")

    row = row.iloc[0]

    return (
        float(row["A"]),
        float(row["B"]),
        float(row["C"]),
        float(row["D"]),
        float(row["E"])
    )


def damage_fraction_from_fragility(
    hail_size, module_area_m2, fragility_curves
):
    """
    Calculate the fraction of system capacity with visible damage
    from hail size and module area.
    """

    # 1. Select the fragility curve based on module area
    A, B, C, D, E = get_fragility_params(
        module_area_m2, fragility_curves)

    # 2. Evaluate visible system damage at the sampled hail size
    damage_percent = fragility_damage_percent(
        hail_size, A, B, C, D, E)

    # 3. Convert percent damage to fraction for the capacity model
    return damage_percent / 100.0


# -----------------------------
# Spatial hail hazard
# -----------------------------

def county_hail_event(location, hail_return, rng=None):
    """
    Determine whether a hail event occurs somewhere in the county.
    Uses only the location-specific return period.
    """
    if rng is None:
        rng = np.random.default_rng()

    rp = get_return_period(location, hail_return)
    p_county = 1.0 / rp

    return rng.random() < p_county


def sample_hail_diameter(location, hail_dist, rng=None):
    """
    Sample hail diameter from the location-specific Gamma distribution
    using the supplied random-number generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    shape, scale = get_gamma_params(location, hail_dist)

    return rng.gamma(shape=shape, scale=scale)


def footprint_bin_label(hail_size_in):
    if hail_size_in < 1:
        return "0_1"
    elif hail_size_in < 2:
        return "1_2"
    elif hail_size_in < 3:
        return "2_3"
    else:
        return "3_4"


def sample_hail_footprint_area(hail_size_in, footprint_cdfs, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    label = footprint_bin_label(hail_size_in)
    cdf_df = footprint_cdfs[label]

    return sample_from_empirical_cdf(
        cdf_df,
        x_col="area_sqmi",
        p_col="cumulative_probability",
        rng=rng
    )


def site_hit_from_footprint(location, footprint_area_sqmi, county_areas, rng=None):
    """
    Determine whether the site lies within the sampled hail footprint.
    """
    if rng is None:
        rng = np.random.default_rng()

    row = county_areas[county_areas["Location"] == location]

    if row.empty:
        raise ValueError(f"No county-area data found for {location}")

    county_area = float(row["A_total_sqmi"].iloc[0])

    coverage = min(footprint_area_sqmi / county_area, 1.0)

    site_hit = rng.random() < coverage

    return site_hit, coverage


def sample_spatial_hail_event(
    location, hail_return, hail_dist, footprint_cdfs, county_areas, rng=None
):
    """
    New hail hazard sequence:
    county occurrence -> hail diameter -> footprint area
    -> spatial coverage -> site hit
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Does a county-level hail event occur?
    county_event = county_hail_event(location, hail_return, rng=rng)

    if not county_event:
        return {
            "county_event": False,
            "hail_size_in": 0.0,
            "footprint_area_sqmi": 0.0,
            "coverage_fraction": 0.0,
            "site_hit": False
        }

    # 2. Sample hail diameter for the county event
    hail_size = sample_hail_diameter(location, hail_dist, rng=rng)

    # 3. Sample hail footprint area conditional on hail-size bin
    footprint_area = sample_hail_footprint_area(
        hail_size, footprint_cdfs, rng=rng
    )

    # 4. Convert footprint area to spatial coverage and determine site hit
    site_hit, coverage = site_hit_from_footprint(
        location, footprint_area, county_areas, rng=rng
    )

    # 5. Return the full hazard realization for debugging / simulation use
    return {
        "county_event": True,
        "hail_size_in": hail_size,
        "footprint_area_sqmi": footprint_area,
        "coverage_fraction": coverage,
        "site_hit": site_hit
    }




# -----------------------------
# Simulation
# -----------------------------

def simulate_project_spatial(
    P0_MW, location, years, hail_return, hail_dist, footprint_cdfs,
    county_areas, fragility_curves, module_area_m2, plr_cdf=None,
    t50=39, t90=42, seed=0, use_post_hail_plr=True,
    track_hail_buckets=True, apply_plr_once=True, debug=False,
):
    """
    Simulate project capacity loss using the new spatial hail pipeline
    and module-area-dependent fragility curves.

    Hail sequence
    -------------
    county hail occurrence
        -> sample hail diameter
        -> sample hail footprint area conditional on hail size
        -> calculate spatial coverage
        -> determine whether site is hit
        -> calculate visible damage from fragility curve
        -> optionally apply post-hail PLR
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

        # Start-of-year physical capacity
        available_start = undamaged_MW + hail_damaged_MW

        # --------------------------------------------------------------
        # Initial year
        # --------------------------------------------------------------

        if i == 0:
            rows.append({
                "year": year,

                # New spatial-hazard outputs
                "county_hail_event": False,
                "site_hit": False,
                "hail_size_in": 0.0,
                "footprint_area_sqmi": 0.0,
                "coverage_fraction": 0.0,

                # Keep hail_event for compatibility with old analysis
                "hail_event": False,

                # Damage
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
                hail_damaged_perf_factor *= 1.0 - extra_plr_per_year
            else:
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
        # 2. NEW SPATIAL HAIL EVENT
        # --------------------------------------------------------------

        county_event = False
        site_hit = False
        hail_size = 0.0
        footprint_area = 0.0
        coverage = 0.0

        # Visible system damage fraction from the module-area-dependent fragility curve
        f_cat = 0.0

        hail_fail_MW = 0.0
        plr_added = 0.0

        spatial_event = sample_spatial_hail_event(
            location=location,
            hail_return=hail_return,
            hail_dist=hail_dist,
            footprint_cdfs=footprint_cdfs,
            county_areas=county_areas,
            rng=rng
        )

        county_event = spatial_event["county_event"]
        site_hit = spatial_event["site_hit"]
        hail_size = spatial_event["hail_size_in"]
        footprint_area = spatial_event["footprint_area_sqmi"]
        coverage = spatial_event["coverage_fraction"]

        # --------------------------------------------------------------
        # 3. Apply fragility damage only if site is inside hail footprint
        # --------------------------------------------------------------

        if site_hit:

            # Visible system damage fraction from hail size and module area
            f_cat = damage_fraction_from_fragility(
                hail_size, module_area_m2, fragility_curves
            )

            # Fragility damage affects all physically surviving MW
            hail_fail_undamaged = undamaged_MW * f_cat
            hail_fail_damaged = hail_damaged_MW * f_cat
            hail_fail_MW = hail_fail_undamaged + hail_fail_damaged

            undamaged_survivors_after_hail = undamaged_MW - hail_fail_undamaged
            damaged_survivors_after_hail = hail_damaged_MW - hail_fail_damaged

            # ----------------------------------------------------------
            # 4. Post-hail PLR
            # ----------------------------------------------------------

            if use_post_hail_plr and plr_cdf is not None:

                sampled_plr = sample_plr_increase(plr_cdf, rng=rng)

                if debug:
                    print(
                        "Year:", year,
                        "| county event:", county_event,
                        "| hail size:", hail_size,
                        "| footprint:", footprint_area,
                        "| coverage:", coverage,
                        "| site hit:", site_hit,
                        "| fragility damage:", f_cat,
                        "| sampled PLR:", sampled_plr
                    )

                # ------------------------------------------------------
                # A. Track hail-damaged capacity separately
                # ------------------------------------------------------

                if track_hail_buckets:

                    # Capacity exposed to hail and surviving becomes
                    # hail-damaged capacity.
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
                # B. Global post-hail PLR
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
            effective_capacity_MW = available_end

        elif track_hail_buckets:
            effective_capacity_MW = (
                undamaged_MW + hail_damaged_MW * hail_damaged_perf_factor
            )

        else:
            effective_capacity_MW = available_end * global_performance_factor

        # --------------------------------------------------------------
        # Save results
        # --------------------------------------------------------------

        rows.append({
            "year": year,

            # New spatial-hazard outputs
            "county_hail_event": county_event,
            "site_hit": site_hit,
            "hail_size_in": hail_size,
            "footprint_area_sqmi": footprint_area,
            "coverage_fraction": coverage,

            # Compatibility:
            # hail_event still means the SITE was affected.
            "hail_event": site_hit,

            "f_rel": f_rel[i],

            # f_cat is now determined by the fragility curve
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


def run_monte_carlo_spatial(
    n_sims, P0_MW, location, years, hail_return, hail_dist,
    footprint_cdfs, county_areas, fragility_curves, module_area_m2,
    plr_cdf=None, t50=39, t90=42, base_seed=0,
    use_post_hail_plr=True, track_hail_buckets=True,
    apply_plr_once=True,
):
    """
    Run Monte Carlo simulations using the new spatial hail pipeline
    and module-area-dependent fragility curves.
    """

    sims = []

    for s in range(n_sims):

        res = simulate_project_spatial(
            P0_MW=P0_MW,
            location=location,
            years=years,
            hail_return=hail_return,
            hail_dist=hail_dist,
            footprint_cdfs=footprint_cdfs,
            county_areas=county_areas,
            fragility_curves=fragility_curves,
            module_area_m2=module_area_m2,
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