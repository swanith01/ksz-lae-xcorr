#!/usr/bin/env python3
"""
Builds the four CSVs of this folder from numbers transcribed out of the papers (see SOURCES.md for the table / arXiv id of
every block).  Re-run after editing:   python data/reference/luminosity_functions/build_compilation.py

  uvlf_points.csv       binned UV LF       phi [Mpc^-3 mag^-1]
  uvlf_fits.csv         Schechter / double-power-law fits (phi* in Mpc^-3)
  lae_lf_points.csv     binned Lya LF      log10 phi [Mpc^-3 dex^-1]   (dex = dlogL)
  lae_lf_fits.csv       Schechter fits     log10 phi*, log10 L*, alpha

Every number was read out of the paper's table by an automated text reader and is NOT yet checked by eye against the PDF:
rows with verify=1 are the ones where the reader reported garbled or inferred entries.
"""
import csv
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
NAN = float("nan")


def write(name, header, rows):
    with open(os.path.join(HERE, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if (isinstance(x, float) and math.isnan(x)) else (f"{x:.8g}" if isinstance(x, float) else x) for x in r])
    print(f"{name}: {len(rows)} rows")


# ------------------------------------------------------------------ UV luminosity function: binned points
UV_HEAD = ["ref_key", "z", "z_label", "M_UV", "phi_Mpc3_mag", "err_up", "err_lo", "upper_limit", "table", "verify", "note"]
uv = []

# Bouwens+21 (arXiv:2102.07775) Table 4: phi_k [Mpc^-3 mag^-1] +- err (symmetric as printed); z~10 rows = Oesch+18 as listed there
B21 = """4,-22.69,0.000005,0.000004
4,-22.19,0.000015,0.000009
4,-21.69,0.000144,0.000022
4,-21.19,0.000344,0.000038
4,-20.69,0.000698,0.000068
4,-20.19,0.001624,0.000131
4,-19.69,0.002276,0.000199
4,-19.19,0.003056,0.000388
4,-18.69,0.004371,0.000689
4,-17.94,0.010160,0.000920
4,-16.94,0.027420,0.003440
4,-15.94,0.028820,0.008740
5,-23.11,0.000001,0.000001
5,-22.61,0.000004,0.000002
5,-22.11,0.000028,0.000007
5,-21.61,0.000092,0.000013
5,-21.11,0.000262,0.000024
5,-20.61,0.000584,0.000044
5,-20.11,0.000879,0.000067
5,-19.61,0.001594,0.000156
5,-19.11,0.002159,0.000346
5,-18.36,0.004620,0.000520
5,-17.36,0.008780,0.001540
6,-22.52,0.000002,0.000002
6,-22.02,0.000014,0.000005
6,-21.52,0.000051,0.000011
6,-21.02,0.000169,0.000024
6,-20.52,0.000317,0.000041
6,-20.02,0.000724,0.000087
6,-19.52,0.001147,0.000157
6,-18.77,0.002820,0.000440
6,-17.77,0.008360,0.001660
6,-16.77,0.017100,0.005260
7,-22.19,0.000001,0.000002
7,-21.69,0.000041,0.000011
7,-21.19,0.000047,0.000015
7,-20.69,0.000198,0.000036
7,-20.19,0.000283,0.000066
7,-19.69,0.000589,0.000126
7,-19.19,0.001172,0.000336
7,-18.69,0.001433,0.000419
7,-17.94,0.005760,0.001440
7,-16.94,0.008320,0.002900
8,-21.85,0.000003,0.000002
8,-21.35,0.000012,0.000004
8,-20.85,0.000041,0.000011
8,-20.10,0.000120,0.000040
8,-19.35,0.000657,0.000233
8,-18.60,0.001100,0.000340
8,-17.60,0.003020,0.001140
9,-21.92,0.000001,0.000001
9,-21.12,0.000007,0.000003
9,-20.32,0.000026,0.000009
9,-19.12,0.000187,0.000150
9,-17.92,0.000923,0.000501
10,-21.25,0.000001,0.000001
10,-20.25,0.000010,0.000005
10,-19.25,0.000034,0.000022
10,-18.25,0.000190,0.000120
10,-17.25,0.000630,0.000520"""
for line in B21.splitlines():
    z, m, p, e = line.split(",")
    note = "Oesch+18 z~10 points as tabulated in Bouwens+21" if z == "10" else ""
    uv.append(["Bouwens21", float(z), "", float(m), float(p), float(e), float(e), 0, "Table 4", 0, note])
uv.append(["Bouwens21", 10.0, "", -22.25, 0.000002, NAN, NAN, 1, "Table 4", 0, "upper limit as printed (<0.000002); Oesch+18 z~10"])

# Donnan+24 (arXiv:2403.03171) Table 2: phi in 1e-6 mag^-1 Mpc^-3; asymmetric errors, 'not stated' lower -> blank
D24 = """9,-20.75,12,8,
9,-20.25,32,13,10
9,-19.75,144,30,28
9,-19.25,235,60,49
9,-18.55,486,157,139
9,-18.05,1110,310,310
9,-17.55,1776,578,510
10,-20.75,4,10,4
10,-20.25,27,13,10
10,-19.75,92,25,20
10,-19.25,177,53,45
10,-18.55,321,127,111
10,-18.05,686,245,223
10,-17.55,1278,486,432
11,-21.25,7,9,5
11,-20.75,14,11,7
11,-20.25,38,16,13
11,-19.75,100,37,30
11,-19.25,144,81,63
11,-18.75,234,118,96
11,-18.25,641,361,281
12.5,-21.25,3,4,
12.5,-20.75,4,5,3
12.5,-20.25,16,9,6
12.5,-19.75,34,23,15
12.5,-19.25,43,35,22
12.5,-18.75,80,51,36
12.5,-18.25,217,153,104
14.5,-20.25,3,6,"""
for line in D24.splitlines():
    z, m, p, up, lo = line.split(",")
    uv.append(["Donnan24", float(z), "", float(m), float(p) * 1e-6, float(up) * 1e-6, float(lo) * 1e-6 if lo else NAN, 0,
               "Table 2", 0, "table header unit 'mag/Mpc^-3' read as 1e-6 Mpc^-3 mag^-1"])

# McLeod+24 (arXiv:2304.14469) Table 4: phi in 1e-5 Mpc^-3 mag^-1, symmetric error; '*' = Donnan+23 bins
M24 = """11.0,-22.57,0.012,0.012,1
11.0,-21.80,0.129,0.128,0
11.0,-20.80,1.254,0.428,0
11.0,-20.05,3.974,1.340,0
11.0,-19.55,9.863,4.197,0
11.0,-18.85,23.490,9.190,1
11.0,-18.23,63.080,28.650,1
13.5,-19.45,2.469,1.659,0
13.5,-18.95,6.199,3.974,0"""
for line in M24.splitlines():
    z, m, p, e, d = line.split(",")
    uv.append(["McLeod24", float(z), "", float(m), float(p) * 1e-5, float(e) * 1e-5, float(e) * 1e-5, 0, "Table 4", 0,
               "bin taken from Donnan+23 (as marked in the table)" if d == "1" else ""])

# Finkelstein+24 (arXiv:2311.04279) binned LF: phi in 1e-5 Mpc^-3 mag^-1, '<' = upper limit, complete=no flagged
F24 = """9,-22.5,0.9,,,1,0
9,-22.0,1.1,0.6,0.7,0,0
9,-21.5,0.9,,,1,0
9,-21.0,2.2,1.0,1.3,0,0
9,-20.5,8.2,3.2,4.0,0,0
9,-20.0,9.6,3.6,4.6,0,0
9,-19.5,28.6,9.1,11.5,0,0
9,-19.0,26.8,10.0,12.4,0,0
9,-18.5,136.0,49.9,61.0,0,1
11,-21.0,0.5,,,1,0
11,-20.5,1.8,0.9,1.2,0,0
11,-20.0,5.4,2.1,2.7,0,0
11,-19.5,7.6,3.0,3.9,0,0
11,-19.0,17.6,7.9,10.3,0,1
11,-18.5,26.3,13.3,18.2,0,1
14,-20.5,1.8,,,1,0
14,-20.0,2.6,1.8,3.3,0,0
14,-19.5,7.3,4.4,6.9,0,0"""
for line in F24.splitlines():
    z, m, p, up, lo, ul, inc = line.split(",")
    uv.append(["Finkelstein24", float(z), "", float(m), float(p) * 1e-5, float(up) * 1e-5 if up else NAN,
               float(lo) * 1e-5 if lo else NAN, int(ul), "Table 4 (binned LF)", 1,
               ("<20% complete bin, do not use for constraints; " if inc == "1" else "") + "errors assigned by position by the reader"])
write("uvlf_points.csv", UV_HEAD, uv)

# ------------------------------------------------------------------ UV LF: fits
FIT_HEAD = ["ref_key", "form", "z", "Mstar", "Mstar_err", "phistar_Mpc3", "phistar_err_up", "phistar_err_lo", "alpha", "alpha_err",
            "beta", "beta_err", "table", "note"]
fit = []
# Bouwens+21 Table 5 Schechter; phi* in 1e-3 Mpc^-3
B21S = [(3.8, -20.93, .08, 1.69, .22, .20, -1.69, .03, ""), (4.9, -21.10, .11, .79, .16, .13, -1.74, .06, ""),
        (5.9, -20.93, .09, .51, .12, .10, -1.93, .08, ""), (6.8, -21.15, .13, .19, .08, .06, -2.06, .11, ""),
        (7.9, -20.93, .28, .09, .09, .05, -2.23, .20, ""), (8.9, -21.15, NAN, .021, .014, .009, -2.33, .19, "M* fixed"),
        (10.2, -21.19, NAN, .0042, .0045, .0022, -2.38, .28, "M* fixed; Oesch+18 row of the table")]
for z, M, Me, p, pu, pl, a, ae, n in B21S:
    fit.append(["Bouwens21", "Schechter", z, M, Me, p * 1e-3, pu * 1e-3, pl * 1e-3, a, ae, NAN, NAN, "Table 5", n])
# Donnan+24 Table 3 DPL: phi* in 1e-5 Mpc^-3 mag^-1; '*' fixed
D24F = [(9, 23.5, 39.2, -19.70, .96, -2.00, .47, -3.81, .49, ""), (10, 14.5, 16.4, -19.98, .61, -1.98, .40, -4.05, NAN, "beta fixed"),
        (11, 3.27, 9.86, -20.73, 1.61, -2.19, .69, -4.29, 1.30, ""), (12.5, .99, .99, -20.82, .71, -2.19, NAN, -4.29, NAN, "alpha, beta fixed"),
        (14.5, .18, .28, -20.82, NAN, -2.19, NAN, -4.29, NAN, "M*, alpha, beta fixed")]
for z, p, pe, M, Me, a, ae, b, be, n in D24F:
    fit.append(["Donnan24", "DPL", z, M, Me, p * 1e-5, pe * 1e-5, pe * 1e-5, a, ae, b, be, "Table 3", n])
# McLeod+24 Table 5 DPL: phi* in 1e-5 Mpc^-3
fit.append(["McLeod24", "DPL", 11.0, -21.62, 1.28, 0.43e-5, 1.15e-5, 1.15e-5, -2.59, .35, -4.99, 2.63, "Table 5", "DPL free"])
fit.append(["McLeod24", "DPL", 11.0, -20.87, .63, 2.04e-5, 2.13e-5, 2.13e-5, -2.35, NAN, -4.16, .76, "Table 5", "DPL fixed alpha"])
write("uvlf_fits.csv", FIT_HEAD, fit)

# ------------------------------------------------------------------ Lya luminosity function: binned points
LH = ["ref_key", "z", "z_label", "log_L", "log_phi", "err_up", "err_lo", "upper_limit", "table", "verify", "note"]
lae = []

# Umeda+25 = SILVERRUSH XIV (arXiv:2411.15495) Table 4: 'value +up -lo', '< x' upper limit, '(lower n/a)'
U25 = """42.25,-2.59 +0.03 -0.03,,,,,,
42.35,-2.71 +0.02 -0.02,,,,,,
42.45,-2.84 +0.02 -0.02,-2.25 +0.02 -0.02,,,,,
42.55,-2.93 +0.01 (lower n/a),-2.58 +0.03 -0.03,,,,,
42.65,-2.99 +0.01 -0.01,-2.75 +0.03 -0.03,,,,,
42.75,-3.15 +0.02 -0.02,-2.96 +0.04 -0.04,-2.91 +0.05 -0.05,,,,
42.85,-3.30 +0.02 -0.02,-3.15 +0.05 -0.05,-3.17 +0.05 -0.05,-3.05 +0.04 -0.04,,,
42.95,-3.53 +0.02 -0.02,-3.36 +0.06 -0.06,-3.42 +0.06 -0.06,-3.27 +0.02 -0.02,-3.71 +0.09 -0.09,,
43.05,-3.72 +0.03 -0.03,-3.62 +0.08 -0.08,-3.78 +0.09 -0.09,-3.56 +0.02 -0.02,-4.11 +0.05 -0.05,,
43.15,-3.95 +0.04 (lower n/a),-3.74 +0.09 -0.09,-3.88 +0.10 -0.10,-3.85 +0.03 -0.03,-4.37 +0.06 -0.06,,
43.25,-4.11 +0.04 -0.04,-3.98 +0.12 -0.13,-4.00 +0.12 -0.12,-4.15 +0.04 (lower n/a),-4.65 +0.07 -0.07,-4.40 +0.29 -0.34,
43.35,-4.26 +0.05 -0.05,-4.31 +0.19 -0.20,-4.75 +0.29 -0.34,-4.41 +0.05 -0.05,-4.83 +0.08 -0.08,-4.95 +0.52 -0.77,< -4.77
43.45,-4.42 +0.06 -0.06,-4.38 +0.20 -0.22,-4.93 +0.36 -0.45,-4.72 +0.07 -0.07,-5.28 +0.14 -0.15,< -4.87,-4.81 +0.36 -0.45
43.55,-4.60 +0.08 (lower n/a),-4.86 +0.36 -0.45,-5.23 +0.52 -0.77,-5.15 +0.12 -0.13,-5.89 +0.29 -0.34,< -4.87,< -5.07
43.65,-4.64 +0.08 (lower n/a),-5.16 +0.52 -0.77,-4.93 +0.36 -0.45,-5.43 +0.17 -0.18,-5.90 +0.29 -0.34,< -4.87,< -5.07
43.75,-4.81 +0.10 (lower n/a),-5.16 +0.52 -0.77,< -4.98,-6.03 +0.36 -0.45,-5.90 +0.29 -0.34,< -4.87,< -5.07
43.85,-4.81 +0.10 -0.10,-5.16 +0.52 -0.77,< -4.98,-6.33 +0.52 -0.77,< -6.12,< -4.87,< -5.07
43.95,-4.78 +0.10 -0.10,-4.86 +0.36 -0.45,< -4.98,-6.33 +0.52 -0.77,-6.38 +0.52 -0.77,< -4.87,< -5.07
44.05,-4.90 +0.11 (lower n/a),-5.16 +0.52 -0.77,< -4.98,< -6.07,-6.38 +0.52 -0.77,< -4.87,< -5.07"""
U25_Z = [2.2, 3.3, 4.9, 5.7, 6.6, 7.0, 7.3]
for line in U25.splitlines():
    cells = line.split(",")
    logL = float(cells[0])
    for z, c in zip(U25_Z, cells[1:]):
        c = c.strip()
        if not c:
            continue
        if c.startswith("<"):
            lae.append(["Umeda25", z, "", logL, float(c[1:]), NAN, NAN, 1, "Table 4", 0, "upper limit"])
            continue
        m = re.match(r"(-?\d+\.\d+) \+(\d+\.\d+) (?:-(\d+\.\d+)|\(lower n/a\))", c)
        lae.append(["Umeda25", z, "", logL, float(m.group(1)), float(m.group(2)), float(m.group(3)) if m.group(3) else NAN, 0,
                    "Table 4", 0, "" if m.group(3) else "lower error n/a in the table"])
# (44.15-44.95 rows of Table 4 only matter at z=2.2 and are left out)

# Kageura+25 (arXiv:2501.05834) Table 2 (JWST spectroscopic census), z_label as in the table; z = mean z of Table 3
K25 = [(5.01, "z~5", [(42.3, -2.45, .11, .12), (42.8, -2.80, .09, .10), (43.3, -3.49, .15, .15)]),
       (5.90, "z~6", [(42.3, -2.80, .08, .13), (42.8, -3.25, .14, .15), (43.3, -4.00, .06, .26)]),
       (6.96, "z~7", [(42.3, -2.91, .20, .31), (42.8, -3.87, .19, .24), (43.3, -4.47, .29, .14)]),
       (8.41, "z~8-9", [(42.3, -3.59, .36, .48), (42.8, -4.38, .11, .74)]),
       (11.00, "z~10-14", [(42.3, -4.57, .52, .83), (42.8, -6.07, .65, .59)])]
for z, lab, pts in K25:
    for L, p, up, lo in pts:
        lae.append(["Kageura25", z, lab, L, p, up, lo, 0, "Table 2", 0, "log Phi per dex; mean z from Table 3"])

# Konno+18 = SILVERRUSH IV (arXiv:1705.01222) Table 3, bin 0.1 dex
K18 = """5.7,42.95,-3.478,0.038,0.193
5.7,43.05,-3.735,0.044,0.199
5.7,43.15,-3.953,0.043,0.198
5.7,43.25,-4.163,0.055,0.210
5.7,43.35,-4.427,0.076,0.231
5.7,43.45,-4.970,0.147,0.308
5.7,43.55,-5.170,0.187,0.355
5.7,43.65,-5.318,0.224,0.401
5.7,43.75,-5.717,0.365,0.606
6.6,43.15,-4.194,0.154,0.317
6.6,43.25,-4.407,0.101,0.258
6.6,43.35,-4.748,0.087,0.243
6.6,43.45,-5.132,0.140,0.300
6.6,43.55,-5.433,0.203,0.374
6.6,43.65,-5.609,0.253,0.438
6.6,43.75,-6.212,0.519,0.917
6.6,43.85,-6.226,0.519,0.917"""
for line in K18.splitlines():
    z, L, p, up, lo = line.split(",")
    lae.append(["Konno18", float(z), "", float(L), float(p), float(up), float(lo), 0, "Table 3", 1 if (z == "6.6" and L == "43.85") else 0,
                "last bin: errors identical to the previous bin, possible transcription artifact" if (z == "6.6" and L == "43.85") else ""])

# Santos+16 (arXiv:1606.07435) Table A3, completeness-corrected log phi
S16 = [(5.7, 42.52, -2.63, .16, .17, 0, ""), (5.7, 42.72, -2.77, .12, .13, 0, ""), (5.7, 42.92, -3.15, .10, .10, 0, ""),
       (5.7, 43.12, -3.54, .08, .08, 0, ""), (5.7, 43.32, -3.91, .09, .10, 0, ""), (5.7, 43.52, -4.27, .11, .12, 0, ""),
       (5.7, 43.72, -5.12, .22, .40, 1, "bin label inferred by the reader (not printed)"),
       (6.6, 42.61, -3.18, .08, .09, 0, ""), (6.6, 42.81, -3.32, .08, .08, 0, ""), (6.6, 43.01, -3.74, .09, .10, 0, ""),
       (6.6, 43.21, -4.10, .10, .11, 0, ""), (6.6, 43.41, -4.60, .14, .16, 0, ""),
       (6.6, 43.61, -4.97, .16, NAN, 1, "lower error garbled in the source text"),
       (6.6, 43.81, -5.51, .20, .26, 1, "last-bin errors partly garbled in the source text")]
for z, L, p, up, lo, v, n in S16:
    lae.append(["Santos16", z, "", L, p, up, lo, 0, "Table A3 (corrected)", v, n])
write("lae_lf_points.csv", LH, lae)

# ------------------------------------------------------------------ Lya LF: fits (log phi* [Mpc^-3 dex^-1], log L* [erg/s])
LFH = ["ref_key", "z", "log_phistar", "log_phistar_up", "log_phistar_lo", "log_Lstar", "log_Lstar_up", "log_Lstar_lo", "alpha", "alpha_err",
       "table", "note"]
lf = []


def lg(val, up, lo):
    """log10 of val with asymmetric linear errors -> (log, err_up, err_lo)"""
    return math.log10(val), math.log10((val + up) / val), (-math.log10((val - lo) / val) if val - lo > 0 else NAN)


# Umeda+25 Table 3 (logs as printed)
for z, p, pu, pl, L, Lu, Ll, a, ae in [(4.9, -2.79, .11, .12, 42.69, .06, .06, -1.61, .07), (5.7, -2.91, .05, .05, 42.75, .03, .03, -1.39, .06),
                                       (6.6, -4.62, .50, .74, 43.28, .26, .18, -2.82, .32), (7.0, -4.49, .85, 1.27, 43.23, .49, .39, -2.49, .82),
                                       (7.3, -5.00, 1.40, 1.51, 43.03, .62, .61, -2.56, .93)]:
    lf.append(["Umeda25", z, p, pu, pl, L, Lu, Ll, a, ae, "Table 3", ""])
# Kageura+25 Table 3 (log L*, alpha fixed to Umeda values)
for z, p, pu, pl, L, a in [(5.01, -2.50, .06, .06, 42.69, -1.61), (5.90, -3.03, .07, .13, 42.75, -1.39), (6.96, -4.45, .09, .10, 43.23, -2.49),
                           (8.41, -4.98, .21, .35, 43.03, -2.56), (11.00, -6.33, .44, .50, 43.03, -2.56)]:
    lf.append(["Kageura25", z, p, pu, pl, L, NAN, NAN, a, NAN, "Table 3", "L* and alpha fixed"])
# Konno+18 Table 4 'classical', logL range 42.4-43.5 and 42.4-44.0 : L* [1e43], phi* [1e-4]
for z, rng, Ls, Lu, Ll, ps, pu, pl, a, au, al in [(5.7, "42.4-43.5", 1.07, .77, .38, 2.46, 3.48, 1.86, -2.26, .76, .44),
                                                   (6.6, "42.4-43.5", .82, .86, .30, 2.83, 3.52, 2.38, -1.86, .79, .67),
                                                   (5.7, "42.4-44.0", 1.64, 2.16, .62, .849, 1.87, .771, -2.56, .53, .45),
                                                   (6.6, "42.4-44.0", 1.66, .30, .69, .467, 1.44, .442, -2.49, .50, .50)]:
    lp, lpu, lpl = lg(ps * 1e-4, pu * 1e-4, pl * 1e-4)
    lL, lLu, lLl = lg(Ls * 1e43, Lu * 1e43, Ll * 1e43)
    lf.append(["Konno18", z, lp, lpu, lpl, lL, lLu, lLl, a, max(au, al), "Table 4 (classical, fit range logL=" + rng + ")", "reader flagged OCR garbling of some cells"])
# Santos+16 Table 4
for z, a, ae, L, Lu, Ll, p, pu, pl, n in [(5.7, -2.3, .35, 43.42, .50, .22, -4.02, .48, .93, "alpha free"), (5.7, -1.5, NAN, 43.06, .05, .04, -3.25, .09, .10, "alpha fixed"),
                                          (5.7, -2.0, NAN, 43.25, .09, .06, -3.63, .12, .16, "alpha fixed"), (6.6, -2.3, .35, 43.45, .35, .18, -4.48, .43, .68, "alpha free"),
                                          (6.6, -1.5, NAN, 43.12, .04, .03, -3.73, .07, .06, "alpha fixed"), (6.6, -2.0, NAN, 43.30, .07, .05, -4.13, .10, .10, "alpha fixed")]:
    lf.append(["Santos16", z, p, pu, pl, L, Lu, Ll, a, ae, "Table 4", n])
# Itoh+18 Table 3 (z=7.0, alpha fixed)
for L, Lu, Ll, p, pu, pl, a, n in [(1.50, .42, .21, .45, .26, .18, -2.5, "fiducial"), (.92, .38, .15, 1.41, 1.14, .77, -2.0, ""), (.63, .11, .12, 2.80, 1.91, .90, -1.5, "")]:
    lp, lpu, lpl = lg(p * 1e-4, pu * 1e-4, pl * 1e-4)
    lL, lLu, lLl = lg(L * 1e43, Lu * 1e43, Ll * 1e43)
    lf.append(["Itoh18", 7.0, lp, lpu, lpl, lL, lLu, lLl, a, NAN, "Table 3", "alpha fixed " + n])
# Ouchi+10 Table 3 (z=6.6, apparent, alpha fixed) and Konno+14 Table 4 (z=7.3, alpha fixed)
lp, lpu, lpl = lg(8.5e-4, 3.0e-4, 2.2e-4); lL, lLu, lLl = lg(4.4e42, 0.6e42, 0.6e42)
lf.append(["Ouchi10", 6.6, lp, lpu, lpl, lL, lLu, lLl, -1.5, NAN, "Table 3", "alpha fixed; apparent L*"])
lp, lpu, lpl = lg(3.7e-4, 17.6e-4, 3.3e-4); lL, lLu, lLl = lg(2.7e42, 8.0e42, 1.2e42)
lf.append(["Konno14", 7.3, lp, lpu, lpl, lL, lLu, lLl, -1.5, NAN, "Table 4", "alpha fixed; huge errors"])
write("lae_lf_fits.csv", LFH, lf)
