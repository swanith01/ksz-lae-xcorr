# Observational luminosity functions: where every number comes from

Compiled 2026-10-15 (session with Swanith Upadhye) for checking the 21cmFAST halo / LBG / LAE catalogues against observations
(scripts/39, scripts/40).  The CSVs are built by `build_compilation.py` (the transcribed numbers are in that file).

**How the numbers were obtained.**  Each table was read out of the paper's PDF / HTML text by an automated reader (WebFetch)
and transcribed without rounding.  Nothing was typed from memory.  They have **not yet been checked by eye against the PDFs**, so
before a number goes in a paper or a talk slide check it against the table cited here.  Rows flagged `verify=1` are the ones where
the reader itself reported garbled or inferred entries.

Units: UV LF `phi` in Mpc^-3 mag^-1 (comoving), M_UV in AB mag.  Lya LF `log_phi` in Mpc^-3 dex^-1, `log_L` in erg/s.

## UV luminosity function

| key | paper | arXiv | what was taken | notes |
|---|---|---|---|---|
| Bouwens21 | Bouwens et al. 2021, "New determinations of the UV luminosity functions from z~9 to z~2 ..." | 2102.07775 | Table 4 (stepwise phi_k, z=4-10) and Table 5 (Schechter, z=3.8-10.2) | z~10 rows are Oesch+18's, as listed in this table; phi_k +- err as printed |
| Donnan24 | Donnan et al. 2024, "JWST PRIMER: a new multi-field determination of the evolving galaxy UV LF at z~9-15" | 2403.03171 | Table 2 (binned, z=9,10,11,12.5,14.5) and Table 3 (double power law) | header unit "mag/Mpc^-3" read as 1e-6 Mpc^-3 mag^-1 |
| McLeod24 | McLeod et al. 2024, "The galaxy UV LF at z~11 from a suite of public JWST ERS, ERO and Cycle-1 programs" | 2304.14469 | Table 4 (binned, z=11 and 13.5) and Table 5 (DPL) | bins marked * come from Donnan+23 |
| Finkelstein24 | Finkelstein et al. 2024, "The complete CEERS early universe galaxy sample: a surprisingly slow evolution of the space density of bright galaxies at z~8.5-14.5" | 2311.04279 (ApJL 969, L2) | binned LF at z~9, 11, 14 (published PDF copy) | errors assigned by position by the reader; bins <20% complete are flagged in `note` |

Looked at but no numbers taken: Fontana et al. 2026 (2610.12121, review of the z>=10 UV LF; the text available had no tables or fit
parameters), Harikane et al. 2022 (GOLDRUSH IV, 2108.01090; Table 5 not in the text that could be read), Harikane et al. 2023
(2208.01612; individual candidates only), Oesch et al. 2018 (1710.11131; LF table not in the readable text, but its z~10 points
are in Bouwens+21 Table 4).

## Lya luminosity function

| key | paper | arXiv | what was taken | notes |
|---|---|---|---|---|
| Umeda25 | Umeda et al. 2025, SILVERRUSH XIV (HSC-SSP + CHORUS, ~20,000 LAEs) | 2411.15495 | Table 4 (binned, z=2.2-7.3 in narrow bands) and Table 3 (Schechter, z=4.9-7.3) | rows above log L = 44.05 left out |
| Kageura25 | Kageura et al. 2025, "Census of Lya emission from ~600 galaxies at z=5-14 ..." (JWST) | 2501.05834 | Table 2 (log Phi at logL = 42.3 / 42.8 / 43.3) and Table 3 (Schechter) | z = mean z of Table 3. **These are the blue circles with black edges in the Lya LF figure you sent** (values match to the digit) |
| Konno18 | Konno et al. 2018, SILVERRUSH IV (z=5.7, 6.6) | 1705.01222 | Table 3 (binned) and Table 4 ("classical" Schechter fits) | reader flagged some OCR garbling; last z=6.6 bin has the same errors as the one before |
| Santos16 | Santos, Sobral & Matthee 2016 | 1606.07435 | Table A3 (completeness-corrected log phi) and Table 4 | z=5.7 last bin label inferred; two z=6.6 lower errors garbled (blank) |
| Itoh18 | Itoh et al. 2018, CHORUS II (z=7.0) | 1805.05944 | Table 3 Schechter only (alpha fixed) | binned values not tabulated in the paper |
| Ouchi10 | Ouchi et al. 2010 (z=6.6) | 1007.2961 | Table 3 Schechter only | apparent L*, alpha fixed to -1.5 |
| Konno14 | Konno et al. 2014 (z=7.3) | 1404.6066 | Table 4 Schechter only | very large errors |

Looked at, no numbers taken: Ota et al. 2017 (1703.02501; LF table beyond the readable text), and the other points that appear in the
last figure you sent (Hu 2010/2019, Ning 2022, Matthee 2015, Wold 2022, Goto 2021, Zheng 2017, Taylor 2021, Ouchi 2008, Konno 2012):
not fetched yet.  The model curves in that figure (normal / log-normal line profile, REW distribution of Tang+24) come from a paper
I could not identify; Kageura+25 itself shows its LFs differently (its Fig. 6).

## Known gaps / to do

* spot-check every row against the PDFs (the `verify=1` rows first);
* HST z=6-8 points of Harikane+22 (GOLDRUSH) and the JWST z=6-8 LFs (Bouwens+23, Willott+24, Varadaraj+23) are not in yet.
