# VessFire comparison: wall3d - RUN 1 (superseded: 3-D curve-sampling bug, see wall3d.md)

140 validation-set cases (completed, horizontal, uninsulated), 18 failed runs of 296. Medians over cases; lower is closer to VessFire. Metric definitions: `validation/vessfire/compare_cases.py`.

## All cases

| config                |   cases |   P_rms_pct |   Tgas_rms_K |   Tdry_rms_K |   Thot_rms_K |   Twet_rms_K |   Tliq_rms_K |   rupt_err_s |   energy_err_abs_pct |   P_bias_pct |   Tgas_bias_K |   Tdry_bias_K |   Twet_bias_K |   Qfire_err_pct |
|:----------------------|--------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------------------:|-------------:|--------------:|--------------:|--------------:|----------------:|
| 1-D wall (regions)    |     138 |        4.96 |        27.69 |        43.42 |        16.31 |        20.12 |         8.7  |        139.5 |                 0.01 |         -0.7 |        -13.85 |        -18.75 |         -1.81 |            2.1  |
| 3-D wall (Hex8 shell) |     140 |        4.49 |        26.82 |        45.63 |        12.5  |        14.79 |        10.22 |        131   |                 0    |         -0.3 |        -11.16 |        -30.17 |         -7.52 |            2.21 |

## Wall < 30 mm

| config                |   cases |   P_rms_pct |   Tgas_rms_K |   Tdry_rms_K |   Thot_rms_K |   Twet_rms_K |   Tliq_rms_K |   rupt_err_s |   energy_err_abs_pct |   P_bias_pct |   Tgas_bias_K |   Tdry_bias_K |   Twet_bias_K |   Qfire_err_pct |
|:----------------------|--------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------------------:|-------------:|--------------:|--------------:|--------------:|----------------:|
| 1-D wall (regions)    |      19 |        7.18 |        28.53 |       281.59 |        40.72 |        41.62 |        10.73 |           54 |                 0.02 |        -0.66 |         -1.86 |       -274.96 |         28.05 |            4.26 |
| 3-D wall (Hex8 shell) |      20 |        8.4  |        32.83 |       290.07 |        35.11 |        23.79 |        14.4  |           44 |                 0    |         0.06 |         -5.58 |       -283.02 |         -7.18 |            4.76 |

## Wall 30-70 mm

| config                |   cases |   P_rms_pct |   Tgas_rms_K |   Tdry_rms_K |   Thot_rms_K |   Twet_rms_K |   Tliq_rms_K |   rupt_err_s |   energy_err_abs_pct |   P_bias_pct |   Tgas_bias_K |   Tdry_bias_K |   Twet_bias_K |   Qfire_err_pct |
|:----------------------|--------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------------------:|-------------:|--------------:|--------------:|--------------:|----------------:|
| 1-D wall (regions)    |     107 |        4.86 |        28.34 |        28.27 |        16.62 |        14.99 |         8.27 |        143   |                 0.01 |        -0.91 |        -19.96 |         -8.3  |         -2.67 |            1.75 |
| 3-D wall (Hex8 shell) |     108 |        4.35 |        26.53 |        33.31 |        12.37 |        14.09 |         8.58 |        139.5 |                 0    |        -0.5  |        -13.28 |        -13.63 |         -8.41 |            2.08 |

## Wall > 70 mm

| config                |   cases |   P_rms_pct |   Tgas_rms_K |   Tdry_rms_K |   Thot_rms_K |   Twet_rms_K |   Tliq_rms_K |   rupt_err_s |   energy_err_abs_pct |   P_bias_pct |   Tgas_bias_K |   Tdry_bias_K |   Twet_bias_K |   Qfire_err_pct |
|:----------------------|--------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------------------:|-------------:|--------------:|--------------:|--------------:|----------------:|
| 1-D wall (regions)    |      12 |        2.52 |        13.12 |       158.03 |         6.21 |         7.35 |        12.1  |        526.5 |                 0.01 |         0.04 |         -1.14 |       -141.55 |         -2.91 |            0.73 |
| 3-D wall (Hex8 shell) |      12 |        2.57 |        13.25 |       153.44 |         5.75 |         4.12 |         8.32 |        376.5 |                 0    |         0.05 |         -1.15 |       -137.61 |          1.29 |            0.66 |

## Tresca rupture time (membrane check)

| config                |   VF ruptures |   model ruptures |   both |   median |dt| s (both) |
|:----------------------|--------------:|-----------------:|-------:|-----------------------:|
| 1-D wall (regions)    |            63 |               68 |     60 |                  139.5 |
| 3-D wall (Hex8 shell) |            63 |               68 |     60 |                  131   |

## Golden cases

| case     |   ('P_rms_pct', '1-D wall (regions)') |   ('P_rms_pct', '3-D wall (Hex8 shell)') |   ('Tdry_rms_K', '1-D wall (regions)') |   ('Tdry_rms_K', '3-D wall (Hex8 shell)') |
|:---------|--------------------------------------:|-----------------------------------------:|---------------------------------------:|------------------------------------------:|
| M03-0003 |                                   0.2 |                                      0.2 |                                    0.2 |                                       0.2 |
| M04-0003 |                                   2.8 |                                      2.8 |                                    0.5 |                                       0.5 |
| M05-0003 |                                   0.5 |                                      0.5 |                                    0.2 |                                       0.5 |
| M06-0003 |                                   7.1 |                                      7.1 |                                   13.7 |                                      12.4 |
| M07-0004 |                                   4.6 |                                      4.6 |                                    5.5 |                                       6.6 |
| M08-0001 |                                   2.7 |                                      2.7 |                                    5.9 |                                       6.7 |
| M09-0001 |                                   4.6 |                                      4.6 |                                    5.5 |                                       6.6 |
| M10-0001 |                                   7.9 |                                      7.9 |                                    9.7 |                                       8.3 |
| M11-0002 |                                  25.6 |                                     23.6 |                                   44.3 |                                      27.8 |
| M12-0004 |                                   4   |                                      4.1 |                                   12.4 |                                      19.1 |
| M15-0002 |                                   8.6 |                                      6.2 |                                    4.9 |                                       4.2 |

## Failed runs

| case     | config                | error                                                                             |
|:---------|:----------------------|:----------------------------------------------------------------------------------|
| M01-0027 | 1-D wall (regions)    | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M01-0027 | 3-D wall (Hex8 shell) | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M12-0010 | 1-D wall (regions)    | ValueError: math domain error                                                     |
| M14-0044 | 1-D wall (regions)    | ValueError: unknown component BNZ (give pseudo properties)                        |
| M14-0044 | 3-D wall (Hex8 shell) | ValueError: unknown component BNZ (give pseudo properties)                        |
| M14-0080 | 1-D wall (regions)    | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0080 | 3-D wall (Hex8 shell) | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0089 | 1-D wall (regions)    | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0089 | 3-D wall (Hex8 shell) | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0157 | 1-D wall (regions)    | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0157 | 3-D wall (Hex8 shell) | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0166 | 1-D wall (regions)    | RuntimeError: root outside bounds                                                 |
| M14-0189 | 1-D wall (regions)    | ValueError: unknown component BNZ (give pseudo properties)                        |
| M14-0189 | 3-D wall (Hex8 shell) | ValueError: unknown component BNZ (give pseudo properties)                        |
| M14-0192 | 1-D wall (regions)    | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0192 | 3-D wall (Hex8 shell) | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M16-0033 | 1-D wall (regions)    | ValueError: unknown component BNZ (give pseudo properties)                        |
| M16-0033 | 3-D wall (Hex8 shell) | ValueError: unknown component BNZ (give pseudo properties)                        |
