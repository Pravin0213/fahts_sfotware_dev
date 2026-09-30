# VessFire comparison: peak_zone

67 validation-set cases (completed, horizontal, uninsulated), 11 failed runs of 144. Medians over cases; lower is closer to VessFire. Metric definitions: `validation/vessfire/compare_cases.py`.

## All cases

| config                        |   cases |   P_rms_pct |   Tgas_rms_K |   Tdry_rms_K |   Thot_rms_K |   Twet_rms_K |   Tliq_rms_K |   rupt_err_s |   energy_err_abs_pct |   P_bias_pct |   Tgas_bias_K |   Tdry_bias_K |   Twet_bias_K |   Qfire_err_pct |
|:------------------------------|--------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------------------:|-------------:|--------------:|--------------:|--------------:|----------------:|
| before (background flux only) |      67 |        7.07 |        37.86 |       271.32 |       271.32 |        32.95 |        15.45 |          759 |                 0.01 |         -1.8 |        -31.06 |       -260.55 |         -0.01 |           -0.15 |
| after (peak zone regions)     |      66 |        5.12 |        30.09 |       264.84 |        25.44 |        25.71 |        11.55 |          139 |                 0.01 |         -0.7 |        -16.12 |       -256.16 |         12.45 |            3.2  |

## Wall < 30 mm

| config                        |   cases |   P_rms_pct |   Tgas_rms_K |   Tdry_rms_K |   Thot_rms_K |   Twet_rms_K |   Tliq_rms_K |   rupt_err_s |   energy_err_abs_pct |   P_bias_pct |   Tgas_bias_K |   Tdry_bias_K |   Twet_bias_K |   Qfire_err_pct |
|:------------------------------|--------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------------------:|-------------:|--------------:|--------------:|--------------:|----------------:|
| before (background flux only) |      20 |        9.01 |        34.96 |       293.12 |       293.12 |        34.09 |        10.11 |          596 |                 0.02 |        -0.56 |        -29.33 |       -286.02 |         21.09 |            1.57 |
| after (peak zone regions)     |      19 |        7.18 |        28.53 |       281.59 |        40.72 |        41.62 |        10.73 |           54 |                 0.02 |        -0.66 |         -1.86 |       -274.96 |         28.05 |            4.26 |

## Wall 30-70 mm

| config                        |   cases |   P_rms_pct |   Tgas_rms_K |   Tdry_rms_K |   Thot_rms_K |   Twet_rms_K |   Tliq_rms_K |   rupt_err_s |   energy_err_abs_pct |   P_bias_pct |   Tgas_bias_K |   Tdry_bias_K |   Twet_bias_K |   Qfire_err_pct |
|:------------------------------|--------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------------------:|-------------:|--------------:|--------------:|--------------:|----------------:|
| before (background flux only) |      40 |        7.18 |        38.69 |       266.68 |       266.68 |        29.81 |        24.67 |        873.5 |                 0.01 |        -4.18 |        -32    |       -257.04 |         -9.65 |           -1.18 |
| after (peak zone regions)     |      40 |        5.12 |        32.01 |       263.56 |        23.9  |        22.4  |        12.6  |        142   |                 0.01 |        -1.4  |        -24.69 |       -254.75 |         -2.16 |            2.31 |

## Wall > 70 mm

| config                        |   cases |   P_rms_pct |   Tgas_rms_K |   Tdry_rms_K |   Thot_rms_K |   Twet_rms_K |   Tliq_rms_K |   rupt_err_s |   energy_err_abs_pct |   P_bias_pct |   Tgas_bias_K |   Tdry_bias_K |   Twet_bias_K |   Qfire_err_pct |
|:------------------------------|--------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------------------:|-------------:|--------------:|--------------:|--------------:|----------------:|
| before (background flux only) |       7 |        2.2  |        24.25 |       275.52 |       275.52 |        22.01 |        36.82 |        nan   |                 0.02 |        -0.29 |        -20.12 |       -259.8  |          9.59 |           -3.17 |
| after (peak zone regions)     |       7 |        2.28 |        21.16 |       274.63 |         8.46 |        20.4  |        27.8  |        526.5 |                 0.01 |         0.08 |        -15.99 |       -259.05 |         13.76 |            1.37 |

## Tresca rupture time (membrane check)

| config                        |   VF ruptures |   model ruptures |   both |   median |dt| s (both) |
|:------------------------------|--------------:|-----------------:|-------:|-----------------------:|
| before (background flux only) |            44 |               27 |     27 |                    759 |
| after (peak zone regions)     |            44 |               49 |     43 |                    139 |

## Golden cases

| case   |
|--------|

## Failed runs

| case     | config                        | error                                                                             |
|:---------|:------------------------------|:----------------------------------------------------------------------------------|
| M14-0044 | before (background flux only) | ValueError: unknown component BNZ (give pseudo properties)                        |
| M14-0044 | after (peak zone regions)     | ValueError: unknown component BNZ (give pseudo properties)                        |
| M14-0080 | before (background flux only) | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0080 | after (peak zone regions)     | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0089 | before (background flux only) | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0089 | after (peak zone regions)     | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0166 | after (peak zone regions)     | RuntimeError: root outside bounds                                                 |
| M14-0189 | before (background flux only) | ValueError: unknown component BNZ (give pseudo properties)                        |
| M14-0189 | after (peak zone regions)     | ValueError: unknown component BNZ (give pseudo properties)                        |
| M14-0192 | before (background flux only) | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0192 | after (peak zone regions)     | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
