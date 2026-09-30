# VessFire comparison: wall_grid_fix

74 validation-set cases (completed, horizontal, uninsulated), 11 failed runs of 231. Medians over cases; lower is closer to VessFire. Metric definitions: `validation/vessfire/compare_cases.py`.

## All cases

| config                   |   cases |   P_rms_pct |   Tgas_rms_K |   Tdry_rms_K |   Twet_rms_K |   Tliq_rms_K |   rupt_err_s |   energy_err_abs_pct |
|:-------------------------|--------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------------------:|
| before (105 mm grid)     |      73 |       10.23 |       119.62 |       119.62 |        59.3  |        15.76 |          897 |                10.03 |
| after (real t, 10 cells) |      73 |        4.6  |        16.33 |         6.24 |        11.75 |         6.5  |          147 |                 0    |
| after (real t, 20 cells) |      74 |        4.59 |        16    |         6.13 |        11.7  |         6.35 |          147 |                 0    |

## Wall 30-70 mm

| config                   |   cases |   P_rms_pct |   Tgas_rms_K |   Tdry_rms_K |   Twet_rms_K |   Tliq_rms_K |   rupt_err_s |   energy_err_abs_pct |
|:-------------------------|--------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------------------:|
| before (105 mm grid)     |      67 |       10.39 |       135.4  |       121.42 |        78.06 |        20.86 |          897 |                10.92 |
| after (real t, 10 cells) |      67 |        4.6  |        21.45 |         8.28 |        12.48 |         6.5  |          147 |                 0    |
| after (real t, 20 cells) |      68 |        4.59 |        20.06 |         7.95 |        12.13 |         6.35 |          147 |                 0    |

## Wall > 70 mm

| config                   |   cases |   P_rms_pct |   Tgas_rms_K |   Tdry_rms_K |   Twet_rms_K |   Tliq_rms_K |   rupt_err_s |   energy_err_abs_pct |
|:-------------------------|--------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------------------:|
| before (105 mm grid)     |       6 |        3.97 |        12.9  |         7.3  |        16.69 |        10.81 |          nan |                 2.72 |
| after (real t, 10 cells) |       6 |        1.84 |         3.91 |         2.4  |         7.35 |         6.63 |          nan |                 0    |
| after (real t, 20 cells) |       6 |        1.84 |         3.91 |         2.41 |         7.35 |         6.6  |          nan |                 0    |

## Tresca rupture time (membrane check)

| config                   |   VF ruptures |   model ruptures |   both |   median |dt| s (both) |
|:-------------------------|--------------:|-----------------:|-------:|-----------------------:|
| before (105 mm grid)     |            19 |               13 |     13 |                    897 |
| after (real t, 10 cells) |            19 |               19 |     17 |                    147 |
| after (real t, 20 cells) |            19 |               19 |     17 |                    147 |

## Golden cases

| case     |   ('P_rms_pct', 'after (real t, 10 cells)') |   ('P_rms_pct', 'after (real t, 20 cells)') |   ('P_rms_pct', 'before (105 mm grid)') |   ('Tdry_rms_K', 'after (real t, 10 cells)') |   ('Tdry_rms_K', 'after (real t, 20 cells)') |   ('Tdry_rms_K', 'before (105 mm grid)') |
|:---------|--------------------------------------------:|--------------------------------------------:|----------------------------------------:|---------------------------------------------:|---------------------------------------------:|-----------------------------------------:|
| M03-0003 |                                         0.2 |                                         0.2 |                                     0.4 |                                          0.2 |                                          0.2 |                                      1.1 |
| M04-0003 |                                         2.8 |                                         2.8 |                                     2.8 |                                          0.5 |                                          0.5 |                                      0.6 |
| M05-0003 |                                         0.5 |                                         0.5 |                                     1.2 |                                          0.2 |                                          0.2 |                                      0   |
| M06-0003 |                                         7.1 |                                         7.1 |                                    26.1 |                                         13.7 |                                         13.7 |                                    178.3 |
| M07-0004 |                                         4.6 |                                         4.6 |                                    18.4 |                                          5.5 |                                          5.5 |                                    143.5 |
| M08-0001 |                                         2.7 |                                         2.7 |                                     2.9 |                                          5.9 |                                          5.9 |                                    137.4 |
| M09-0001 |                                         4.6 |                                         4.6 |                                    18.4 |                                          5.5 |                                          5.5 |                                    143.5 |
| M10-0001 |                                         7.9 |                                         7.9 |                                    28.5 |                                          9.7 |                                          9.7 |                                    171.5 |
| M11-0002 |                                        25.6 |                                        25.6 |                                    59.4 |                                         44.3 |                                         44.3 |                                    111.3 |
| M12-0004 |                                         4   |                                         4   |                                     8.7 |                                         12.4 |                                         12.4 |                                    159   |
| M15-0002 |                                         8.6 |                                         8.6 |                                    11.7 |                                          4.9 |                                          5   |                                     14.9 |

## Failed runs

| case     | config                   | error                                                                             |
|:---------|:-------------------------|:----------------------------------------------------------------------------------|
| M01-0027 | before (105 mm grid)     | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M01-0027 | after (real t, 10 cells) | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M01-0027 | after (real t, 20 cells) | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M12-0010 | before (105 mm grid)     | ValueError: math domain error                                                     |
| M12-0010 | after (real t, 10 cells) | ValueError: math domain error                                                     |
| M14-0157 | before (105 mm grid)     | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0157 | after (real t, 10 cells) | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M14-0157 | after (real t, 20 cells) | ValueError: initial state has no vapour phase - liquid-full vessels not supported |
| M16-0033 | before (105 mm grid)     | ValueError: unknown component BNZ (give pseudo properties)                        |
| M16-0033 | after (real t, 10 cells) | ValueError: unknown component BNZ (give pseudo properties)                        |
| M16-0033 | after (real t, 20 cells) | ValueError: unknown component BNZ (give pseudo properties)                        |
