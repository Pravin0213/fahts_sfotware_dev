# Calibration re-check: calibration_recheck

One option changed at a time from the baseline profile (VesselFireOptions defaults),
run only on the cases it can affect. Cells: median baseline -> option over those cases
(lower = closer to VessFire). `P better / worse`: cases whose pressure RMS improves /
degrades by more than 0.5 %-points. Metric definitions: `compare_cases.py`.

Baseline over all 73 cases: P_rms_pct 4.6, Tgas_rms_K 16.3, Tdry_rms_K 6.2, Twet_rms_K 11.7, Tliq_rms_K 6.5

| option                         | affects   |   cases | P_rms_pct   | Tgas_rms_K   | Tdry_rms_K   | Twet_rms_K   | Tliq_rms_K   | rupt_Tr_err_s   | rupt_vM_err_s   | P better / worse   |
|:-------------------------------|:----------|--------:|:------------|:-------------|:-------------|:-------------|:-------------|:----------------|:----------------|:-------------------|
| wet_above_crit=single-phase    | liquid    |      43 | 7.0 -> 8.6  | 18.7 -> 18.7 | 22.4 -> 22.5 | 11.7 -> 14.7 | 6.5 -> 8.1   | 141.5 -> 266.5  | 147.5 -> 226.0  | 7 / 7              |
| wet_above_crit=supercritical   | liquid    |      43 | 7.0 -> 8.6  | 18.7 -> 18.7 | 22.4 -> 22.5 | 11.7 -> 14.8 | 6.5 -> 8.1   | 141.5 -> 281.5  | 147.5 -> 226.0  | 9 / 7              |
| wet_boiling=full               | liquid    |      43 | 7.0 -> 8.5  | 18.7 -> 18.7 | 22.4 -> 22.5 | 11.7 -> 13.7 | 6.5 -> 8.1   | 141.5 -> 221.0  | 147.5 -> 182.5  | 9 / 7              |
| liq_grashof=beta               | liquid    |      43 | 7.0 -> 7.0  | 18.7 -> 18.7 | 22.4 -> 22.4 | 11.7 -> 11.7 | 6.5 -> 6.5   | 141.5 -> 141.5  | 147.5 -> 147.5  | 0 / 1              |
| interface_mass=True            | liquid    |      43 | 7.0 -> 6.5  | 18.7 -> 18.9 | 22.4 -> 22.4 | 11.7 -> 11.7 | 6.5 -> 7.2   | 141.5 -> 266.0  | 147.5 -> 266.0  | 5 / 6              |
| h_corr=churchill_chu           | fire      |      45 | 7.9 -> 10.0 | 30.4 -> 38.6 | 22.5 -> 31.2 | 20.2 -> 23.7 | 19.4 -> 18.4 | 147.0 -> 113.0  | 98.0 -> 81.0    | 13 / 26            |
| h_corr=churchill_chu (no fire) | bdv_line  |      31 | 8.6 -> 12.1 | 17.6 -> 31.3 | 10.5 -> 11.0 | 12.5 -> 20.7 | 8.1 -> 7.6   |                 |                 | 8 / 17             |
| rad_internal=False             | fire      |      45 | 7.9 -> 8.3  | 30.4 -> 28.9 | 22.5 -> 13.7 | 20.2 -> 27.1 | 19.4 -> 19.1 | 147.0 -> 121.0  | 98.0 -> 88.0    | 8 / 14             |
| water_mode=physical            | water     |       3 | 3.2 -> 3.2  | 5.7 -> 5.3   | 3.7 -> 3.8   | 31.5 -> 31.5 | 1.4 -> 9.6   | 328.0 -> 258.0  | 279.0 -> 189.0  | 1 / 0              |
| water_mode=sink                | water     |       4 | 3.6 -> 3.1  | 23.1 -> 23.1 | 8.1 -> 8.1   | 56.3 -> 57.3 | 1.2 -> 0.9   | 328.0 -> 328.0  | 279.0 -> 279.0  | 1 / 0              |
| psv_liquid=gas                 | psv_flow  |      19 | 7.0 -> 11.3 | 27.6 -> 34.4 | 33.9 -> 37.6 | 23.9 -> 24.3 | 27.9 -> 28.6 | 171.0 -> 171.0  | 121.0 -> 121.0  | 2 / 4              |
| flux=balance                   | fire      |      45 | 7.9 -> 4.6  | 30.4 -> 23.0 | 22.5 -> 46.9 | 20.2 -> 37.0 | 19.4 -> 16.9 | 147.0 -> 52.5   | 98.0 -> 82.0    | 28 / 12            |
| eps_surf_fire=0.85             | fire      |      45 | 7.9 -> 4.9  | 30.4 -> 35.0 | 22.5 -> 47.9 | 20.2 -> 50.7 | 19.4 -> 29.9 | 147.0 -> 119.0  | 98.0 -> 132.0   | 23 / 15            |
| line_diameter=inner            | bdv_line  |      31 | 8.6 -> 9.1  | 17.6 -> 16.3 | 10.5 -> 10.5 | 12.5 -> 12.5 | 8.1 -> 8.1   |                 |                 | 1 / 1              |
