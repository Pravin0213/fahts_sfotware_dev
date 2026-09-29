"""Heat transfer from the inner wall into the vessel contents, and across the interface.

Heat flux q > 0 means heat flows from the wall INTO the fluid. Property dicts (SI):
  single-phase film/bulk props  : {"rho","cp","mu","k","beta"}   (beta = 1/v dv/dT)
  saturation props at pressure P: {"T_sat","rho_l","rho_v","h_l","h_v","h_fg",
                                   "sigma","mu_l","mu_v","k_l","k_v","cp_l","cp_v",
                                   "P","P_c","M"}  (M in kg/kmol)
"""
