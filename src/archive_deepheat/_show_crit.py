import json

for t in ("slope005", "slope007", "relax15"):
    j = json.load(open(f"C:/Users/zhen.luo/ocean_solver/_spinC_{t}_ana.json"))
    c = j["criteria"]
    print(t, "b:", round(c["b_deept_trend"]["global_C_per_yr"], 5),
          "d:", round(c["d_ohc_trend"]["ZJ_per_yr"], 2),
          "amoc:", round(c["f_amoc_sanity"]["amoc_final_sv"], 1),
          "ratio:", round(c["f_amoc_sanity"]["amoc26_ratio"], 3),
          "warnings:", len(c["f_amoc_sanity"]["warn"]))
