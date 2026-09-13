import json
j = json.load(open("results/campaign_compare/tenyr_g3650d_016.json"))
def walk(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                walk(v, pre + k + ".")
            else:
                print("%-46s %s" % (pre + k, v))
    elif isinstance(o, list):
        print("%-46s [list n=%d] %s" % (pre, len(o), o[:8]))
walk(j)
