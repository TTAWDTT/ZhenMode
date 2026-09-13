import psutil, time, datetime
for pr in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_times']):
    try:
        cl = pr.info['cmdline'] or []
    except Exception:
        continue
    s = " ".join(str(x) for x in cl)
    if '_prodAB' in s:
        el = time.time() - pr.info['create_time']
        ct = pr.info['cpu_times']
        print("pid", pr.info['pid'])
        print("elapsed  %.1f min" % (el / 60))
        print("cpu time %.1f min  (%.0f%% of one core avg)"
              % (ct.user + ct.system) / 60 if False else
              "cpu time %.1f min" % ((ct.user + ct.system) / 60))
        print("started", datetime.datetime.fromtimestamp(pr.info['create_time']))
