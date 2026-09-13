import psutil
tot = 0
for pr in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
    try:
        cl = pr.info['cmdline'] or []
    except Exception:
        continue
    s = " ".join(str(x) for x in cl)
    if 'python' in (pr.info['name'] or '').lower() and '.py' in s:
        try:
            cpu = psutil.Process(pr.info['pid']).cpu_percent(interval=0.4)
        except Exception:
            cpu = -1
        mb = pr.info['memory_info'].rss / 1e6 if pr.info['memory_info'] else 0
        print("%-7s pid=%-6d cpu%%=%-7.1f rss=%7.0fMB  %s"
              % (pr.info['name'], pr.info['pid'], cpu, mb, s[:110]))
        tot += 1
print("python .py procs:", tot)
