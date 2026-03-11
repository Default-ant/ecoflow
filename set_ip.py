"""
EcoFlow AI — IP Webcam address updater
Run:  python set_ip.py
"""
import re, pathlib

TARGET = pathlib.Path(__file__).parent / "ecoflow_ai.py"
IP_RE  = re.compile(r'(default="http://)[\d.]+(:8080/video")')

current = TARGET.read_text()
match   = IP_RE.search(current)
if match:
    old_ip = re.search(r'[\d.]+', match.group(0)).group()
    print(f"Current IP: {old_ip}")
else:
    print("Could not find current IP in ecoflow_ai.py")

new_ip = input("Enter new IP address (e.g. 192.168.1.42): ").strip()
if not re.match(r'^\d+\.\d+\.\d+\.\d+$', new_ip):
    print("Invalid IP format. Nothing changed.")
else:
    updated = IP_RE.sub(rf'\g<1>{new_ip}\g<2>', current)
    TARGET.write_text(updated)
    print(f"✅  IP updated to {new_ip} in ecoflow_ai.py")
