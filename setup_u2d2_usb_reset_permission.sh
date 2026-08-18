#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-/dev/book_hand}"
RULE_FILE="/etc/udev/rules.d/99-book-hand-usb-reset.rules"

if [[ ${EUID} -ne 0 ]]; then
  echo "Run with sudo:"
  echo "  sudo bash $0 ${PORT}"
  exit 1
fi

python3 - "$PORT" "$RULE_FILE" <<'PY'
import grp
import os
import pwd
import sys
from pathlib import Path

port = Path(sys.argv[1])
rule_file = Path(sys.argv[2])

if not port.exists():
    raise SystemExit(f"[ERROR] {port} does not exist")

real_port = port.resolve(strict=True)
tty_name = real_port.name
sysdev = (Path('/sys/class/tty') / tty_name / 'device').resolve()

usb = None
cur = sysdev
while True:
    if (cur / 'busnum').exists() and (cur / 'devnum').exists():
        usb = cur
        break
    if cur.parent == cur:
        break
    cur = cur.parent

if usb is None:
    raise SystemExit(f"[ERROR] USB parent not found from {real_port}")

def read(name: str) -> str:
    p = usb / name
    return p.read_text().strip() if p.exists() else ''

vendor = read('idVendor')
product = read('idProduct')
serial = read('serial')
busnum = int(read('busnum'))
devnum = int(read('devnum'))
usbfs = Path('/dev/bus/usb') / f'{busnum:03d}' / f'{devnum:03d}'

if not vendor or not product:
    raise SystemExit('[ERROR] idVendor/idProduct could not be read')

parts = [
    'SUBSYSTEM=="usb"',
    f'ATTR{{idVendor}}=="{vendor}"',
    f'ATTR{{idProduct}}=="{product}"',
]
if serial:
    parts.append(f'ATTR{{serial}}=="{serial}"')
parts += ['MODE="0660"', 'GROUP="dialout"']
rule = ', '.join(parts) + '\n'

rule_file.write_text(rule)
os.chmod(rule_file, 0o644)

print('[U2D2 USB RESET SETUP]')
print(f'  port      : {port} -> {real_port}')
print(f'  sysfs     : {usb}')
print(f'  bus_id    : {usb.name}')
print(f'  VID:PID   : {vendor}:{product}')
print(f'  serial    : {serial or "N/A"}')
print(f'  usbfs     : {usbfs}')
print(f'  rule      : {rule_file}')
print(f'  rule text : {rule.strip()}')

# Current device gets permission immediately; future reconnects are handled by udev.
if usbfs.exists():
    try:
        gid = grp.getgrnam('dialout').gr_gid
        os.chown(usbfs, -1, gid)
        os.chmod(usbfs, 0o660)
        print(f'  current node permission updated: {usbfs}')
    except Exception as exc:
        print(f'  [WARN] current node chmod/chgrp failed: {exc}')

sudo_user = os.environ.get('SUDO_USER', '')
if sudo_user and sudo_user != 'root':
    try:
        user = pwd.getpwnam(sudo_user)
        groups = {g.gr_name for g in grp.getgrall() if sudo_user in g.gr_mem}
        primary = grp.getgrgid(user.pw_gid).gr_name
        groups.add(primary)
        if 'dialout' not in groups:
            print(f'  [WARN] {sudo_user} is not in dialout group.')
            print(f'         Run: sudo usermod -aG dialout {sudo_user}')
            print('         Then log out/in once.')
        else:
            print(f'  user      : {sudo_user} (dialout OK)')
    except Exception as exc:
        print(f'  [WARN] user group check failed: {exc}')
PY

udevadm control --reload-rules

echo
echo "Setup complete. Check with:"
echo "  ls -l /dev/book_hand"
echo "  readlink -f /dev/book_hand"
echo "  ls -l /dev/bus/usb/*/* | tail"