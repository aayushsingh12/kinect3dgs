import glob
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(script, *args, env=None):
    e = {**os.environ, **(env or {})}
    return subprocess.run(["bash", str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, env=e)


def test_bash_syntax():
    for s in glob.glob(str(ROOT / "scripts" / "*.sh")):
        assert subprocess.run(["bash", "-n", s]).returncode == 0, s


def fake_usb(tmp_path, pids):
    for i, pid in enumerate(pids):
        d = tmp_path / f"1-{i}"
        d.mkdir()
        (d / "idVendor").write_text("045e\n")
        (d / "idProduct").write_text(pid + "\n")
    return {"KINECT3DGS_SYSFS_USB": str(tmp_path)}


def test_usb_list_missing_and_found(tmp_path):
    r = run("usb_list.sh", env=fake_usb(tmp_path, ["02c2"]))
    assert r.returncode == 1 and "camera: MISSING" in r.stdout
    t2 = tmp_path / "b"
    t2.mkdir()
    r = run("usb_list.sh", env=fake_usb(t2, ["02ae", "02ad"]))
    assert r.returncode == 0 and "camera: FOUND" in r.stdout


def test_system_setup_check_empty_etc(tmp_path):
    r = run("system_setup.sh", "--check", env={"KINECT3DGS_ETC": str(tmp_path)})
    assert r.returncode == 1
    assert "MISSING  udev rule" in r.stdout and "MISSING  modprobe blacklist" in r.stdout


def test_system_setup_check_ok(tmp_path):
    rules = tmp_path / "udev/rules.d"
    rules.mkdir(parents=True)
    src = ROOT / "third_party/libfreenect/platform/linux/udev/51-kinect.rules"
    (rules / "51-kinect.rules").write_bytes(src.read_bytes())
    mp = tmp_path / "modprobe.d"
    mp.mkdir()
    (mp / "kinect3dgs-blacklist.conf").write_text("blacklist gspca_kinect\ninstall gspca_kinect /bin/false\n")
    r = run("system_setup.sh", "--check", env={"KINECT3DGS_ETC": str(tmp_path)})
    assert "OK       udev rule" in r.stdout and "OK       modprobe blacklist" in r.stdout


def test_system_setup_apply_refuses_without_root(tmp_path):
    if os.getuid() == 0:
        return
    assert run("system_setup.sh", env={"KINECT3DGS_ETC": str(tmp_path)}).returncode == 1
