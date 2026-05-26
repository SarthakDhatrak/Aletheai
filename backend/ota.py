"""
OTA — Over-The-Air Hardware Validation Module

Provides tools for configuring consumer Wi-Fi USB dongles into monitor mode,
triggering NDP sounding via lightweight ping floods, and running end-to-end
OTA validation of the Aletheia pipeline using real-world beamforming feedback.

Designed for Linux hosts with standard iw/ip tooling.
Windows/macOS support is limited to adapter detection only.
"""

import os
import sys
import time
import logging
import subprocess
import threading
from typing import Dict, Any, List, Optional
from backend.config import OTA_NDP_TRIGGER_INTERVAL, OTA_CHANNEL, OTA_VALIDATION_DURATION

logger = logging.getLogger("ota")


class OTAConfigurator:
    """
    Manages Wi-Fi adapter configuration for Over-The-Air BFI capture.
    Supports adapter detection, monitor mode enable/disable, and channel setting.
    """

    @staticmethod
    def detect_wifi_adapters() -> List[Dict[str, Any]]:
        """
        Enumerates available Wi-Fi adapters on the system.
        Returns a list of dicts with adapter info.
        """
        adapters = []

        if sys.platform.startswith("linux"):
            try:
                result = subprocess.run(
                    ["iw", "dev"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    current_iface = None
                    current_info = {}
                    for line in result.stdout.splitlines():
                        line = line.strip()
                        if line.startswith("Interface"):
                            if current_iface:
                                adapters.append(current_info)
                            current_iface = line.split()[-1]
                            current_info = {
                                "interface": current_iface,
                                "type": "unknown",
                                "channel": None,
                                "mac": None
                            }
                        elif line.startswith("type"):
                            current_info["type"] = line.split()[-1]
                        elif line.startswith("channel"):
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    current_info["channel"] = int(parts[1])
                                except ValueError:
                                    pass
                        elif line.startswith("addr"):
                            current_info["mac"] = line.split()[-1]
                    if current_iface:
                        adapters.append(current_info)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.warning("'iw' command not found or timed out. Cannot enumerate adapters.")

        elif sys.platform == "win32":
            try:
                result = subprocess.run(
                    ["netsh", "wlan", "show", "interfaces"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    current_info = {}
                    for line in result.stdout.splitlines():
                        line = line.strip()
                        if ":" in line:
                            key, _, value = line.partition(":")
                            key = key.strip().lower()
                            value = value.strip()
                            if "name" in key:
                                if current_info.get("interface"):
                                    adapters.append(current_info)
                                current_info = {"interface": value, "type": "managed", "channel": None, "mac": None}
                            elif "physical" in key and "address" in key:
                                current_info["mac"] = value
                            elif "channel" in key:
                                try:
                                    current_info["channel"] = int(value)
                                except ValueError:
                                    pass
                    if current_info.get("interface"):
                        adapters.append(current_info)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.warning("'netsh' command not found. Cannot enumerate adapters.")

        return adapters

    @staticmethod
    def get_adapter_capabilities(interface: str) -> Dict[str, bool]:
        """
        Checks if an adapter supports monitor mode and packet injection.
        Linux only via 'iw phy' inspection.
        """
        capabilities = {
            "monitor_mode": False,
            "injection": False,
            "vht_support": False
        }

        if not sys.platform.startswith("linux"):
            logger.info("Adapter capability check only supported on Linux.")
            return capabilities

        try:
            # Find the phy device for this interface
            result = subprocess.run(
                ["iw", "dev", interface, "info"],
                capture_output=True, text=True, timeout=5
            )
            phy_name = None
            for line in result.stdout.splitlines():
                if "wiphy" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        phy_name = f"phy{parts[-1]}"
                    break

            if phy_name:
                result = subprocess.run(
                    ["iw", phy_name, "info"],
                    capture_output=True, text=True, timeout=10
                )
                output = result.stdout.lower()
                capabilities["monitor_mode"] = "monitor" in output
                capabilities["vht_support"] = "vht" in output

        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return capabilities

    @staticmethod
    def enable_monitor_mode(interface: str) -> bool:
        """
        Puts a Wi-Fi adapter into monitor mode (Linux only).
        Standard sequence: ip link down → iw set monitor → ip link up
        """
        if not sys.platform.startswith("linux"):
            logger.error("Monitor mode configuration requires Linux.")
            return False

        try:
            cmds = [
                ["sudo", "ip", "link", "set", interface, "down"],
                ["sudo", "iw", interface, "set", "monitor", "control"],
                ["sudo", "ip", "link", "set", interface, "up"],
            ]
            for cmd in cmds:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    logger.error(f"Failed: {' '.join(cmd)} → {result.stderr.strip()}")
                    return False

            logger.info(f"Monitor mode enabled on {interface}")
            return True

        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error(f"Monitor mode setup failed: {e}")
            return False

    @staticmethod
    def disable_monitor_mode(interface: str) -> bool:
        """Restores a Wi-Fi adapter to managed mode (Linux only)."""
        if not sys.platform.startswith("linux"):
            return False

        try:
            cmds = [
                ["sudo", "ip", "link", "set", interface, "down"],
                ["sudo", "iw", interface, "set", "type", "managed"],
                ["sudo", "ip", "link", "set", interface, "up"],
            ]
            for cmd in cmds:
                subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            logger.info(f"Managed mode restored on {interface}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore managed mode: {e}")
            return False

    @staticmethod
    def set_channel(interface: str, channel: int) -> bool:
        """Locks a monitor-mode interface to a specific Wi-Fi channel."""
        if not sys.platform.startswith("linux"):
            return False

        try:
            result = subprocess.run(
                ["sudo", "iw", interface, "set", "channel", str(channel)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Channel set to {channel} on {interface}")
                return True
            else:
                logger.error(f"Failed to set channel: {result.stderr.strip()}")
                return False
        except Exception as e:
            logger.error(f"Channel set failed: {e}")
            return False


class NDPTrigger:
    """
    Generates lightweight background traffic to force Wi-Fi Access Points to send
    Null Data Packets (NDPs) for MU-MIMO sounding, which triggers client devices
    to broadcast standard-compliant BFI frames.
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._process: Optional[subprocess.Popen] = None
        self.mode = "ping"  # "ping" or "iperf"
        self.target_ip: Optional[str] = None
        self.packets_sent: int = 0

    def start_ping_flood(self, target_ip: str, interval: float = OTA_NDP_TRIGGER_INTERVAL):
        """
        Starts a lightweight ping loop to generate traffic that forces
        the router to perform beamforming sounding.
        """
        self.stop()
        self.target_ip = target_ip
        self.mode = "ping"
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._ping_loop, args=(target_ip, interval), daemon=True)
        self._thread.start()
        logger.info(f"NDP trigger started: ping {target_ip} every {interval}s")

    def _ping_loop(self, target_ip: str, interval: float):
        """Internal ping loop running in a background thread."""
        while not self._stop_event.is_set():
            try:
                if sys.platform == "win32":
                    cmd = ["ping", "-n", "1", "-w", "100", target_ip]
                else:
                    cmd = ["ping", "-c", "1", "-W", "1", target_ip]

                subprocess.run(cmd, capture_output=True, timeout=2)
                self.packets_sent += 1
            except (subprocess.TimeoutExpired, Exception):
                pass
            self._stop_event.wait(interval)

    def start_iperf_stream(self, target_ip: str, duration: int = 60):
        """
        Starts a heavier iperf3 traffic stream to force MU-MIMO beamforming.
        Requires iperf3 server running on the target.
        """
        self.stop()
        self.target_ip = target_ip
        self.mode = "iperf"
        self._stop_event.clear()

        try:
            self._process = subprocess.Popen(
                ["iperf3", "-c", target_ip, "-t", str(duration), "-b", "1M"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            logger.info(f"iperf3 stream started to {target_ip}")
        except FileNotFoundError:
            logger.error("iperf3 not found. Install it or use ping mode.")

    def stop(self):
        """Stops the NDP trigger."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if self._process:
            self._process.terminate()
            self._process = None
        logger.info("NDP trigger stopped.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "active": not self._stop_event.is_set() and self._thread is not None,
            "mode": self.mode,
            "target_ip": self.target_ip,
            "packets_sent": self.packets_sent
        }


def validate_ota_pipeline(
    interface: str,
    duration: float = OTA_VALIDATION_DURATION,
    channel: int = OTA_CHANNEL,
    trigger_ip: Optional[str] = None
) -> Dict[str, Any]:
    """
    End-to-end OTA validation: enables monitor mode, optionally starts NDP trigger,
    captures packets for the specified duration, parses BFI frames, and reports results.

    Args:
        interface: Wi-Fi adapter interface name
        duration: Capture duration in seconds
        channel: Wi-Fi channel to monitor
        trigger_ip: Optional IP to ping for NDP triggering

    Returns:
        Validation report dict
    """
    report = {
        "interface": interface,
        "channel": channel,
        "duration_sec": duration,
        "status": "not_started",
        "frames_captured": 0,
        "bfi_frames_parsed": 0,
        "parse_success_rate": 0.0,
        "unique_clients": [],
        "errors": []
    }

    # 1. Configure adapter
    configurator = OTAConfigurator()
    caps = configurator.get_adapter_capabilities(interface)
    if not caps["monitor_mode"]:
        report["status"] = "error"
        report["errors"].append("Adapter does not support monitor mode")
        return report

    if not configurator.enable_monitor_mode(interface):
        report["status"] = "error"
        report["errors"].append("Failed to enable monitor mode")
        return report

    if not configurator.set_channel(interface, channel):
        report["errors"].append(f"Failed to set channel {channel}")

    # 2. Start NDP trigger if IP provided
    trigger = None
    if trigger_ip:
        trigger = NDPTrigger()
        trigger.start_ping_flood(trigger_ip)

    # 3. Capture packets
    try:
        from scapy.all import sniff as scapy_sniff
        from backend.parser import parse_bfi_packet

        captured_packets = []
        clients_seen = set()

        def packet_handler(pkt):
            captured_packets.append(pkt)
            parsed = parse_bfi_packet(pkt)
            if parsed:
                clients_seen.add(parsed.get("src", "unknown"))

        logger.info(f"Starting OTA capture on {interface} for {duration}s...")
        scapy_sniff(
            iface=interface,
            prn=packet_handler,
            timeout=duration,
            store=0
        )

        # 4. Parse results
        bfi_count = 0
        for pkt in captured_packets:
            parsed = parse_bfi_packet(pkt)
            if parsed:
                bfi_count += 1

        report["frames_captured"] = len(captured_packets)
        report["bfi_frames_parsed"] = bfi_count
        report["parse_success_rate"] = bfi_count / max(1, len(captured_packets))
        report["unique_clients"] = list(clients_seen)
        report["status"] = "completed"

    except Exception as e:
        report["status"] = "error"
        report["errors"].append(str(e))

    # 5. Cleanup
    if trigger:
        trigger.stop()
    configurator.disable_monitor_mode(interface)

    logger.info(f"OTA validation complete: {report['bfi_frames_parsed']} BFI frames parsed "
                f"from {report['frames_captured']} total packets")
    return report
