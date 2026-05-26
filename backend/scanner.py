import os
import subprocess
import threading
from backend.bfld import bfld_layer

class NetworkScanner:
    def __init__(self, subnet_prefix="192.168.139"):
        """
        A lightweight network scanner that uses ping sweeps to populate the ARP cache,
        then reads the ARP table to discover devices without requiring root/sudo privileges.
        """
        self.subnet_prefix = subnet_prefix
        
    def _ping_host(self, ip):
        # Send a single ping with a fast timeout (0.2s) to populate ARP
        subprocess.run(["ping", "-c", "1", "-W", "0.2", ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    def sweep_network(self):
        """Pings all 255 addresses in the subnet concurrently."""
        threads = []
        for i in range(1, 255):
            ip = f"{self.subnet_prefix}.{i}"
            t = threading.Thread(target=self._ping_host, args=(ip,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
    def get_connected_devices(self):
        """
        Runs the sweep and reads the ARP table.
        Returns a list of dicts with IP and anonymized MAC addresses.
        """
        # 1. Sweep to discover new devices
        self.sweep_network()
        
        # 2. Read ARP table
        devices = []
        try:
            with open("/proc/net/arp", "r") as f:
                lines = f.readlines()[1:] # skip header
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 4:
                        ip = parts[0]
                        mac = parts[3]
                        if mac != "00:00:00:00:00:00":
                            # Privacy Guard: Hash the physical MAC before it leaves the backend
                            anonymized_mac = bfld_layer.anonymize(mac)
                            devices.append({
                                "ip": ip,
                                "mac_anonymized": anonymized_mac
                            })
        except Exception as e:
            print(f"[Scanner] Error reading ARP table: {e}")
            
        return devices

# Singleton instance
# To make it robust across environments, we can guess the subnet prefix dynamically
def guess_subnet_prefix():
    try:
        # Simple extraction of local IP to get subnet C-class
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        parts = local_ip.split(".")
        return f"{parts[0]}.{parts[1]}.{parts[2]}"
    except:
        return "192.168.1"

import socket
scanner = NetworkScanner(subnet_prefix=guess_subnet_prefix())
