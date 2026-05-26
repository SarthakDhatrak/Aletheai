import time
import math
import random
import logging
import threading
from typing import Callable, Dict, Any, Optional
from scapy.all import sniff
from backend.config import (
    DEFAULT_INTERFACE, SIMULATOR_TICK_RATE_HZ, SIM_PACKET_LOSS_RATE,
    SHIELD_VERSION, SHIELD_NUM_TAPS, OTA_CHANNEL, OTA_NDP_TRIGGER_INTERVAL
)
from backend.simulator import BFISimulator
from backend.parser import parse_bfi_packet, parse_raw_bfi_payload

logger = logging.getLogger("bfi_sniffer")

class BFISniffer:
    def __init__(self, interface: str = DEFAULT_INTERFACE):
        self.interface = interface
        self.mode = "simulation"  # "simulation", "live", or "ota"
        self.simulator = BFISimulator()
        self.callback: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Calibration & Shield configuration state
        self.layout_distance = 4.0
        self.layout_azimuth = 0.0
        self.layout_height = 1.5
        self.bf_format = "vht"
        self.shield_active = False
        self.shield_seed = 42
        self.shield_authorized = True
        self.shield_version = SHIELD_VERSION  # 1 = legacy, 2 = multi-tap
        self.shield_num_taps = SHIELD_NUM_TAPS  # Configurable tap count for Shield v2
        
        # OTA configuration
        self.ota_channel = OTA_CHANNEL
        self.ota_trigger_ip: Optional[str] = None
        self.ota_trigger_interval = OTA_NDP_TRIGGER_INTERVAL
        self._ndp_trigger = None
        
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def set_mode(self, mode: str):
        if mode not in ("simulation", "live", "ota"):
            return
        self.mode = mode
        logger.info(f"Sniffer mode set to: {mode}")

    def set_simulator_state(self, state: str):
        self.simulator.set_state(state)
        logger.info(f"Simulator state set to: {state}")

    def start(self, callback: Callable[[Dict[str, Any]], None]):
        if self.running:
            self.stop()
            
        self.callback = callback
        self.running = True
        self._stop_event.clear()
        
        if self.mode == "simulation":
            self._thread = threading.Thread(target=self._run_simulation, daemon=True)
        elif self.mode == "ota":
            self._thread = threading.Thread(target=self._run_ota, daemon=True)
        else:
            self._thread = threading.Thread(target=self._run_live, daemon=True)
            
        self._thread.start()
        logger.info(f"Sniffer started in {self.mode} mode.")

    def stop(self):
        if not self.running:
            return
            
        self.running = False
        self._stop_event.set()
        
        # Stop NDP trigger if active
        if self._ndp_trigger:
            self._ndp_trigger.stop()
            self._ndp_trigger = None
        
        if self._thread:
            # We don't block forever if live sniff doesn't exit immediately
            self._thread.join(timeout=2.0)
            self._thread = None
            
        logger.info("Sniffer stopped.")

    def _run_simulation(self):
        dt = 1.0 / SIMULATOR_TICK_RATE_HZ
        next_tick = time.time()
        
        while not self._stop_event.is_set() and self.running:
            now = time.time()
            if now < next_tick:
                time.sleep(max(0.001, next_tick - now))
                continue
                
            next_tick = now + dt
            
            # Set layout variables on simulator before updating physics
            self.simulator.layout_distance = self.layout_distance
            self.simulator.layout_azimuth = self.layout_azimuth
            self.simulator.layout_height = self.layout_height
            
            # Physics update
            self.simulator.update_physics(dt)
            
            # Simulate packet loss
            if random.random() < SIM_PACKET_LOSS_RATE:
                continue
            
            # Generate dummy standard action packet payload
            payload = self.simulator.generate_packet_payload(
                bf_format=self.bf_format,
                shield_active=self.shield_active,
                shield_seed=self.shield_seed,
                shield_version=self.shield_version,
                shield_num_taps=self.shield_num_taps
            )
            
            # Parse it
            parsed = parse_raw_bfi_payload(payload)
            if parsed and self.callback:
                # Add synthetic metadata
                parsed["src"] = "00:0a:95:9d:68:16"  # Mock client (iPhone)
                parsed["dst"] = "ac:8b:cd:15:3e:22"  # Mock router (Netgear)
                parsed["bssid"] = "ac:8b:cd:15:3e:22"
                parsed["snr"] = int(-45 + 5 * math.sin(self.simulator.time_offset * 0.1) + random.uniform(-2, 2))
                parsed["timestamp"] = time.time()
                
                try:
                    self.callback(parsed)
                except Exception as e:
                    logger.error(f"Callback error in sniffer: {e}")

    def _run_live(self):
        logger.info(f"Listening on monitor interface {self.interface} using Scapy...")
        
        def scapy_callback(pkt):
            if not self.running or self._stop_event.is_set():
                return
            try:
                parsed = parse_bfi_packet(pkt)
                if parsed and self.callback:
                    parsed["timestamp"] = time.time()
                    self.callback(parsed)
            except Exception as e:
                logger.error(f"Error parsing sniffed packet: {e}")

        def stop_filter(pkt):
            return self._stop_event.is_set() or not self.running

        try:
            sniff(
                iface=self.interface,
                prn=scapy_callback,
                stop_filter=stop_filter,
                store=0
            )
        except Exception as e:
            logger.error(f"Live sniffing failed on {self.interface}: {e}")
            logger.info("Falling back to simulation mode.")
            self.mode = "simulation"
            self._run_simulation()

    def _run_ota(self):
        """
        OTA mode: enables monitor mode, starts NDP trigger, and captures
        real BFI frames from over-the-air Wi-Fi traffic.
        """
        from backend.ota import OTAConfigurator, NDPTrigger

        logger.info(f"Starting OTA capture on {self.interface}, channel {self.ota_channel}...")

        # 1. Configure adapter
        configurator = OTAConfigurator()
        if not configurator.enable_monitor_mode(self.interface):
            logger.error("Failed to enable monitor mode. Falling back to simulation.")
            self.mode = "simulation"
            self._run_simulation()
            return

        configurator.set_channel(self.interface, self.ota_channel)

        # 2. Start NDP trigger if target IP provided
        if self.ota_trigger_ip:
            self._ndp_trigger = NDPTrigger()
            self._ndp_trigger.start_ping_flood(
                self.ota_trigger_ip,
                interval=self.ota_trigger_interval
            )

        # 3. Start live capture (same as live mode but with OTA setup)
        def scapy_callback(pkt):
            if not self.running or self._stop_event.is_set():
                return
            try:
                parsed = parse_bfi_packet(pkt)
                if parsed and self.callback:
                    parsed["timestamp"] = time.time()
                    # Extract real RadioTap metadata if available
                    if hasattr(pkt, "dBm_AntSignal"):
                        parsed["snr"] = pkt.dBm_AntSignal
                    if hasattr(pkt, "ChannelFrequency"):
                        parsed["channel_freq"] = pkt.ChannelFrequency
                    self.callback(parsed)
            except Exception as e:
                logger.error(f"OTA parse error: {e}")

        def stop_filter(pkt):
            return self._stop_event.is_set() or not self.running

        try:
            sniff(
                iface=self.interface,
                prn=scapy_callback,
                stop_filter=stop_filter,
                store=0
            )
        except Exception as e:
            logger.error(f"OTA sniffing failed: {e}")
        finally:
            # Cleanup
            if self._ndp_trigger:
                self._ndp_trigger.stop()
                self._ndp_trigger = None
            configurator.disable_monitor_mode(self.interface)
