import hashlib
import datetime
import os

class BFLDPrivacyLayer:
    """
    BFLD (Beamforming Feedback Layer for Detection) Privacy Module
    Inspired by RuView. Cryptographically scrambles MAC addresses using a 
    daily rotating salt to prevent identity tracking and cross-site correlation.
    """
    def __init__(self):
        self._secret_key = os.environ.get("BFLD_SECRET", "aletheia_bfld_secret_key")

    def _get_daily_salt(self) -> bytes:
        # The salt rotates every day at UTC midnight
        date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        return f"{self._secret_key}_{date_str}".encode('utf-8')

    def anonymize_mac(self, mac_address: str) -> str:
        if not mac_address or mac_address.startswith("BFLD-"):
            return mac_address
        
        # Standardize MAC address format
        mac_standard = mac_address.upper().strip()
        
        # Hash with daily salt
        salt = self._get_daily_salt()
        data_to_hash = mac_standard.encode('utf-8') + salt
        
        # SHA-256 Digest
        hash_digest = hashlib.sha256(data_to_hash).hexdigest().upper()
        
        # Return a short privacy identifier
        return f"BFLD-{hash_digest[:8]}"

# Singleton instance
bfld_layer = BFLDPrivacyLayer()
