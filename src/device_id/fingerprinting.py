import hashlib
import re

class IdentityProfiler:
    """
    Identifies device class using MAC OUI, JA3/JA4 TLS hash, and DHCP option list.
    Applies majority voting and falls back to Generic IoT on ambiguity.
    """
    def __init__(self):
        # Database mappings
        # MAC OUI mappings (first 3 bytes of MAC)
        self.mac_oui_db = {
            "00:1a:2b": "Camera",
            "00:04:a3": "Thermostat",
            "00:1e:c0": "SmartPlug",
            "3c:d9:2b": "SmartSpeaker",
            "e0:76:d0": "SmartLight"
        }
        
        # JA3/JA4 fingerprint mappings
        # (MD5 of TLS client hello: e.g. SSLVersion,CipherSuites,Extensions,ECCurves,ECCurveFormats)
        self.ja3_db = {
            hashlib.md5(b"771,49195-49199,65281-0-11-35,23-24,0").hexdigest(): "Camera",
            hashlib.md5(b"771,49187-49191,10-11-13,23,0").hexdigest(): "Thermostat",
            hashlib.md5(b"771,49195-49200-49161,65281-0-11,23,0").hexdigest(): "SmartPlug",
            hashlib.md5(b"771,52392-52393,0-23-65281,23-24,0").hexdigest(): "SmartSpeaker",
            hashlib.md5(b"769,47-53,10-11,23,0").hexdigest(): "SmartLight"
        }
        
        # DHCP Option Fingerprints (ordered string of DHCP options)
        self.dhcp_db = {
            "1,3,6,15,28,42": "Camera",
            "1,3,6,12,15,26": "Thermostat",
            "1,3,6,15,119": "SmartPlug",
            "1,3,6,15,121,249": "SmartSpeaker",
            "1,3,6": "SmartLight"
        }

        # Device class name to ID mapping
        self.class_name_to_id = {
            "Generic": 0,
            "Camera": 1,
            "Thermostat": 2,
            "SmartPlug": 3,
            "SmartSpeaker": 4,
            "SmartLight": 5
        }

    def clean_mac(self, mac: str) -> str:
        if not mac:
            return ""
        return mac.lower().replace("-", ":")

    def lookup_mac(self, mac: str) -> str | None:
        cleaned = self.clean_mac(mac)
        match = re.match(r"^([0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2})", cleaned)
        if match:
            oui = match.group(1)
            return self.mac_oui_db.get(oui, None)
        return None

    def lookup_ja3(self, tls_hello_str: str) -> str | None:
        if not tls_hello_str:
            return None
        # Hash the TLS hello string to match JA3 structure
        h = hashlib.md5(tls_hello_str.encode('utf-8')).hexdigest()
        return self.ja3_db.get(h, None)

    def lookup_dhcp(self, option_list_str: str) -> str | None:
        if not option_list_str:
            return None
        # Clean spacing
        cleaned = ",".join([o.strip() for o in option_list_str.split(",") if o.strip()])
        return self.dhcp_db.get(cleaned, None)

    def profile_device(self, mac: str = None, tls_hello_str: str = None, 
                       dhcp_options: str = None) -> tuple[int, bool]:
        """
        Profiles device based on available signals using majority voting.
        
        Returns:
            device_class_id: int index of the identified device class
            is_ambiguous: bool indicating if a majority could not be reached (fallback to Generic)
        """
        votes = []
        
        # 1. MAC Lookup
        mac_class = self.lookup_mac(mac)
        if mac_class:
            votes.append(mac_class)
            
        # 2. TLS Lookup
        tls_class = self.lookup_ja3(tls_hello_str)
        if tls_class:
            votes.append(tls_class)
            
        # 3. DHCP Lookup
        dhcp_class = self.lookup_dhcp(dhcp_options)
        if dhcp_class:
            votes.append(dhcp_class)
            
        if not votes:
            # Ambiguous: Fallback to Generic IoT
            return self.class_name_to_id["Generic"], True
            
        # Perform majority vote
        counts = {}
        for vote in votes:
            counts[vote] = counts.get(vote, 0) + 1
            
        # Find maximum vote count
        max_count = max(counts.values())
        candidates = [k for k, v in counts.items() if v == max_count]
        
        # If there is a tie or no clear candidate with > 50% voting confidence
        if len(candidates) > 1 or max_count < 2 and len(votes) >= 3:
            return self.class_name_to_id["Generic"], True
            
        winner = candidates[0]
        return self.class_name_to_id[winner], False
