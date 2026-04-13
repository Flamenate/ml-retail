import ipaddress
import bisect
import pandas as pd
from pathlib import Path

DBIP_CSV_PATH = Path.cwd() / 'data' / 'raw' / 'dbip-country-lite-2026-02.csv'

def is_private_ip(ip_str):
    """Returns True if IP is private/loopback/reserved, False if public, None if malformed."""
    try:
        return ipaddress.ip_address(str(ip_str).strip()).is_private
    except ValueError:
        return None  # malformed or missing IP

def load_dbip(csv_path):
    """Load DB-IP Lite CSV and convert IP ranges to integers for binary search.
    CSV format: ip_start, ip_end, country_code (no header)"""
    db = pd.read_csv(csv_path, header=None, names=['ip_start', 'ip_end', 'country'])
    db['ip_start_int'] = db['ip_start'].apply(lambda ip: int(ipaddress.ip_address(ip)))
    db['ip_end_int']   = db['ip_end'].apply(lambda ip: int(ipaddress.ip_address(ip)))
    db = db.sort_values('ip_start_int').reset_index(drop=True)
    return db

def lookup_country(ip_str, starts, ends, countries):
    """Binary search over sorted IP ranges to find the country of an IP.
    Returns ISO country code, 'Private', or 'Unknown'."""
    try:
        ip = ipaddress.ip_address(str(ip_str).strip())
        if ip.is_private:
            return 'Private'
        ip_int = int(ip)
        # Find the last range start <= ip_int
        idx = bisect.bisect_right(starts, ip_int) - 1
        if idx >= 0 and ip_int <= ends[idx]:
            return countries[idx]
        return 'Unknown'
    except Exception:
        return 'Unknown'