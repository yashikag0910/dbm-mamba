import os
import glob
import ipaddress
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.preprocessing import log_compact, ZScoreNormalizer

# Mapping for CICIoT2023 attack classes to integer labels based on exact folder names
CICIOT_CLASS_MAP = {
    'Benign_Final': 0,
    'DDoS-RSTFINFLOOD': 1,
    'DDoS-ACK_Fragmentation': 2,
    'DDoS-UDP_Fragmentation': 3,
    'DDoS-ICMP_Fragmentation': 4,
    'DDoS-HTTP_Flood': 5,
    'DDoS-SynonymousIP_Flood': 6,
    'DDoS-SYN_Flood': 7,
    'DDoS-UDP_Flood': 8,
    'DDoS-TCP_Flood': 9,
    'DDoS-ICMP_Flood': 10,
    'DDoS-SlowLoris': 11,
    'DDoS-PSHACK_FLOOD': 12,
    'DoS-UDP_Flood': 13,
    'DoS-SYN_Flood': 14,
    'DoS-TCP_Flood': 15,
    'DoS-HTTP_Flood': 16,
    'DNS_Spoofing': 17,
    'MITM-ArpSpoofing': 18,
    'Recon-HostDiscovery': 19,
    'Recon-OSScan': 20,
    'Recon-PortScan': 21,
    'Recon-PingSweep': 22,
    'VulnerabilityScan': 23,
    'SqlInjection': 24,
    'CommandInjection': 25,
    'XSS': 26,
    'DictionaryBruteForce': 27,
    'Mirai-greeth_flood': 28,
    'Mirai-udpplain': 29,
    'Mirai-greip_flood': 30,
    'BrowserHijacking': 31,
    'Backdoor_Malware': 32,
    'Uploading_Attack': 33
}

# Mapping for ToN_IoT attack classes
TON_IOT_CLASS_MAP = {
    'normal': 0,
    'scanning': 1,
    'ddos': 2,
    'dos': 3,
    'injection': 4,
    'password': 5,
    'ransomware': 6,
    'xss': 7,
    'mitm': 8,
    'backdoor': 9
}

# Mapping for IoT-23 attack classes
IOT23_CLASS_MAP = {
    'benign': 0,
    'partofahorizontalportscan': 1,
    'c&c': 2,
    'attack': 3,
    'okiru': 4,
    'c&c-heartbeat': 5,
    'c&c-mirai': 6,
    'filedownload': 7,
    'c&c-heartbeat-attack': 8,
    'c&c-filedownload': 9,
    'c&c-heartbeat-filedownload': 10,
    'c&c-partofahorizontalportscan': 11
}

# ---------------------------------------------------------------------------
# Device identity.
#
# Device class is the flow's own network identity (source IP), assigned per
# flow by parse_iot23 / parse_ton_iot via _build_sequences_grouped_by_device.
# This replaces the earlier approach of deriving a device from the attack label,
# which was circular (device == attack type) and collapsed all benign traffic
# onto a single device -- see the note in parse_ciciot2023 for the one dataset
# (CICIoT2023) that has no usable per-flow identity and is modeled as a single
# global device. IoT-23 and ToN_IoT retain real device IPs that appear in both
# benign and attack traffic, so each device gets its own memory bank.
# ---------------------------------------------------------------------------

# Legacy MAC prefixes retained only for the synthetic-data generator and the
# fingerprinting demo; real datasets no longer key device identity off these.
MAC_PREFIXES = ["00:1A:2B", "00:04:A3", "00:1E:C0", "3C:D9:2B", "E0:76:D0"]


class IoTSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapper for sequential IoT gateway packet data.
    Stores tensors of sequences: (num_samples, seq_len, num_features)
    """
    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor, device_classes: torch.Tensor, macs: list = None):
        self.sequences = sequences
        self.labels = labels
        self.device_classes = device_classes
        self.macs = macs if macs is not None else ["00:1e:c0:b4:a1:02"] * len(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.device_classes[idx], self.macs[idx]


def generate_synthetic_data(num_samples: int = 5000, seq_len: int = 16, num_features: int = 4,
                            num_classes: int = 34, num_devices: int = 5) -> IoTSequenceDataset:
    """
    Generates synthetic packet sequences for testing or dev_mode fallback.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    lengths = np.random.choice([60, 74, 512, 1024, 1500], size=(num_samples, seq_len, 1))
    lengths = lengths + np.random.normal(0, 5, size=lengths.shape)
    lengths = np.clip(lengths, 40, 1500)

    direction = np.random.choice([0.0, 1.0], size=(num_samples, seq_len, 1), p=[0.6, 0.4])
    iat = np.random.exponential(scale=0.1, size=(num_samples, seq_len, 1))
    flags = np.random.choice([2.0, 16.0, 18.0, 24.0], size=(num_samples, seq_len, 1)) / 32.0

    seq_data = np.concatenate([lengths, direction, iat, flags], axis=-1)
    labels = np.random.randint(0, num_classes, size=(num_samples,))
    device_classes = np.random.randint(1, num_devices + 1, size=(num_samples,))

    macs = []
    for d_class in device_classes:
        prefix = MAC_PREFIXES[(d_class - 1) % len(MAC_PREFIXES)]
        macs.append(f"{prefix}:11:22:{d_class:02X}")

    sequences_tensor = torch.tensor(seq_data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    device_classes_tensor = torch.tensor(device_classes, dtype=torch.long)

    return IoTSequenceDataset(sequences_tensor, labels_tensor, device_classes_tensor, macs)


def downsample_dataset(dataset: IoTSequenceDataset, num_samples: int) -> IoTSequenceDataset:
    N = len(dataset)
    if N <= num_samples:
        return dataset
    np.random.seed(42)
    indices = np.random.choice(N, size=num_samples, replace=False)

    sequences = dataset.sequences[indices]
    labels = dataset.labels[indices]
    device_classes = dataset.device_classes[indices]
    macs = [dataset.macs[i] for i in indices]

    return IoTSequenceDataset(sequences, labels, device_classes, macs)


# ---------------------------------------------------------------------------
# Build sequences by grouping rows that share the same label, so each sequence
# is semantically coherent (k consecutive flows of the same traffic/attack
# category) rather than an arbitrary mix of unrelated flows.
#
# `attack_to_device_map` maps each label -> a device class. This path is now
# used ONLY by CICIoT2023, which has no per-flow device identity and passes a
# constant map assigning every flow to the single global device (class 0).
# Datasets that DO have a real device identity (IoT-23 / ToN_IoT source IP) use
# _build_sequences_grouped_by_device instead.
# ---------------------------------------------------------------------------
def _build_sequences_grouped_by_label(lengths, direction, iat, flags, labels, seq_len, attack_to_device_map):
    seq_data_list = []
    seq_labels_list = []
    device_classes_list = []
    macs_list = []

    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    for label_val in unique_labels:
        idx = np.where(labels == label_val)[0]
        n = len(idx) // seq_len
        if n == 0:
            continue

        d_class = attack_to_device_map.get(int(label_val), 1)
        mac_prefix = MAC_PREFIXES[(d_class - 1) % len(MAC_PREFIXES)]

        for j in range(n):
            chunk_idx = idx[j * seq_len:(j + 1) * seq_len]

            seq = np.stack([
                lengths[chunk_idx],
                direction[chunk_idx],
                iat[chunk_idx],
                flags[chunk_idx],
            ], axis=-1)

            seq_data_list.append(seq)
            seq_labels_list.append(int(label_val))
            device_classes_list.append(d_class)
            macs_list.append(f"{mac_prefix}:11:22:{(j % 256):02X}")

    if not seq_data_list:
        return None

    seq_data = np.stack(seq_data_list, axis=0)
    seq_labels = np.array(seq_labels_list, dtype=int)
    device_classes = np.array(device_classes_list, dtype=int)

    sequences_tensor = torch.tensor(seq_data, dtype=torch.float32)
    labels_tensor = torch.tensor(seq_labels, dtype=torch.long)
    device_classes_tensor = torch.tensor(device_classes, dtype=torch.long)

    return IoTSequenceDataset(sequences_tensor, labels_tensor, device_classes_tensor, macs_list)


# ---------------------------------------------------------------------------
# Device-labeled sequence construction (source-IP identity).
#
# This is the correct realization of the proposal's device-aware design.
# Unlike the label-derived mapping above (where a device class is inferred
# from the *attack label*, which collapses all benign traffic onto a single
# device and makes routing circular), here the device class is the flow's
# own network identity -- its source IP. A single device therefore contributes
# BOTH benign and attack sequences, which is exactly what per-device behavioral
# memory banks require (they are fit on a device's benign traffic and used to
# score that same device's suspicious traffic).
#
# Applicable to IoT-23 (Zeek `id.orig_h`) and ToN_IoT Network (`src_ip`).
# CICIoT2023 flow CSVs retain no IP/MAC, so they cannot be modeled per-device
# and fall back to a single global device (class 0) -- see parse_ciciot2023.
# ---------------------------------------------------------------------------

def _is_private_ip(ip) -> bool:
    """True if `ip` is an RFC1918 / link-local address, i.e. a local device."""
    try:
        return ipaddress.ip_address(str(ip).strip()).is_private
    except ValueError:
        return False


def _synth_mac(device_id: int) -> str:
    """
    Deterministic locally-administered MAC that encodes a device_id in its low
    bytes. Used only for display / the fingerprinting fallback path; the
    evaluation threads the true integer device_class through the pipeline
    directly (network-identity binding), so routing does not depend on this.
    """
    d = int(device_id) & 0xFFFFFF
    return f"02:00:00:{(d >> 16) & 0xFF:02x}:{(d >> 8) & 0xFF:02x}:{d & 0xFF:02x}"


def _build_sequences_grouped_by_device(lengths, direction, iat, flags, labels,
                                       device_raw_ids, order_key, seq_len,
                                       min_flows_per_device=None):
    """
    Group flows by device identity, order each device's flows by `order_key`
    (timestamp), and slice into fixed-length packet sequences.

    Args:
        device_raw_ids: per-flow device identity (e.g. source-IP strings). Only
            flows with a non-empty identity are used; identities are densely
            remapped to integer device classes 1..K (0 is reserved for the
            Generic/unknown bank).
        order_key: per-flow sort key (timestamp) used to keep each device's
            sequence temporally coherent.
        min_flows_per_device: devices with fewer than this many flows are
            dropped (too few to form even one sequence / a stable bank).

    A sequence is labeled benign (0) only if ALL its flows are benign;
    otherwise it takes the majority non-benign attack label.
    """
    if min_flows_per_device is None:
        min_flows_per_device = seq_len

    lengths = np.asarray(lengths, dtype=float)
    direction = np.asarray(direction, dtype=float)
    iat = np.asarray(iat, dtype=float)
    flags = np.asarray(flags, dtype=float)
    labels = np.asarray(labels)
    device_raw_ids = np.asarray(device_raw_ids, dtype=object)
    order_key = np.asarray(order_key, dtype=float)

    # Valid device rows only (non-empty identity)
    valid = np.array([bool(str(d).strip()) and str(d) != 'nan' for d in device_raw_ids])
    if not np.any(valid):
        return None

    # Dense-remap identities -> integer device classes starting at 1
    unique_ids = {}
    next_id = 1
    seq_list, seq_labels, seq_devices, macs_list = [], [], [], []

    for raw in np.unique(device_raw_ids[valid]):
        idx = np.where((device_raw_ids == raw) & valid)[0]
        if len(idx) < min_flows_per_device:
            continue
        # temporal order within the device
        idx = idx[np.argsort(order_key[idx], kind='stable')]
        n = len(idx) // seq_len
        if n == 0:
            continue

        if raw not in unique_ids:
            unique_ids[raw] = next_id
            next_id += 1
        dev_id = unique_ids[raw]
        mac = _synth_mac(dev_id)

        for j in range(n):
            chunk = idx[j * seq_len:(j + 1) * seq_len]
            seq = np.stack([lengths[chunk], direction[chunk], iat[chunk], flags[chunk]], axis=-1)

            chunk_labels = labels[chunk]
            if np.all(chunk_labels == 0):
                seq_label = 0
            else:
                atk = chunk_labels[chunk_labels != 0]
                vals, counts = np.unique(atk, return_counts=True)
                seq_label = int(vals[np.argmax(counts)])

            seq_list.append(seq)
            seq_labels.append(seq_label)
            seq_devices.append(dev_id)
            macs_list.append(mac)

    if not seq_list:
        return None

    sequences_tensor = torch.tensor(np.stack(seq_list, axis=0), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(seq_labels, dtype=int), dtype=torch.long)
    device_classes_tensor = torch.tensor(np.array(seq_devices, dtype=int), dtype=torch.long)

    return IoTSequenceDataset(sequences_tensor, labels_tensor, device_classes_tensor, macs_list)


def load_iot_dataset(path: str, seq_len: int = 16, dev_mode: bool = True, dataset_type: str = "ciciot2023") -> IoTSequenceDataset:
    """
    Loads one of the real datasets (CICIoT2023, ToN_IoT, IoT-23) and constructs sequential data.
    If files are missing or paths are not found, falls back to generating synthetic data to prevent crashing.
    """
    if not os.path.exists(path):
        print(f"[WARNING] Path '{path}' not found. Falling back to synthetic dataset generation.")
        num_samples = 2000 if dev_mode else 50000
        return generate_synthetic_data(num_samples=num_samples, seq_len=seq_len)

    print(f"Loading dataset from '{path}' ({dataset_type.upper()})...")

    if dataset_type == "iot_23":
        files = sorted(glob.glob(os.path.join(path, "**", "conn.log.labeled"), recursive=True))
        if not files:
            print("[WARNING] No conn.log.labeled files found in IoT-23 path. Falling back to synthetic.")
            return generate_synthetic_data(num_samples=2000 if dev_mode else 50000, seq_len=seq_len)

        max_rows = 50000 if dev_mode else 1000000
        rows_per_file = max(2000, max_rows // len(files))

        dfs = []
        for file in files:
            try:
                fields = None
                with open(file, 'r') as f:
                    for line in f:
                        if line.startswith("#fields"):
                            fields = line.strip().split()[1:]
                            break
                if not fields:
                    continue

                df_file = pd.read_csv(file, sep=r'\s+', comment='#', names=fields, nrows=rows_per_file)
                if len(df_file) > 0 and len(df_file['label'].unique()) == 1:
                    try:
                        df_mid = pd.read_csv(file, sep=r'\s+', comment='#', names=fields, skiprows=20000, nrows=rows_per_file)
                        if len(df_mid) > 0:
                            df_file = pd.concat([df_file, df_mid], ignore_index=True)
                    except Exception:
                        pass
                dfs.append(df_file)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        if not dfs:
            return generate_synthetic_data(num_samples=2000 if dev_mode else 50000, seq_len=seq_len)

        df = pd.concat(dfs, ignore_index=True)

    elif dataset_type == "ton_iot":
        network_path = os.path.join(path, "TON_IoT datasets", "Processed_datasets", "Processed_Network_dataset")
        if not os.path.exists(network_path):
            network_path = path

        files = sorted(glob.glob(os.path.join(network_path, "**", "*.csv"), recursive=True))
        if not files:
            print("[WARNING] No CSV files found in ToN_IoT path. Falling back to synthetic.")
            return generate_synthetic_data(num_samples=2000 if dev_mode else 50000, seq_len=seq_len)

        max_rows = 150000 if dev_mode else 1000000
        rows_per_file = max(10000, max_rows // len(files))

        dfs = []
        for file in files:
            try:
                df_file = pd.read_csv(file, nrows=rows_per_file)
                df_file = df_file.rename(columns=lambda x: x.strip().replace(" ", ""))
                if len(df_file) > 0 and 'label' in df_file.columns and len(df_file['label'].unique()) == 1:
                    try:
                        df_mid = pd.read_csv(file, skiprows=150000, nrows=rows_per_file, header=0, names=df_file.columns)
                        df_file = pd.concat([df_file, df_mid], ignore_index=True)
                    except Exception:
                        pass
                dfs.append(df_file)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        if not dfs:
            return generate_synthetic_data(num_samples=2000 if dev_mode else 50000, seq_len=seq_len)

        df = pd.concat(dfs, ignore_index=True)

    else:  # ciciot2023
        folders = sorted([d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)])
        if not folders:
            folders = [path]

        max_rows = 50000 if dev_mode else 1000000
        max_rows_per_folder = max_rows // len(folders)

        dfs = []
        for folder in folders:
            folder_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
            if not folder_files:
                folder_files = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))

            if not folder_files:
                continue

            folder_name = os.path.basename(folder)
            folder_rows = 0
            for file in folder_files:
                if folder_rows >= max_rows_per_folder:
                    break
                try:
                    chunk_size = 10000
                    for chunk in pd.read_csv(file, chunksize=chunk_size):
                        chunk = chunk.rename(columns=lambda x: x.strip().replace(" ", ""))
                        chunk = chunk.fillna(0.0)

                        if 'label' not in chunk.columns:
                            class_label = "Benign_Final"
                            for k in CICIOT_CLASS_MAP.keys():
                                if k.lower() in folder_name.lower() or folder_name.lower() in k.lower():
                                    class_label = k
                                    break
                            chunk['label'] = class_label

                        if 'flow_duration' not in chunk.columns and 'IAT' in chunk.columns:
                            chunk['flow_duration'] = chunk['IAT']

                        rows_to_add = min(len(chunk), max_rows_per_folder - folder_rows)
                        if rows_to_add > 0:
                            dfs.append(chunk.iloc[:rows_to_add])
                            folder_rows += rows_to_add

                        if folder_rows >= max_rows_per_folder:
                            break
                except Exception as e:
                    print(f"Error reading {file}: {e}")

        if not dfs:
            print("[WARNING] No CSV files found in subdirectories. Falling back to synthetic dataset.")
            num_samples = 2000 if dev_mode else 50000
            return generate_synthetic_data(num_samples=num_samples, seq_len=seq_len)

        df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(df)} rows. Processing into flow sequences of length {seq_len}...")

    if dataset_type == "ciciot2023":
        ds = parse_ciciot2023(df, seq_len)
    elif dataset_type == "ton_iot":
        ds = parse_ton_iot(df, seq_len)
    else:  # iot_23
        ds = parse_iot23(df, seq_len)

    if dev_mode:
        ds = downsample_dataset(ds, 2000)

    return ds


def parse_ciciot2023(df: pd.DataFrame, seq_len: int) -> IoTSequenceDataset:
    """
    Parses CICIoT2023 DataFrame into sequential flow tokens, grouped by label so
    each sequence is semantically coherent.

    NOTE ON DEVICE IDENTITY: CICIoT2023's published flow CSVs contain only
    aggregated statistical features -- there is no per-flow source IP or MAC, so
    a genuine device identity cannot be recovered. We therefore assign ALL
    CICIoT2023 flows to a single global device (class 0, the Generic bank)
    instead of deriving a device from the attack label (which was circular and
    collapsed all benign traffic onto one device). On this dataset the framework
    reduces to a single global behavioral bank; device-aware evaluation is done
    on IoT-23 / ToN_IoT, which retain per-flow source-IP device identity.
    """
    required_cols = ['Header_Length', 'flow_duration', 'Rate', 'label']
    for c in required_cols:
        if c not in df.columns:
            print(f"[WARNING] Missing column '{c}' in CICIoT2023. Available columns: {list(df.columns[:10])}. Falling back to synthetic.")
            return generate_synthetic_data(num_samples=len(df)//seq_len, seq_len=seq_len)

    labels = []
    for lbl in df['label']:
        matched = False
        for k, v in CICIOT_CLASS_MAP.items():
            if k.lower() in str(lbl).lower():
                labels.append(v)
                matched = True
                break
        if not matched:
            labels.append(0)

    labels = np.array(labels)
    lengths = df['Header_Length'].values

    direction = np.zeros(len(df))
    if 'rst_flag_number' in df.columns and 'syn_flag_number' in df.columns:
        direction = ((df['syn_flag_number'] > 0) | (df['rst_flag_number'] > 0)).astype(float).values

    if 'IAT' in df.columns:
        iat = df['IAT'].values
    else:
        iat = (df['flow_duration'] / (df['Rate'] + 1e-5)).values
    iat = np.nan_to_num(iat, nan=0.0, posinf=0.0, neginf=0.0)
    
    flags = np.zeros(len(df))
    flag_cols = ['syn_flag_number', 'ack_flag_number', 'rst_flag_number', 'fin_flag_number', 'psh_flag_number']
    for idx, col in enumerate(flag_cols):
        if col in df.columns:
            flags += df[col].values * (2 ** idx)
    flags = flags / 32.0

    # Single global device (class 0) -- see docstring: no per-flow identity exists.
    ciciot_single_device = {int(l): 0 for l in np.unique(labels)}
    ds = _build_sequences_grouped_by_label(lengths, direction, iat, flags, labels, seq_len, ciciot_single_device)
    if ds is None:
        return generate_synthetic_data(num_samples=10, seq_len=seq_len)
    return ds


def parse_ton_iot(df: pd.DataFrame, seq_len: int) -> IoTSequenceDataset:
    """
    Parses a ToN_IoT network-flow DataFrame into sequential flow tokens grouped
    by DEVICE. Device identity is the private source IP (`src_ip`); the ToN_IoT
    testbed uses stable device IPs that appear in both normal and attack traffic,
    so each device contributes benign and attack sequences.
    """
    src_ip = df['src_ip'].astype(str).values if 'src_ip' in df.columns else np.array([''] * len(df))
    src_priv = np.array([_is_private_ip(ip) for ip in src_ip])
    device_ip = np.where(src_priv, src_ip, '')
    direction = src_priv.astype(float)  # 1.0 = local device is the source (egress)

    lengths = pd.to_numeric(df['src_ip_bytes'], errors='coerce').fillna(0.0).values \
        if 'src_ip_bytes' in df.columns else np.zeros(len(df))

    duration = pd.to_numeric(df['duration'], errors='coerce').fillna(0.0).values \
        if 'duration' in df.columns else np.zeros(len(df))
    src_pkts = pd.to_numeric(df['src_pkts'], errors='coerce').fillna(0.0).values \
        if 'src_pkts' in df.columns else np.zeros(len(df))
    dst_pkts = pd.to_numeric(df['dst_pkts'], errors='coerce').fillna(0.0).values \
        if 'dst_pkts' in df.columns else np.zeros(len(df))
    iat = duration / (src_pkts + dst_pkts + 1e-5)
    iat = np.nan_to_num(iat, nan=0.0, posinf=0.0, neginf=0.0)

    flags = np.zeros(len(df))
    if 'conn_state' in df.columns:
        state_map = {'S0': 2.0, 'SF': 18.0, 'REJ': 4.0}
        flags = df['conn_state'].apply(lambda s: state_map.get(s, 16.0)).values / 32.0

    if 'type' in df.columns:
        labels = df['type'].astype(str).str.strip().str.lower().map(
            lambda t: TON_IOT_CLASS_MAP.get(t, 0)
        ).values
    else:
        labels = (df['label'].values if 'label' in df.columns else np.zeros(len(df)))
    labels = np.array(labels, dtype=int)

    if 'ts' in df.columns:
        order_key = pd.to_numeric(df['ts'], errors='coerce').fillna(0.0).values
    else:
        order_key = np.arange(len(df), dtype=float)

    ds = _build_sequences_grouped_by_device(lengths, direction, iat, flags, labels,
                                            device_ip, order_key, seq_len)
    if ds is None:
        return generate_synthetic_data(num_samples=10, seq_len=seq_len)
    return ds


def parse_iot23(df: pd.DataFrame, seq_len: int) -> IoTSequenceDataset:
    """
    Parses an IoT-23 Zeek connection log into sequential flow tokens grouped by
    DEVICE. Device identity is the local host of each flow (`id.orig_h` when it
    is a private address, otherwise `id.resp_h` if that is private) -- a real,
    per-flow network identity that spans both benign and malicious traffic.
    Packet direction is taken from that device's perspective (egress=1 when the
    device originated the flow, ingress=0 when it was the responder).
    """
    orig_h = df['id.orig_h'].astype(str).values if 'id.orig_h' in df.columns else np.array([''] * len(df))
    resp_h = df['id.resp_h'].astype(str).values if 'id.resp_h' in df.columns else np.array([''] * len(df))

    orig_priv = np.array([_is_private_ip(ip) for ip in orig_h])
    resp_priv = np.array([_is_private_ip(ip) for ip in resp_h])

    # Device = the local endpoint; egress when the device is the originator.
    device_ip = np.where(orig_priv, orig_h, np.where(resp_priv, resp_h, ''))
    direction = orig_priv.astype(float)  # 1.0 = device originated (egress)

    lengths = pd.to_numeric(df['orig_ip_bytes'], errors='coerce').fillna(0.0).values

    duration = pd.to_numeric(df['duration'], errors='coerce').fillna(0.0).values
    orig_pkts = pd.to_numeric(df['orig_pkts'], errors='coerce').fillna(0.0).values
    resp_pkts = pd.to_numeric(df['resp_pkts'], errors='coerce').fillna(0.0).values
    iat = duration / (orig_pkts + resp_pkts + 1e-5)
    iat = np.nan_to_num(iat, nan=0.0, posinf=0.0, neginf=0.0)

    flags = np.zeros(len(df))
    if 'conn_state' in df.columns:
        state_map = {'S0': 2.0, 'SF': 18.0, 'REJ': 4.0}
        flags = df['conn_state'].apply(lambda s: state_map.get(s, 16.0)).values / 32.0

    label_col = 'detailed-label' if 'detailed-label' in df.columns else 'label'

    def map_iot23_label(lbl_val):
        lbl_val = str(lbl_val).strip().lower()
        if lbl_val == '-' or 'benign' in lbl_val or 'normal' in lbl_val:
            return 0
        for k, v in IOT23_CLASS_MAP.items():
            if k != 'benign' and k in lbl_val:
                return v
        return 3

    labels = np.array(df[label_col].apply(map_iot23_label).values, dtype=int)

    if 'ts' in df.columns:
        order_key = pd.to_numeric(df['ts'], errors='coerce').fillna(0.0).values
    else:
        order_key = np.arange(len(df), dtype=float)

    ds = _build_sequences_grouped_by_device(lengths, direction, iat, flags, labels,
                                            device_ip, order_key, seq_len)
    if ds is None:
        return generate_synthetic_data(num_samples=10, seq_len=seq_len)
    return ds