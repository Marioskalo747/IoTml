#Libraries
import io #in-memory buffers (CSVs written to zip/tar)
import sys 
import tarfile
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SMOKE_DATA_DIR = ROOT / "smoke_data"
RNG= np.random.RandomState(42)  #fixed seed for reproducibility

N = 900 #rows per class

'''generate distinct statistics per class'''
def class_signal(label_id, n):
    base = (label_id + 1) * 1.7
    dur = RNG.exponential(0.5 + label_id, n) + 0.01 #exponential distribution for duration
    pkts = RNG.poisson(0.5 + 30*label_id, n) + 1 #poisson for packet counts (+1 avoid zero)
    bts =  pkts * (RNG.normal(120 + 200*label_id, 40, n).clip(40, None)) #no packets smaller than 40 bytes (header)
    return {"dur": dur, "pkts": pkts, "bytes": bts.astype(int), "base": base}

'''Builds synthetic Bot-IoT'''
def makebot_iot():
    d = SMOKE_DATA_DIR / "bot_iot"
    d.mkdir(parents=True, exist_ok=True)
    cats = ["Normal", "DDoS", "DoS", "Reconnaissance", "Theft"] #BoT-IoT classes
    frames = []
    for i, cat in enumerate(cats):
        sig = class_signal(i, N)
        frames.append(pd.DataFrame({"pkSeqID": np.arange(N), "stime": 1.5e9 + np.arange(N),
        "flgs": "e", "flgs_number": 1, "proto": RNG.choice(["tcp", "udp", "icmp"], N),
        "proto_number": 1, "saddr": [f"192.168.100.{RNG.randint(2,250)}" for _ in range(N)],
        "sport": RNG.randint(1024, 65535, N), "daddr": [f"192.168.100.{RNG.randint(2,250)}" for _ in range(N)],
        "dport": RNG.choice([80, 443, 23, 1883], N), "pkts": sig["pkts"], "bytes": sig["bytes"],
        "state": RNG.choice(["CON", "RST", "FIN"], N), "state_number": 1, "ltime": 1.5e9 + np.arange(N) + sig["dur"], "seq": np.arange(N),
        "dur": sig["dur"], "mean": sig["bytes"] / sig["pkts"], "stddev": RNG.rand(N) * 10, "sum": sig["bytes"],
        "min": 40, "max": 1500, "spkts": sig["pkts"] // 2, "dpkts": sig["pkts"] - sig["pkts"] // 2,
        "sbytes": sig["bytes"] // 2, "dbytes": sig["bytes"] - sig["bytes"] // 2,
        "rate": sig["pkts"] / sig["dur"], "srate": sig["pkts"] / sig["dur"] / 2, "drate": sig["pkts"] / sig["dur"] / 2,
        "TnBPSrcIP": RNG.randint(100, 1e5, N), "TnBPDstIP": RNG.randint(100, 1e5, N),
        "TnP_PSrcIP": RNG.randint(1, 1000, N), "TnP_PDstIP": RNG.randint(1, 1000, N),
        "TnP_PerProto": RNG.randint(1, 1000, N), "TnP_Per_Dport": RNG.randint(1, 1000, N),
        "AR_P_Proto_P_SrcIP": RNG.rand(N) * 2, "AR_P_Proto_P_DstIP": RNG.rand(N) * 2,
        "N_IN_Conn_P_DstIP": RNG.randint(1, 100, N), "N_IN_Conn_P_SrcIP": RNG.randint(1, 100, N),
        "AR_P_Proto_P_Sport": RNG.rand(N) * 2, "AR_P_Proto_P_Dport": RNG.rand(N) * 2,
        "Pkts_P_State_P_Protocol_P_DestIP": RNG.randint(1, 1000, N), "Pkts_P_State_P_Protocol_P_SrcIP": RNG.randint(1, 1000, N),
        "attack": 0 if cat == "Normal" else 1, "category": cat, "subcategory": "Normal" if cat == "Normal" else "HTTP"}))
        
    df = pd.concat(frames, ignore_index=True)
    buf = io.StringIO() #csv in memory
    df.to_csv(buf, index=False)
    with zipfile.ZipFile(d / "bot_iot.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:  #path to zip 
        zf.writestr("5%/All features/UNSW_2018_IoT_Botnet_Full5pc_1.csv", buf.getvalue()) 

'''Builds synthetic ToN-IoT dataset'''
def maketon_iot():
    d = SMOKE_DATA_DIR / "ton_iot"
    d.mkdir(parents=True, exist_ok=True)
    cats = ["normal", "ddos", "dos", "scanning", "password", "xss", "injection", "backdoor", "mitm"] #ToN-IoT classes
    frames = []
    for i, t in enumerate(cats):
        sig = class_signal(i, N)
        frames.append(pd.DataFrame({"ts": 1.6e9 + np.arange(N),
        "src_ip": [f"10.0.0.{RNG.randint(2,250)}" for _ in range(N)],
        "src_port": RNG.randint(1024, 65535, N),
        "dst_ip": [f"10.0.0.{RNG.randint(2,250)}" for _ in range(N)],
        "dst_port": RNG.choice([80, 443, 22, 1883, 8080], N),
        "proto": RNG.choice(["tcp", "udp", "icmp"], N),
        "service": RNG.choice(["-", "http", "dns", "mqtt"], N),
        "duration": sig["dur"],"src_bytes": sig["bytes"] // 2,"dst_bytes": sig["bytes"] - sig["bytes"] // 2,
        "conn_state": RNG.choice(["SF", "S0", "REJ", "OTH"], N),
        "missed_bytes": 0,"src_pkts": sig["pkts"] // 2,"src_ip_bytes": sig["bytes"] // 2,
        "dst_pkts": sig["pkts"] - sig["pkts"] // 2,"dst_ip_bytes": sig["bytes"] - sig["bytes"] // 2,
        #application layer features (mostly dummy values)
        "dns_query": "-", "dns_qclass": 0, "dns_qtype": 0, "dns_rcode": 0,
        "dns_AA": "-", "dns_RD": "-", "dns_RA": "-", "dns_rejected": "-",
        "ssl_version": "-", "ssl_cipher": "-", "ssl_resumed": "-",
        "ssl_established": "-", "ssl_subject": "-", "ssl_issuer": "-",
        "http_trans_depth": "-", "http_method": "-", "http_uri": "-",
        "http_version": "-", "http_request_body_len": 0,
        "http_response_body_len": 0, "http_status_code": 0,
        "http_user_agent": "-", "http_orig_mime_types": "-",
        "http_resp_mime_types": "-", "weird_name": "-", "weird_addl": "-",
        "weird_notice": "-", "label": 0 if t == "normal" else 1, "type": t}))
    
    pd.concat(frames, ignore_index=True).to_csv(d / "ton_iot.csv", index=False)
    
'''Builds synthetic CICIOT dataset'''    
def makeciciot():
    d = SMOKE_DATA_DIR / "ciciot2023"
    d.mkdir(parents=True, exist_ok=True)
    cats = ["Benign_Final", "DDoS-ICMP_Flood", "DoS-TCP_Flood","Mirai-greeth_flood", "Recon-PortScan", "DictionaryBruteForce","MITM-ArpSpoofing", "SqlInjection"] #CICIOT classes
    with zipfile.ZipFile(d / "ciciot2023.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:  ###onoma
        for i, t in enumerate(cats):
            sig = class_signal(i, N)
            #CICIOT hasn't IPs and ports (packet stats only)
            df = pd.DataFrame({"Header_Length": RNG.randint(20, 60, N),
            "Protocol Type": RNG.choice([6, 17, 1], N), "Time_To_Live": RNG.randint(30, 255, N), #6=TCP, 17=UDP, 1=ICMP
            "Rate": sig["pkts"] / sig["dur"],"fin_flag_number": RNG.randint(0, 2, N),
            "syn_flag_number": RNG.randint(0, 2, N),"rst_flag_number": RNG.randint(0, 2, N),
            "psh_flag_number": RNG.randint(0, 2, N),"ack_flag_number": RNG.randint(0, 2, N),
            "ece_flag_number": 0, "cwr_flag_number": 0,
            "ack_count": RNG.poisson(2, N), "syn_count": RNG.poisson(2, N),
            "fin_count": RNG.poisson(1, N), "rst_count": RNG.poisson(1, N),
            "HTTP": RNG.randint(0, 2, N), "HTTPS": RNG.randint(0, 2, N),
            "DNS": 0, "Telnet": 0, "SMTP": 0, "SSH": 0, "IRC": 0,
            "TCP": RNG.randint(0, 2, N), "UDP": RNG.randint(0, 2, N),
            "DHCP": 0, "ARP": 0, "ICMP": RNG.randint(0, 2, N), "IGMP": 0,
            "IPv": 1, "LLC": 1,"Tot sum": sig["bytes"], "Min": 40, "Max": 1500,
            "AVG": sig["bytes"] / sig["pkts"], "Std": RNG.rand(N) * 50,
            "Tot size": sig["bytes"] / sig["pkts"],"IAT": sig["dur"] / sig["pkts"],
            "Number": sig["pkts"],"Variance": RNG.rand(N)})
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            zf.writestr(f"{t}/{t}.csv", buf.getvalue())
            
#zeek file header (the #-prefixed lines that declare the separator, the fields and their types, \\x09 is the tab character)             
ZEEK_HEADER = """#separator \\x09
#set_separator\t,
#empty_field\t(empty)
#unset_field\t-
#path\tconn
#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state\tlocal_orig\tlocal_resp\tmissed_bytes\thistory\torig_pkts\torig_ip_bytes\tresp_pkts\tresp_ip_bytes\ttunnel_parents\tlabel\tdetailed-label
#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\tcount\tcount\tstring\tbool\tbool\tcount\tstring\tcount\tcount\tcount\tcount\tset[string]\tstring\tstring
"""
'''Builds synthetic IoT23 dataset'''
def makeiot23():
    d = SMOKE_DATA_DIR / "iot23"
    d.mkdir(parents=True, exist_ok=True)
    cats = [("benign", "-"), ("Malicious", "PartOfHorizontalPortScan"), ("Malicious", "DDoS"), ("Malicious", "Okiru"), ("Malicious", "C&C")] #IOT23 classes
    lines = []
    for i, (cat, subcat) in enumerate(cats):
        sig = class_signal(i, N)
        for j in range(N):
            op = int(sig["pkts"][j] // 2) #original packets and bytes (half of total)
            rp = int(sig["pkts"][j]- sig["pkts"][j] // 2) #response packets and bytes (the other half)
            ob = int(sig["bytes"][j] // 2) #original bytes and response bytes (half of total)
            rb = int(sig["bytes"][j] - sig["bytes"][j] // 2) #response bytes (the other half)
            #ports 23/2323 classic Mirai Telnet scenario
            lines.append("\t".join(map(str, [1.55e9 +j, f"C{i}{j}", f"192.168.1.{RNG.randint(2,250)}", RNG.randint(1024, 65535), f"192.168.1.{RNG.randint(2,250)}", 
            RNG.choice([23, 2323, 80, 443, 8081]), RNG.choice(["tcp", "udp", "icmp"]), "-", round(sig["dur"][j], 4), ob, rb,
            RNG.choice(["SF", "S0", "REJ"]), "-", "-", 0, "S", op, ob + 40 * op, rp, rb + 40 * rp, "-", cat, subcat ])))
        
    content = ZEEK_HEADER + "\n".join(lines) + "\n"
    tar_path = d / "iot_23.tar.gz"   ###ONOMA
    with tarfile.open(tar_path, "w:gz") as tf:
        data = content.encode()
        info = tarfile.TarInfo("iot_23_small/CTU-Smoke-1/bro/conn.log.labeled")   ##onoma
        info.size = len(data) #tar requires size to be set
        tf.addfile(info, io.BytesIO(data))
                                    


'''Generates all synthetic IoT datasets'''
def smoke_datasets():
    print("Generating synthetic IoT datasets in", SMOKE_DATA_DIR)
    makebot_iot()
    maketon_iot()
    makeciciot()
    makeiot23()
    print("done")
    
if __name__ == "__main__":
    smoke_datasets()