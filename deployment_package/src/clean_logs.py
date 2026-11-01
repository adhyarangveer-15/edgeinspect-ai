import pandas as pd
import sqlite3
import os

csv_path = "reports/inspection_log.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    mapping = {
        "auto": "Automotive Engine Components",
        "cast": "Automotive Engine Components",
        "packag": "Packaging & Containers",
        "seal": "Packaging & Containers",
        "pharma": "Pharmaceutical Tablets",
        "tablet": "Pharmaceutical Tablets",
        "steel": "Steel & Metal Surface",
        "metal": "Steel & Metal Surface",
        "textile": "Textile & Fabric",
        "fabric": "Textile & Fabric",
        "sample_": "Printed Circuit Board (PCB)",
        "test_pcb": "Printed Circuit Board (PCB)"
    }
    for idx, row in df.iterrows():
        pname = str(row["part_name"]).lower()
        for k, v in mapping.items():
            if k in pname:
                df.at[idx, "application_domain"] = v
                break
    df.to_csv(csv_path, index=False)
    print("reports/inspection_log.csv updated successfully!")

db_path = "reports/inspections.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("UPDATE inspection_records SET application_domain = 'Automotive Engine Components' WHERE part_name LIKE '%auto%' OR part_name LIKE '%cast%'")
    cur.execute("UPDATE inspection_records SET application_domain = 'Packaging & Containers' WHERE part_name LIKE '%packag%' OR part_name LIKE '%seal%'")
    cur.execute("UPDATE inspection_records SET application_domain = 'Pharmaceutical Tablets' WHERE part_name LIKE '%pharma%' OR part_name LIKE '%tablet%'")
    cur.execute("UPDATE inspection_records SET application_domain = 'Steel & Metal Surface' WHERE part_name LIKE '%steel%' OR part_name LIKE '%metal%'")
    cur.execute("UPDATE inspection_records SET application_domain = 'Textile & Fabric' WHERE part_name LIKE '%textile%' OR part_name LIKE '%fabric%'")
    conn.commit()
    conn.close()
    print("reports/inspections.db updated successfully!")
