import csv
import time
import math
import random
from pathlib import Path

from mininet.log import setLogLevel, info
from mininet.node import Controller
from mn_wifi.net import Mininet_wifi

AP_POS = {
    "ap1": (10, 50),
    "ap2": (50, 50),
    "ap3": (90, 50),
    "ap4": (40, 10),  # rogue AP
}

def rssi_from_distance(distance, rssi_at_1m=-30.0, path_loss_exp=3.0):
    # Hindari log(0)
    distance = max(distance, 1.0)
    return round(rssi_at_1m - 10 * path_loss_exp * math.log10(distance), 2)

def collect_data():
    net = Mininet_wifi(controller=Controller)

    info("Membangun Topologi untuk Pengambilan Data\n")

    ap1 = net.addAccessPoint('ap1', ssid='WiFi-Kampus', mode='g', channel='1', position='10,50,0')
    ap2 = net.addAccessPoint('ap2', ssid='WiFi-Kampus', mode='g', channel='6', position='50,50,0')
    ap3 = net.addAccessPoint('ap3', ssid='WiFi-Kampus', mode='g', channel='11', position='90,50,0')
    ap4 = net.addAccessPoint('ap4', ssid='WiFi-Kampus', mode='g', channel='1', position='40,10,0')

    sta1 = net.addStation('sta1', ip='10.0.0.1/8')
    c1 = net.addController('c1')

    net.setPropagationModel(model='logDistance', exp=3)
    net.configureWifiNodes()
    net.build()

    c1.start()
    for ap in [ap1, ap2, ap3, ap4]:
        ap.start([c1])

    info("Memulai Pengambilan 10.201 Data\n")

    output_path = Path("../02_Dataset/dataset_rssi.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'rssi_ap1', 'rssi_ap2', 'rssi_ap3', 'rssi_ap4', 'is_rogue'])

        for x in range(0, 101, 1): # Step diubah jadi 1
            for y in range(0, 101, 1): # Step diubah jadi 1
                pos = f'{x},{y},0'
                sta1.setPosition(pos)
                
                # Dipercepat agar 10rb data tidak memakan waktu berjam-jam
                time.sleep(0.01) 

                # Hitung jarak dasar
                d1 = ((x - AP_POS["ap1"][0])**2 + (y - AP_POS["ap1"][1])**2)**0.5
                d2 = ((x - AP_POS["ap2"][0])**2 + (y - AP_POS["ap2"][1])**2)**0.5
                d3 = ((x - AP_POS["ap3"][0])**2 + (y - AP_POS["ap3"][1])**2)**0.5
                d4 = ((x - AP_POS["ap4"][0])**2 + (y - AP_POS["ap4"][1])**2)**0.5

                # INJEKSI RANDOM NOISE 
                noise = lambda: random.uniform(-3.0, 3.0)
                
                rssi1 = rssi_from_distance(d1) + noise()
                rssi2 = rssi_from_distance(d2) + noise()
                rssi3 = rssi_from_distance(d3) + noise()
                rssi4 = rssi_from_distance(d4) + noise()

                label = 1 if d4 < 25 else 0

                writer.writerow([x, y, rssi1, rssi2, rssi3, rssi4, label])
                
            # Log dicetak setiap baris X selesai (agar tidak memenuhi layar)
            print(f"Progres: Selesai merekam baris X = {x}/100")

    info("\n*** Selesai! Dataset baru dengan 10.201 sampel berhasil disimpan di 02_Dataset/dataset_rssi.csv\n")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    collect_data()