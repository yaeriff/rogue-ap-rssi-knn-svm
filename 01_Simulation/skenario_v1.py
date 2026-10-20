from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI

def topology():
    "Membuat Jaringan Nirkabel dengan 3 AP Legal dan 1 Rogue AP."
    net = Mininet_wifi()

    info("Menambahkan Nodes\n")
    # AP Legal
    ap1 = net.addAccessPoint('ap1', ssid='WiFi-Kampus', mode='g', channel='1',
                         position='10,50,0', dpid='0000000000000001')
    ap2 = net.addAccessPoint('ap2', ssid='WiFi-Kampus', mode='g', channel='6',
                         position='50,50,0', dpid='0000000000000002')
    ap3 = net.addAccessPoint('ap3', ssid='WiFi-Kampus', mode='g', channel='11',
                         position='90,50,0', dpid='0000000000000003')

    # Rogue AP
    rogue_ap = net.addAccessPoint('ap4', ssid='WiFi-Kampus', mode='g', channel='1',
                              position='40,10,0', dpid='0000000000000004')

    # Station
    sta1 = net.addStation('sta1', ip='10.0.0.1/8', position='10,10,0')

    c1 = net.addController('c1')

    info("Mengatur Model Propagasi\n")
    net.setPropagationModel(model='logDistance', exp=3)

    info("Konfigurasi WiFi\n")
    net.configureWifiNodes()

    info("Build & Start\n")
    net.build()
    c1.start()
    ap1.start([c1])
    ap2.start([c1])
    ap3.start([c1])
    rogue_ap.start([c1])

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    topology()