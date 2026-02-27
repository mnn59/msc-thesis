from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import Link, Intf, TCLink
from mininet.topo import Topo
from mininet.util import custom, pmonitor
import logging
import os
from functools import partial
import socket
import json
import time
import heapq
import sys

class CustomTopo(Topo):
    def __init__(self, nodeNum, linkSet, bandwidths, losses, **opts):
        Topo.__init__(self,**opts)
        self.__nodenum = nodeNum
        self.__linkset = linkSet
        self.__bandwidths = bandwidths
        self.__losses = losses

        self.__switches = []
        self.__hosts = []

        self.create_net()
        self.add_hosts()

    '''create the network topo'''
    def create_net(self):
        for i in range(self.__nodenum):
            self.__switches.append(self.addSwitch("s" + str(i + 1)))
        for i in range(len(self.__linkset)):
            node1 = self.__linkset[i][0]
            node2 = self.__linkset[i][1]
            self.addLink(self.__switches[node1], self.__switches[node2], bw=self.__bandwidths[i], delay='5ms', loss=self.__losses[i], max_queue_size=1000) 
    
    '''add host for each switch(node)'''
    def add_hosts(self):
        if self.__nodenum >= 255:
            print("ERROR!!!")
            exit()
        for i in range(self.__nodenum):
            self.__hosts.append(self.addHost("h" + str(i + 1), mac=("00:00:00:00:00:%02x" % (i + 1)), ip = "10.0.0." + str(i + 1)))
            self.addLink(self.__switches[i], self.__hosts[i], bw=1000, delay='0ms') # bw here should be large enough
        

def generate_request(net, src, src_port, dst, dst_port, rtype, demand, rtime, time_step): 
    TIME_OUT = 5
    src_host = net.hosts[src]
    dst_host = net.hosts[dst]

    popens = {}
    popens[dst_host] = dst_host.popen("python3 server.py %s %d %d %d %d" % (dst_host.IP(), dst_port, rtime, rtype, time_step))
    time.sleep(0.1)
    popens[src_host] = src_host.popen("python3 client.py %s %d %s %d %d %d %d" % (dst_host.IP(), dst_port, src_host.IP(), src_port, demand, rtime, time_step))
    src_popen = popens[src_host]
    dst_popen = popens[dst_host]
    ind = 0
    time_stamp = time.time()
    for host, line in pmonitor(popens):
        if time.time() - time_stamp > TIME_OUT:
            print("Request:", "src:", src, "dst:", dst, "rtype:", rtype, "demand:", demand)
            delay = TIME_OUT * 1000
            throughput = 0
            loss = 1.
            print("time out!")
            break
        if host:
            print("<%s>: %s" % (host.name, line))
            
            if host == dst_host:
                ret = line.split()
                delay = float(ret[1])
                throughput = float(ret[4])
                loss = float(ret[7])
                #flag = True
                if ind == 1: # avoid using the first data received from server
                    break
                else:
                    ind += 1
            
    return delay, throughput, loss, (src_popen, dst_popen)

def load_topoinfo(toponame):
    topo_file = open("./topo_info/%s.txt" % toponame, "r")
    content = topo_file.readlines()
    nodeNum, linkNum = map(int, content[0].split())
    linkSet = []
    bandwidths = []
    losses = []
    for i in range(linkNum):
        u, v, w, c, loss = map(int, content[i + 1].split())
        linkSet.append([u - 1, v - 1])
        bandwidths.append(float(c) / 1000) 
        losses.append(loss)
    return nodeNum, linkSet, bandwidths, losses

if __name__ == '__main__':
    print ("testbed initializing ...")
    toponame = sys.argv[1]
    if toponame == "test":
        nodeNum = 4
        linkSet = [[0, 1], [1, 2], [2, 3], [0, 3]]
        bandwidths = [1, 5, 5, 5]
        losses = [0, 0, 0, 0] # 0% must be int
    else:
        nodeNum, linkSet, bandwidths, losses = load_topoinfo(toponame)
    print ("topoinfo loading finished.")
    requests_pq = [] # put the popens of requests' server and client process
    
    topo = CustomTopo(nodeNum, linkSet, bandwidths, losses)
    CONTROLLER_IP = "127.0.0.1" # Your ryu controller server IP
    CONTROLLER_PORT = 5001 
    OVSSwitch13 = partial(OVSSwitch, protocols='OpenFlow13')
    net = Mininet(topo=topo, switch=OVSSwitch13, link=TCLink, controller=None)
    net.addController('controller', controller=RemoteController, ip=CONTROLLER_IP, port=CONTROLLER_PORT)
    net.start()
    
    
    # build communication with DRL client
    print ("waiting to simenv")
    # If using unique server for testbed, set TCP_IP to the server IP 
    TCP_IP = "127.0.0.1"
    TCP_PORT = 5000
    BUFFER_SIZE = 1024
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    
    conn, addr = s.accept()
    print('Connection address:', addr)
    time_step = 0
    # receive instruction from sim_env.py and generate request and send results
    while True:
        try:
            msg = conn.recv(BUFFER_SIZE)
        except Exception as e:
            # sock.close()
            print("Socket error:", e)
            conn.close()
            s.close()
            break
        #print("msg:", msg)
         
        while len(requests_pq) > 0 and requests_pq[0][0] <= time_step:
            ind, popens = heapq.heappop(requests_pq)
            popens[0].kill()
            popens[1].kill()
        

        data_js = json.loads(msg)
        rtime = data_js['rtime']
        delay, throughput, loss, popens = generate_request(net, data_js['src'], data_js['src_port'], data_js['dst'], data_js['dst_port'], data_js['rtype'], data_js['demand'], 1000000, time_step) # rtime is a deprecated para
        
        heapq.heappush(requests_pq, (rtime + time_step, popens))
        
        ret = {
                'delay': delay,
                'throughput': throughput,
                'loss': loss,
                }
        
        # For Abi link failure & demand change test
        # we let testbed send the failure information to simenv for simple implementation 
        if time_step == 10000:  
            
            # link failue
            # net.configLinkStatus('s1', 's5', 'down')
            # ret['change'] = 'link_failure'
            
            # demand change
            ret['change'] = "demand_change"

            # initialization
            # pass
        
        
        msg = json.dumps(ret)
        conn.send(msg.encode())
        time_step += 1

    CLI(net)
    





















# """
# Enhanced Testbed for DRL-OR with 5 Scenario Support
# =====================================================
# Replaces the hardcoded 'if time_step == 10000' block with an
# event-driven system. Pass --scenario on the command line.

# Usage:
#     sudo python3 testbed.py initialization
#     sudo python3 testbed.py link_failure
#     sudo python3 testbed.py traffic_change
#     sudo python3 testbed.py cascading_failure
#     sudo python3 testbed.py link_degradation

# Original DRL-OR testbed: you had to manually comment/uncomment
# the link_failure / demand_change / initialization blocks.
# Now it's automatic.
# """

# from mininet.net import Mininet
# from mininet.node import Controller, RemoteController, OVSSwitch
# from mininet.cli import CLI
# from mininet.log import setLogLevel, info
# from mininet.link import Link, Intf, TCLink
# from mininet.topo import Topo
# from mininet.util import custom, pmonitor
# import logging
# import os
# from functools import partial
# import socket
# import json
# import time
# import heapq
# import sys


# # =============================================================================
# # SCENARIO EVENT TABLES
# # =============================================================================
# # Each scenario has a list of (timestep, action, mininet_ops)
# # 'action' is sent to simenv via ret['change']
# # 'mininet_ops' is a list of Mininet operations to execute

# SCENARIO_EVENTS = {
#     'initialization': [],  # No events

#     'link_failure': [
#         {
#             'timestep': 10000,
#             'action': 'link_failure',
#             'description': 'Link 0-4 fails (s1-s5)',
#             'link_ops': [('s1', 's5', 'down')],
#             'bw_ops': [],
#         },
#     ],

#     'traffic_change': [
#         {
#             'timestep': 10000,
#             'action': 'demand_change',
#             'description': 'Switch to mid load',
#             'link_ops': [],
#             'bw_ops': [],
#         },
#     ],

#     'cascading_failure': [
#         {
#             'timestep': 10000,
#             'action': 'cascade_failure_1',
#             'description': 'Link 0-4 fails (s1-s5, bottleneck)',
#             'link_ops': [('s1', 's5', 'down')],
#             'bw_ops': [],
#         },
#         {
#             'timestep': 50000,
#             'action': 'cascade_failure_2',
#             'description': 'Link 1-3 fails (s2-s4, rerouting stress)',
#             'link_ops': [('s2', 's4', 'down')],
#             'bw_ops': [],
#         },
#         {
#             'timestep': 100000,
#             'action': 'cascade_failure_3',
#             'description': 'Link 4-7 fails (s5-s8, alt path)',
#             'link_ops': [('s5', 's8', 'down')],
#             'bw_ops': [],
#         },
#         {
#             'timestep': 150000,
#             'action': 'partial_recovery',
#             'description': 'Link 0-4 restored (s1-s5)',
#             'link_ops': [('s1', 's5', 'up')],
#             'bw_ops': [],
#         },
#     ],

#     'link_degradation': [
#         {
#             'timestep': 10000,
#             'action': 'link_degradation_stage1',
#             'description': 'Link 0-4 at 60% (1.5 Mbps)',
#             'link_ops': [],
#             'bw_ops': [('s1', 's5', 1.5)],   # 60% of 2.5 Mbps
#         },
#         {
#             'timestep': 40000,
#             'action': 'link_degradation_stage2',
#             'description': 'Link 0-4 at 20% (0.5 Mbps)',
#             'link_ops': [],
#             'bw_ops': [('s1', 's5', 0.5)],
#         },
#         {
#             'timestep': 80000,
#             'action': 'link_degradation_stage3',
#             'description': 'Link 0-4 at 5% (0.125 Mbps)',
#             'link_ops': [],
#             'bw_ops': [('s1', 's5', 0.125)],
#         },
#         {
#             'timestep': 120000,
#             'action': 'link_recovery',
#             'description': 'Full recovery (2.5 Mbps)',
#             'link_ops': [],
#             'bw_ops': [('s1', 's5', 2.5)],
#         },
#         {
#             'timestep': 150000,
#             'action': 'link_degradation_stage2',
#             'description': 'Second cycle: 20% (0.5 Mbps)',
#             'link_ops': [],
#             'bw_ops': [('s1', 's5', 0.5)],
#         },
#     ],
# }


# # =============================================================================
# # MININET HELPERS
# # =============================================================================

# def apply_link_ops(net, link_ops):
#     """Apply link up/down operations."""
#     for sw1, sw2, status in link_ops:
#         print(f"    configLinkStatus({sw1}, {sw2}, {status})")
#         net.configLinkStatus(sw1, sw2, status)


# def apply_bw_ops(net, bw_ops):
#     """
#     Apply bandwidth changes to Mininet links using tc.
#     This is for degradation scenarios where the link stays up
#     but capacity is reduced.
#     """
#     for sw1_name, sw2_name, new_bw_mbps in bw_ops:
#         node1 = net.getNodeByName(sw1_name)
#         node2 = net.getNodeByName(sw2_name)
#         rate_kbps = int(new_bw_mbps * 1000)

#         # Apply to both directions
#         for src, dst in [(node1, node2), (node2, node1)]:
#             for intf in src.intfList():
#                 if intf.link:
#                     other = intf.link.intf1 if intf.link.intf2 == intf else intf.link.intf2
#                     if other.node == dst:
#                         intf.cmd(f'tc qdisc del dev {intf.name} root 2>/dev/null || true')
#                         intf.cmd(f'tc qdisc add dev {intf.name} root handle 1: htb default 1')
#                         intf.cmd(f'tc class add dev {intf.name} parent 1: classid 1:1 htb rate {rate_kbps}kbit ceil {rate_kbps}kbit')
#                         print(f"    {intf.name}: rate limited to {rate_kbps} kbps")


# # =============================================================================
# # TOPOLOGY (same as original DRL-OR testbed)
# # =============================================================================

# class CustomTopo(Topo):
#     def __init__(self, nodeNum, linkSet, bandwidths, losses, **opts):
#         Topo.__init__(self, **opts)
#         self.__nodenum = nodeNum
#         self.__linkset = linkSet
#         self.__bandwidths = bandwidths
#         self.__losses = losses
#         self.__switches = []
#         self.__hosts = []
#         self.create_net()
#         self.add_hosts()

#     def create_net(self):
#         for i in range(self.__nodenum):
#             self.__switches.append(self.addSwitch("s" + str(i + 1)))
#         for i in range(len(self.__linkset)):
#             node1 = self.__linkset[i][0]
#             node2 = self.__linkset[i][1]
#             self.addLink(self.__switches[node1], self.__switches[node2],
#                          bw=self.__bandwidths[i], delay='5ms',
#                          loss=self.__losses[i], max_queue_size=1000)

#     def add_hosts(self):
#         if self.__nodenum >= 255:
#             print("ERROR!!!")
#             exit()
#         for i in range(self.__nodenum):
#             self.__hosts.append(self.addHost(
#                 "h" + str(i + 1),
#                 mac=("00:00:00:00:00:%02x" % (i + 1)),
#                 ip="10.0.0." + str(i + 1)))
#             self.addLink(self.__switches[i], self.__hosts[i],
#                          bw=1000, delay='0ms')


# # =============================================================================
# # REQUEST GENERATION (same as original DRL-OR testbed)
# # =============================================================================

# def generate_request(net, src, src_port, dst, dst_port, rtype, demand, rtime, time_step):
#     TIME_OUT = 5
#     src_host = net.hosts[src]
#     dst_host = net.hosts[dst]

#     popens = {}
#     popens[dst_host] = dst_host.popen(
#         "python3 server.py %s %d %d %d %d" % (dst_host.IP(), dst_port, rtime, rtype, time_step))
#     time.sleep(0.1)
#     popens[src_host] = src_host.popen(
#         "python3 client.py %s %d %s %d %d %d %d" % (
#             dst_host.IP(), dst_port, src_host.IP(), src_port, demand, rtime, time_step))

#     src_popen = popens[src_host]
#     dst_popen = popens[dst_host]
#     ind = 0
#     time_stamp = time.time()
#     for host, line in pmonitor(popens):
#         if time.time() - time_stamp > TIME_OUT:
#             print("Request:", "src:", src, "dst:", dst, "rtype:", rtype, "demand:", demand)
#             delay = TIME_OUT * 1000
#             throughput = 0
#             loss = 1.
#             print("time out!")
#             break
#         if host:
#             ind += 1
#             if ind == 1:
#                 delay, throughput, loss = parse_server_output(line)
#                 break

#     return delay, throughput, loss, [src_popen, dst_popen]


# def parse_server_output(line):
#     """
#     Parse server.py output which can be in two formats:
#       Format 1 (labeled):  "delay: 12.3 ms throughput: 1500.0 Kbps loss_rate: 0.01"
#       Format 2 (raw):      "12.3 1500.0 0.01"
#     """
#     line = line.strip()
#     if 'delay:' in line:
#         # Labeled format from server.py:
#         # "delay: %f ms throughput: %f Kbps loss_rate: %f"
#         parts = line.split()
#         # parts: ['delay:', '12.3', 'ms', 'throughput:', '1500.0', 'Kbps', 'loss_rate:', '0.01']
#         try:
#             delay = float(parts[1])
#             throughput = float(parts[4])
#             loss = float(parts[7])
#         except (IndexError, ValueError) as e:
#             print(f"[WARN] Failed to parse labeled output: {line} -> {e}")
#             delay, throughput, loss = 5000.0, 0.0, 1.0
#     else:
#         # Raw numeric format: "12.3 1500.0 0.01"
#         parts = line.split()
#         try:
#             delay = float(parts[0])
#             throughput = float(parts[1])
#             loss = float(parts[2])
#         except (IndexError, ValueError) as e:
#             print(f"[WARN] Failed to parse raw output: {line} -> {e}")
#             delay, throughput, loss = 5000.0, 0.0, 1.0
#     return delay, throughput, loss


# # =============================================================================
# # MAIN — event-driven, replaces hardcoded if/else
# # =============================================================================

# def main(scenario='initialization'):
#     """
#     Main testbed function with event-driven scenario support.

#     BEFORE (original DRL-OR):
#         if time_step == 10000:
#             # manually comment/uncomment one of:
#             # net.configLinkStatus('s1', 's5', 'down')
#             # ret['change'] = 'link_failure'
#             # ret['change'] = "demand_change"
#             # pass  # initialization

#     AFTER (this version):
#         Events are automatically triggered from SCENARIO_EVENTS table.
#         Just pass the scenario name on command line.
#     """

#     print(f"\n{'='*60}")
#     print(f"  TESTBED — Scenario: {scenario}")
#     events = SCENARIO_EVENTS.get(scenario, [])
#     print(f"  Events: {len(events)}")
#     for ev in events:
#         print(f"    t={ev['timestep']:>7,}: {ev.get('description', ev['action'])}")
#     print(f"{'='*60}\n")

#     # --- Read topology ---
#     toponame = "Abi"  # Change if needed
#     topo_file = f"../gat-mappo/net_env/inputs/{toponame}/{toponame}.txt"
#     with open(topo_file) as f:
#         lines = f.readlines()
#     first_line = lines[0].strip().split()
#     nodeNum = int(first_line[0])
#     linkNum = int(first_line[1])

#     linkSet = []
#     bandwidths = []
#     losses = []
#     for i in range(1, linkNum + 1):
#         parts = lines[i].strip().split()
#         n1 = int(parts[0]) - 1
#         n2 = int(parts[1]) - 1
#         linkSet.append((n1, n2))
#         bandwidths.append(float(parts[2]) / 1000.0)  # Kbps -> Mbps
#         losses.append(float(parts[4]))

#     # --- Mininet setup ---
#     topo = CustomTopo(nodeNum, linkSet, bandwidths, losses)
#     CONTROLLER_IP = "127.0.0.1"
#     CONTROLLER_PORT = 5001
#     OVSSwitch13 = partial(OVSSwitch, protocols='OpenFlow13')
#     net = Mininet(topo=topo, switch=OVSSwitch13, link=TCLink, controller=None)
#     net.addController('controller', controller=RemoteController,
#                       ip=CONTROLLER_IP, port=CONTROLLER_PORT)
#     net.start()

#     # --- Socket to simenv ---
#     print("Waiting for simenv connection...")
#     TCP_IP = "127.0.0.1"
#     TCP_PORT = 5000
#     BUFFER_SIZE = 1024
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.bind((TCP_IP, TCP_PORT))
#     s.listen(1)
#     conn, addr = s.accept()
#     print(f"Connected: {addr}")

#     requests_pq = []
#     time_step = 0
#     event_idx = 0  # <-- tracks which event to trigger next

#     while True:
#         try:
#             msg = conn.recv(BUFFER_SIZE)
#         except:
#             break

#         # Clean up expired requests
#         while len(requests_pq) > 0 and requests_pq[0][0] <= time_step:
#             ind, popens = heapq.heappop(requests_pq)
#             popens[0].kill()
#             popens[1].kill()

#         data_js = json.loads(msg)
#         rtime = data_js['rtime']
#         delay, throughput, loss, popens = generate_request(
#             net, data_js['src'], data_js['src_port'],
#             data_js['dst'], data_js['dst_port'],
#             data_js['rtype'], data_js['demand'],
#             1000000, time_step)

#         heapq.heappush(requests_pq, (rtime + time_step, popens))

#         ret = {
#             'delay': delay,
#             'throughput': throughput,
#             'loss': loss,
#         }

#         # ============================================================
#         # EVENT-DRIVEN SCENARIO HANDLING
#         # Replaces:  if time_step == 10000: ...
#         # ============================================================
#         while event_idx < len(events) and time_step >= events[event_idx]['timestep']:
#             ev = events[event_idx]
#             print(f"\n[TESTBED t={time_step}] >>> {ev.get('description', ev['action'])}")

#             # Apply Mininet link up/down
#             if ev.get('link_ops'):
#                 apply_link_ops(net, ev['link_ops'])

#             # Apply bandwidth changes (degradation)
#             if ev.get('bw_ops'):
#                 apply_bw_ops(net, ev['bw_ops'])

#             # Tell simenv about the change
#             ret['change'] = ev['action']

#             event_idx += 1

#         msg = json.dumps(ret)
#         conn.send(msg.encode())
#         time_step += 1

#     CLI(net)


# if __name__ == "__main__":
#     scenario = sys.argv[1] if len(sys.argv) > 1 else 'initialization'
#     if scenario not in SCENARIO_EVENTS:
#         print(f"Unknown scenario: {scenario}")
#         print(f"Available: {list(SCENARIO_EVENTS.keys())}")
#         sys.exit(1)
#     main(scenario)