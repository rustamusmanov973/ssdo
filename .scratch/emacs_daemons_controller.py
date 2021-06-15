import argparse

parser = argparse.ArgumentParser(description='Emacs daemons controller')
parser.add_argument('integers', metavar='N', type=int, nargs='*',
                    help='an integer for the accumulator')
parser.add_argument('--restart', dest='restart', action='store_true',
                    help='sum the integers (default: find the max)')
parser.add_argument('--list', dest='list', action='store_true',
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
import os
hostname = os.popen("hostname").read().strip()
emacs_servers = os.popen("ls ~/.emacs.d/servers/*").read().strip().split()
print(f"ls ~/.emacs.d/servers/*\n---->{emacs_servers}")
avail_srvs = []
db = 1
emx_pids = [[i.split()[1], i] for i in os.popen("ps aux | grep emacs$")]
print("emx_pids:")
for k,j in emx_pids:
    print(k, j)
for em_server in emacs_servers:
    check_avail_cmd_1 = f"""emacsclient -s '{em_server}' --eval "(emacs-pid)" """
    check_avail_cmd_2 = f"""emacsclient -s '{em_server}' --eval "server-name" """
    pid = os.popen(check_avail_cmd_1).read().strip()
    srv_name = os.popen(check_avail_cmd_2).read().strip().strip('"')
    if srv_name in [i.split('/')[-1] for i in emacs_servers]:
        avail_srvs.append([em_server, pid, srv_name])

unavail_pids = list(set([a[0] for a in emx_pids]) - set([i[1] for i in avail_srvs]))
idle_srvs = list(set(emacs_servers) - set([i[0] for i in avail_srvs]))
print("Before:", "avail_srvs", avail_srvs, "idle_srvs", idle_srvs)
# Killing emacs processes that have no server
for pid in unavail_pids:
    if input(f"Detected emacs without server running with pid: {pid}, kill it? [Y/n]") == "Y":
        os.popen(f"kill -9 {pid}")
    else:
        print("Aborted.")
for server in idle_srvs:
    cmd = f'rm "{server}"'
    if input(f"Detected emacs socket file: {server}. Delete it with <|{cmd}|>? [Y/n]") == "Y":
        os.system(cmd)
    else:
        print("Aborted.")

print("After:", "avail_srvs", avail_srvs, "idle_srvs", idle_srvs)

if args.integers:
    input_servers = ["%se%s".format(hostname[0], integer) for integer in args.integers]
    print(args.integers)


print("kjkj")
