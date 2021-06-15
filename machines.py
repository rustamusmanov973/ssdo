import sys
import os
import json
machs = json.loads(open(os.path.expanduser("~/.mach_attrs_corred.json")).read())

# print('machines.py')
P.infer_cls = False
P.check_existance = False

class Server:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            if any([k.startswith(i) for i in ['e', 'f', "d"]]):
                v = P(v)
            self.__setattr__(k, v)
    def run(self, command, **kwargs):
        import fabric
        # con = fabric.Connection(cur_ser.alt.a_host_alias)
        con = fabric.Connection(self.a_host_alias)
        if 'cwd' in kwargs.keys():
            cwd = kwargs['cwd']
            del kwargs['cwd']
            command = f"cd {cwd}; {command}"
        con.run(command, **kwargs)

    def emc(self, cmd_, fab_connection=None):
        emc_cmd_file_name = P(f"/tmp/emc_cmd_{self.a_ola}.el")
        emc_cmd_file_name.write_text(cmd_)
        emc_cmd_file_name = P(f"/ssh:{cur_ser.a_host_alias}:/tmp/emc_cmd_{self.a_ola}.el")
        far_cmd = f"""
        emacsclient -s "{self.d_hes}/{self.a_ola}e0" --eval '(load-file "{emc_cmd_file_name}")'
        """
        self.run(far_cmd.strip("\n "))


    def cp(self, text_):
        copy_text_file_name = P(f"/tmp/copy_text_{self.a_ola}")
        copy_text_file_name.write_text(text_)
        copy_text_file_name = P(f"/ssh:{cur_ser.a_host_alias}:/tmp/copy_text_{self.a_ola}")
        emc_copy_command = f'(with-temp-file "{copy_text_file_name}" (kill-region (point-min) (point-max)))'
        self.emc(emc_copy_command)



mid = os.popen('cat /etc/machine-id 2>/dev/null').read().strip()
cur_ser = None
all_machines = {}
for ser_n, kws in machs.items():
    server = Server(**kws)
    from copy import copy
    all_machines[ser_n] = copy(server)
    get_ipython().ex(f'{ser_n} = all_machines[ser_n]')


gpu.alt = lom2
srv.alt = lom2
lom2.alt = gpu

for ser_n, server in all_machines.items():
    if server.a_mid == mid:
        get_ipython().ex('cur_ser = server')
if 'get_ipython' in globals().keys():
    for thing in dir(cur_ser):
        if thing.startswith("_"):
            continue
        get_ipython().ex(f"{thing} = cur_ser.{thing}")

