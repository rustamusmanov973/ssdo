# print("dgfl started")
from scipy import stats
import subprocess
# import cupy as cp
import numpy as np
import ipython_blocking
from collections import defaultdict
from textwrap import dedent
import textwrap
import pandas as pd
import inspect
import pybel
import pdb
import time
import fabric
import textwrap
from modeller import *
import traceback
# if l2:
#     os.environ["GMXLIB"] = "/mnt/scratch/users/fbbstudent/work/rustam/dgfl/md/GLB_0_0"
from pymol import cmd, stored

import re
import ipywidgets as widgets
from fnmatch import fnmatch
from braceexpand import braceexpand
import gromacs

from IPython.display import clear_output, Image, display_javascript, Javascript
from multiprocessing import Pool
from Bio.PDB import PDBParser
from numpy import array
import sysrsync
import datetime
from IPython.core.display import display, HTML
from rdkit import Chem
from xmlrpc.client import ServerProxy
import matplotlib.pyplot as plt
from rdkit.Chem.PyMol import MolViewer
from matplotlib import gridspec
import rdkit.Chem.Draw.IPythonConsole  # pretty rdchem.Mol visualisations
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from ipywidgets import widgets
import sqlalchemy
from pandas import DataFrame, read_table, option_context
import pylab as pl
from time import sleep as sl
from numpy import random as nprandom
from io import StringIO
import types
import __main__
import sys
import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.rms
import MDAnalysis.analysisis.encore
import MDAnalysis.analysis.distances
from types import FunctionType
from pathlib import PosixPath
import os
import pymysql
import ipysheet
import uuid
import modeller
from invoke import run
from multiprocessing import Process
import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# STOP
if 'cur_ser' in globals().keys():
    pml = ServerProxy(uri=f'http://localhost:{cur_ser.a_xmlRpcPort}/RPC2')
else:
    print('pml will be defined ...')
# import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'

if 'P' in globals().keys():
    P.infer_cls = True
    P.check_existance = False

if 'get_ipython' in globals().keys():
    import ipdb
    import nglview as nv
    from IPython import get_ipython
    import ipython_blocking
    ipython_blocking.load_ipython_extensions()
    from ipysheet import sheet, cell, row, column, cell_range

    # c = fabric.Connection(cur_ser.alt.a_host_alias)
    # c = fabric.Connection(cur_ser.alt.a_host_alias)
    # c = fabric.Connection(cur_ser.alt.a_host_alias)
    r = fabric.Connection("localhost", user="ec", port=22122)


print_ = print
def prin(x, *args, **kwargs):
    print_(f'{datetime.datetime.now()}: {x}', *args, **kwargs)
# def lg(message, name=f'ag_{os.getpid()}'):
#     # apv(message, name)
#     print(f'ALG: {message}')
# def lge(message, name=f'ag_{os.getpid()}'):
#     # apv(message, name)
#     print(f'ERRROR: {message}'.center(100, '='))
def printList(*list_):
    str_pat = u"->".join(["{}" for j in range(len(list_))])
    return str_pat.format(*list_)
def tmp_pickle_path_resolver(var_name, md_dir=None):
    if md_dir:
        double_name = md_dir.s.replace("/",">>")
        pickle_path = d__tdir.joinpath(f'|{double_name}|__|{var_name}|')
    else:
        pickle_path = d__tdir.joinpath(f'{var_name}')
    return pickle_path
# def wrv(variable, var_name, md_dir=None, db=l2, full=False):
def wrv(variable, var_name, md_dir=None, db=0, full=False):
    import pickle
    pickle_path = tmp_pickle_path_resolver(var_name, md_dir=md_dir)
    if db: print(f'md_dir: {md_dir}; wrv: {pickle_path} {str(variable)[:20]}')
    timestamp = datetime.datetime.now()
    if not full:
        variable = {
            'data': variable,
            'time': timestamp,
            'id_' : 0
                    }
        if md_dir:
            variable['id_'] = os.stat(md_dir.s).st_ino
    pickle.dump(variable, pickle_path.open("wb"))
def rmv(var_name, md_dir=None, db=0, if_not_exists='IOError', full=False):
    import pickle
    pickle_path = tmp_pickle_path_resolver(var_name, md_dir=md_dir)
    pickle_path.rm()

def rdv(var_name, md_dir=None, db=0, if_not_exists='IOError', full=False):
    import pickle
    pickle_path = tmp_pickle_path_resolver(var_name, md_dir=md_dir)
    if not P(pickle_path).exists():
        return if_not_exists
    if db:
        print(f'rdv: {pickle_path}')
    value =  pickle.load(open(pickle_path, "rb"))
    if md_dir:
        written_id_ =  value['id_']
        true_id_ = os.stat(md_dir.s).st_ino
        if written_id_ != true_id_:
            print(f'\
            Inode mismatch. written_id_ {written_id_} != true_id_ {true_id_}. \
            Removing tmp dir: {pickle_path}\
            Returning IOError not really).\
            ')
            return value['data']
            # pickle_path.rm()
            # return 'IOError'
    if full:
        return value
    else:
        return value['data']

def apv(variable, var_name, md_dir=None, db=False, if_not_exists=''):
    old_value = rdv(var_name, md_dir=md_dir, if_not_exists=if_not_exists, full=True)
    if isinstance(old_value, str):
        wrv(old_value + '\n' + variable, var_name, md_dir=md_dir)
    elif isinstance(old_value, list):
        wrv(old_value + [variable], var_name, md_dir=md_dir)

def tmp_pickle_str2path(tmp_str):
    match = re.search(f'\|(.*?)\|__\|(.*?)\|', tmp_str)
    return match.group(1).replace('>>', '/'), match.group(2)

def tdir_audit(tmp_dir=0):
    if not tmp_dir:
        tmp_dir = cur_ser.d__tdir
    for file_ in tmp_dir.ls:
        if '>>' in file_.name:
            pickle_path, var_name = tmp_pickle_str2path(file_.name)
            pickle_path = P(pickle_path)
            if not pickle_path.exists():
                print(f'XXX Path {pickle_path} not exists')
                file_.rm()
                continue
            record = pickle.load(open(file_.s, "rb"))
            timestamp = datetime.datetime.now()
            if not isinstance(record, dict) or 'data' not in record.keys():
                print(f'Rewriting old-style path: {pickle_path}. file in t_dir: {file_}.')
                variable = {
                    'data': record,
                    'time': timestamp,
                    'id_' : os.stat(pickle_path.s).st_ino
                }
                print(variable)
                pickle.dump(variable, open(file_.s, "wb"))
            else:
                written_id_ =  record['id_']
                true_id_ = os.stat(pickle_path.s).st_ino
                if true_id_ != written_id_:
                    print(f'removing old tmp file: {file_}')
                # file_.rm()

# def optimize(atmsel, sched):
#     # conjugate gradient
#     for step in sched:
#         step.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)
#     # md
#     refine(atmsel)
#     cg = conjugate_gradients()
#     cg.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)
# # molecular dynamics
# def refine(atmsel):
#     # at T=1000, max_atom_shift for 4fs is cca 0.15 A.
#     md = molecular_dynamics(cap_atom_shift=0.39, md_time_step=4.0,
#                             md_return='FINAL')
#     init_vel = True
#     for (its, equil, temps) in ((200, 20, (150.0, 250.0, 400.0, 700.0, 1000.0)),
#                                 (200, 600,
#                                  (1000.0, 800.0, 600.0, 500.0, 400.0, 300.0))):
        # for temp in temps:
        #     md.optimize(atmsel, init_velocities=init_vh(f'H3D_{sd.name}_{frame}')
        #     if isinstance(dir_.rdv('bin_'), tuple) and len(dir_.rdv('bin_')) > 1:
        #         if dir_.rdv('status') == 'FM' and dir_.rdv('bin_')[1] < 8000:
        #             rmdirs.append(dir_)
def highlight_hrx_muts(x):
    hrx_mut_columns = [i for i in df.columns if int(i[1:]) in hrx_muts.keys()]
    r = 'background-color: orange'
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1[df1.columns] = 'background-color: yellow'
    df1[hrx_mut_columns] = r
    return df1
# def flush_status():
#     for mdir in cur_ser.mdir.mdirs:
#         mdir.wrv('FM', 'status')
def sq():
    run('squeue -u fbbstudent')
def sqdf():
    ex_time = os.popen('squeue -o "%A     %S" -u fbbstudent').read()
    ex_time_df = pd.read_fwf(StringIO(ex_time))
    k = os.popen('squeue -u fbbstudent').read()
    sqdf = pd.read_fwf(StringIO(k))
    sqdf = sqdf.merge(ex_time_df, how='outer').set_index('JOBID')
    return sqdf
def lg_result(message, name=f'results'):
    print(f'LG_RESULT: {message}')
    apv(f'AG#{ag_num}: {message}', name)
def agent_lg(func):
    def decorated_function(*args, **kwargs):
        t0 = time.time()
        start_mes = f'{func.__name__} started ...'
        lg(start_mes)
        result = func(*args, **kwargs)
        t1 = time.time()
        delta_t = datetime.timedelta(seconds=t1 - t0)
        end_mess = f'{func.__name__} ended ...: {delta_t}'
        lg(end_mess)
        return result

    return decorated_function
def status(func):
    def decorated_function(*args, **kwargs):
        t0 = time.time()
        start_mes = f'{args[0].name:.<10}.{func.__name__:.<30} started ...'
        print(start_mes)
        if not args[0].exists():
            args[0].mkdir()
        now = datetime.datetime.now()
        fun_start_info_string = f'|{func.__name__:25}|__|STARTED|__|{now}|'
        args[0].wrv(fun_start_info_string, 'status')
        status_history = args[0].rdv('status_history', if_not_exists=[])
        if isinstance(status_history, str):
            status_history = []
        if func.__name__ == 'initialize':
            status_history = [] #rewriting history
        if func.__name__ == 'make_and_analyze_cv_table':
            for _ in range(2):
                if status_history[-1].split("|")[1].strip() in ['make_cv_table', 'make_and_analyze_cv_table']:
                    status_history = status_history[:-1]
        status_history.append(fun_start_info_string)
        status_history = args[0].wrv(status_history, 'status_history')
        result = func(*args, **kwargs)
        t1 = time.time()
        now = datetime.datetime.now()
        delta_t = datetime.timedelta(seconds=t1 - t0)
        end_mess = f'{args[0].name:<10}.{func.__name__:.<30} ended ...: {delta_t}'
        print(end_mess)
        fun_start_info_string = f'|{func.__name__:25}|__|ENDED  |__|{now}|__|DT:{delta_t}|'
        status_history = args[0].rdv('status_history', if_not_exists=[])
        status_history.append(fun_start_info_string)
        status_history = args[0].wrv(status_history, 'status_history')
        args[0].wrv(fun_start_info_string, 'status')
        return result

    return decorated_function
def watch_fun(func):
    def decorated_function(*args, **kwargs):
        old_result = None
        while True:
            if 'get_ipython' in globals().keys():
                t0 = time.time()
                result = func(*args, **kwargs)
                # if result == old_result:
                #     continue
                t1 = time.time()
                delta_t = datetime.timedelta(seconds=t1 - t0)
                print(f'{func.__name__} ended ...: {delta_t}')
                print(f'result: {result}')
                sl(1)

    return decorated_function
def print_universe(func):
    def decorated_function(*args, **kwargs):
        t0 = time.time()
        print(f'{func.__name__} started ...')
        result = func(*args, **kwargs)
        t1 = time.time()
        delta_t = datetime.timedelta(seconds=t1 - t0)
        if not result:
            result_s = "NoneType"
        else:
            result_s = len(result.trajectory)
        print(f'{func.__name__} ended ...: {delta_t}. Traj:. Len: {result_s}')
        return result

    return decorated_function

@for_all_methods(status)
class E(P):
    # def __new__(cls, *args, **kwargs):
    #     return super().__new__(....)

    def __init__(self, path, db=0):
        super().__init__(path)
        self.db = db
        self.ri = self.joinpath('ri')
        self.cg = self.joinpath('ri/cg')
        self.mp = self.joinpath('ri/cg/mp')
        self.ss = self.joinpath('ri/ss')
        self.s_str = self.joinpath('ri/ss/lpla_lg_0.pdb')
        self.s_str2 = self.joinpath('ri/ss/lpla_lg2_0.pdb')
        self.tmp = self.joinpath('ri/.tmp')
        self.msc = self.joinpath('ri/msc')
        self.ro = self.joinpath('ro')
        self.scorefile = self.joinpath('ro/scorefile.txt')
        self.rpr = self.joinpath('rpr')

        self.e_dirs = Ps([self.ri, self.cg, self.mp, self.ss,
                          self.tmp, self.msc, self.ro, self.rpr])

    @property
    def ara(self):
        print('ara Aha2')


    @property
    def decoys(self):
        if [*self.ro.rglob('lpla_lg2_0__DE_*')]:
            return Ps([P(i) for i in self.ro.rglob('lpla_lg2_0__DE_*')])

    def initialize(self):  # ed_AGT is RR3 without garbage
        self.rwdir()
        self.e_dirs.mkdir()


        self.tmp.mkdir()
    def pymolize_traj_files(self):
        pm(
            f'load {self.ss.joinpath("prot_from_traj.pdb")}',
            f'load {self.cg.joinpath("lig_from_traj.pdb")}',
            f'alter org, chain="X"',
            f'alter org, resn="UNL"',
            f'alter polymer, chain="A"',
            f'save {self.s_str}',
            f'save {self.cg.joinpath("lig.pdb")}, org',
            db=self.db
        )

    def prepare_mop(self, db=0):
        pybel_molecule = [*pybel.readfile('pdb', self.cg.joinpath('lig.pdb').str)][0]
        assert pybel_molecule, 'corrupted / empty pdb file: ' + self.cg.joinpath('lig.pdb')
        pybel_molecule.write(
            format='mop', filename=self.mp.joinpath("lig.mop").str,
            opt={'k': f'PM6 CHARGE={args.net_charge} PRECISE pKa EF THREADS=14'},
            overwrite=True
        )
        mpc_frz(self.mp.joinpath('lig.mop').str)

    def mop_opt(self):
        my_env = os.environ.copy()
        my_env.update(cur_ser.a_mopac_env_vars)
        self.mp.run(f'{cur_ser.e__mopac_exe} {self.mp["lig.mop"]}', env=my_env)

    def mop_out_to_mol2(self):
        mop_out = [*pybel.readfile('mopout', self.mp.joinpath('lig.out').str)][0]
        mop_out.write('mol2', self.mp.joinpath('lig.mol2').str, overwrite=True)

    def gen_prms(self):


        self.cg.run(f'python2 {cur_ser.e__molfile2params_exe} -n UNL -p UNL --conformers-in-one-file --clobber {self.mp["lig.mol2"]}')

    def gen_s_str(self):
        assert self.cg.joinpath('UNL.pdb').exists()
        pm(
            f'load {self.s_str}',
            'remove org',
            f'load {self.cg.joinpath("UNL.pdb")}',
            f'save {self.s_str}'
        )

    def gen_s_str2(self):
        complex_file = self.s_str.read_text().split('\n')
        complex_file2 = []
        at = 0
        for string in complex_file:
            if at and string.startswith('HET'):
                complex_file2.append('TER')
            at = string.startswith('ATOM')
            if string.startswith('CONE'):
                continue
            complex_file2.append(string)
        self.s_str2.write_text('\n'.join(complex_file2))

    def generate_design_flags(self, *args, **kwargs):
        kwargs['file_name'] = str(self.msc.joinpath('ed.flags'))
        gnr_dsn_flgs(*args, **kwargs)

    def generate_relax_flags(self, *args, **kwargs):
        kwargs['file_name'] = str(self.msc.joinpath('ed.flags'))
        gnr_dsn_flgs(*args, **kwargs)

    def generate_resfile(self, db=1):
        if args.quick:
            rst_str_lst = ['NATAA', 'START', '', '2 A PIKAA W']
        else:
            rst_str_lst = ['START', '', '1 X NATAA']
        for cat_AA in args.cat_aa:
            rst_str_lst.append(f'{cat_AA} A NATRO')
        self.msc['LIG.resfile'].write_text('\n'.join(rst_str_lst))

    # def prepare_ed_run(self):
    def launch(self):
        self.pymolize_traj_files()
        self.prepare_mop()
        self.mop_opt()
        self.mop_out_to_mol2()
        self.gen_prms()
        self.gen_s_str()
        self.gen_s_str2()
        self.generate_design_flags(db=1)
        self.generate_resfile(db=1)
        self.run_ed()
        # self.make_visu_df()
        # # self.make_visu_pse()

        # self.prepare_kids()
        # self.launch_kids()

    def run_ed(self, db=0):
        self.wrv('RE', 'status')
        print(f'Rewriting output directory: {self.ro}')
        self.ro.rwdir()
        cmd_ = [cur_ser.e__ed_exe, '@ri/msc/ed.flags',
                '-out:path:pdb', self.ro,
                '-out:file:o', self.ro.joinpath('scorefile.txt'),
                '-in:file:s', self.s_str2,
                '-extra_res_fa', self.cg.joinpath('UNL.params')
                ]
        print('=====> Launching ed!!:')
        cmd_str = ' '.join([*map(str, cmd_)]) if isinstance(cmd_, list) else str(cmd_)
        print(cmd_str)
        self['log'].rm()
        proc = subprocess.Popen(
            cmd_,
            stdout=open(self['log'].s, "ab"),
            stderr=open(self['log'].s, "ab"),
            shell=isinstance(cmd_, str),
            cwd=self
        )
        self.wrv(proc.pid, 'proc_pid', db=1)
        # proc.wait()

        # return proc

    def stop_ed(self):
        import signal
        proc_pid = self.rdv('proc_pid')
        os.killpg(os.getpgid(proc_pid), signal.SIGTERM)

    def run_rlx(self, db=0):
        self.wrv('RE', 'status')
        print(f'Rewriting output directory: {self.ro}')
        self.ro.rwdir()
        cmd_ = [cur_ser.e__rlx_exe, '@ri/msc/rlx.flags',
                '-out:path:pdb', self.ro,
                '-out:file:o', self.ro.joinpath('scorefile.txt'),
                '-in:file:s', self.s_str2,
                '-extra_res_fa', self.cg.joinpath('UNL.params')
                ]

        print('=====> Launching ed!!:')
        cmd_str = ' '.join([*map(str, cmd_)]) if isinstance(cmd_, list) else str(cmd_)
        print(cmd_str)

        proc = subprocess.Popen(
            cmd_,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=isinstance(cmd_, str),
            cwd=self
        )

        if db:
            live_process_output(proc, return_=0)
        return proc

    def make_visu(self):
        self.make_visu_df()
        self.make_visu_pse()

    def make_visu_df(self, return_dec2muts_df=True, lpla=True):
        ed_df = pd.read_fwf(self.scorefile.str)
        d = defaultdict(lambda: defaultdict(int))
        for __DE_i in ed_df.description:
            # s_str = cur_ser.d__sdir.joinpath('3a7r.pdb') if lpla else self.s_str
            s_str = args.E_pdb
            # lpla =
            ss_str = PDBParser().get_structure("self.s_str", s_str)
            dec_str = PDBParser().get_structure("dec_str", self.ro.joinpath(__DE_i + '.pdb'))

            dec_residues = dec_str.child_list[0].child_list[0].child_list
            ss_residues = ss_str.child_list[0].child_list[0].child_list
            for residue_index in range(len(dec_residues)):
                ss_residue, dec_residue = *map(lambda x: aaMap[x[residue_index].get_resname()],
                                               [ss_residues, dec_residues]),
                if ss_residue != dec_residue:
                    d[f"{ss_residue}{residue_index + 1}"][__DE_i] = dec_residue
        self.decs2muts_df = DataFrame(d).fillna('')
        self.summary_df = pd.merge(self.decs2muts_df, ed_df, right_on='description', left_index=True)
        self.summary_df = self.summary_df.set_index(self.summary_df.pop('description'))

        self.wrv(self.summary_df, 'summary_df')
        self.wrv(self.decs2muts_df, 'decs2muts_df')

        if return_dec2muts_df:
            return self.decs2muts_df
        else:
            return self.summary_df.fillna('')

    def make_visu_pse(self):
        print("run make_visu_df first!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        pm(*[f'load {ed_dir}, {ed_dir.double_name_prefix}' for ed_dir in self.decoys])
        decVisu()
        cmnds, resis = [], []
        for ed_dir in self.decoys:
            ed_dir_name = ed_dir.name_prefix
            for column in self.decs2muts_df.columns:  # = 'V151'
                resi = column[1:]  # 151
                resis.append(resi)
                mutation = self.decs2muts_df.loc[ed_dir_name][column]  # = "A"
                if mutation:
                    cmnds += [f'select mut_residue, {ed_dir.double_name_prefix} & i. {resi}',
                              f'color pink, mut_residue',
                              f'label mut_residue & n. CB, "{column + mutation}"', ]
        cmnds += [f'select mut_residues, i. ' + "+".join(resis), 'hide spheres, e. h',
                  "set pse_export_version, 1.721",
                  f'save {self.joinpath("ed.pse")}']

        pm(*cmnds, r=False)

    def load_visu(self):
        pma(f'load {self.joinpath("ed.pse")}')

    # NEW!
    @property
    def hrxs(self):
        hrxs = [i for i in self.dirs if i.name.startswith('HRX')]
        return hrxs

    # @status
    # def run_hrxs(self):
    #     for i, decoy in enumerate(self.decoys):
    #         hrx_d = self['HRX_{i}']
    #         hrx_d.initialize()
    #         hrx_d.prepare()
    #         hrx_d.run_hrx()
    #         hrx_d.make_cv_table()
    #         #  hrx_d.make_visu_hrx_1() # bad code
    #         hrx_d.make_min_cluster_centroids()
    #         hrx_d.run_plds()


    def prepare_kids(self):
        # Not clear, maybe some other method of self.kids formation
        n_dirs, dir_nam = args.stage2kids[self.stage]

        for dir_idx in range(n_dirs):
            kid = self[f'{dir_nam}_{dir_idx}__{self.stage + 1}']
            kid.initialize()
        if type(self.kid) == MM:
            for kid in self.kids:
                print('Preparation will be done in launch step: "load_structures()')
                pass


class M(P):
    def __new__(cls, path=''):
        return PosixPath.__new__(cls, path)

    def __init__(self, path):
        super().__init__(path)
        # for some_path in self.iterdir():
        # if some_path.is_good():
        #     setattr(self, some_path.nom, some_path)
        self.nom = self.str
        self.md_center_xtc = self.joinpath('md_center.xtc')
        self.solv_ions_gro = self.joinpath('solv_ions.gro')
        self.mdFit_xtc = self.joinpath('mdFit.xtc')
        # self.md_xtc = self.joinpath('md.xtc')
        # self.traj_trr = self.joinpath('traj.trr')
        # self.traj_comp_xtc = self.joinpath('traj_comp.xtc')
        # self.topol_tpr = self.joinpath('topol.tpr')
        self.mda_xtc = self.joinpath('mda.xtc')
        self.mda_gro = self.joinpath('mda.gro')
        self.aaa_bb_rmsd_xtc = self.joinpath('aaa_bb_rmsd.xtc')
        self.rmsd_xtc = self.joinpath('rmsd.xtc')
        self.ed = self.par
        self._tmp = self.joinpath('.tmp')
        self.unl_acpype = self.joinpath('unl.acpype')
        self.unl_NEW_pdb = self.unl_acpype.joinpath('unl_NEW.pdb')
        self.unl_GMX_gro = self.unl_acpype.joinpath('unl_GMX.gro')
        self.topol_top = self.joinpath('topol.top')
        self.good_frames = []
        self.important_contacts = [0, 1, 2, 3] + [5, 6, 7] + [10]

    def initialize(self):
        self.rwdir()  # it was mkdir
        self.unl_acpype.mkdir()
        self._tmp.mkdir()

    @property
    def md_xtc(self):
        traj_name =  self['traj_comp.xtc']
        return traj_name if traj_name.exists() else self['md.xtc']


    @property
    def md_tpr(self):
        traj_name =  self['topol.tpr']
        return traj_name if traj_name.exists() else self['md.tpr']

    @property
    def frame(self):
        return int(self.name.split('_')[-1])

    @property
    def rmsd(self):
        return rdv('rmsd', self)

    @property
    def aaa_bb_rmsd(self):
        return rdv('aaa_bb_rmsd', self)

    # def deploy_to_lom2(self):

    @property
    def h_data(self):
        return rdv('h_data', self)

    def compress_traj(self, pilot=1, sln='not type H and (resname UNL or (protein and (around 20 resname UNL)))', nf=40,
                      fs=0, db=1, start=0, stop=-1):

        # During agent act mdFit.xtc is not formed, while aaa_bb_rmsd_universe is.
        #
        # for file_name in ['md.tpr', 'mdFit.xtc']: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for file_name in ['md.tpr', 'aaa_bb_rmsd.xtc']:
            if not self.joinpath(file_name).exists():
                print("AAA", len(globals()), 'get_ipython' in globals().keys())

                warn(f"{file_name} does not exist")
                return 0

        u = self.get_aaa_bb_rmsd_universe()
        a_sln = u.select_atoms(sln)
        a_sln.write(self.mda_gro)
        if pilot:
            print('pilot: loading self.mda_gro...')
            pma(f'load {self.mda_gro}, {self.nom}_gro')
            input('Is the selection (in gro) OK? [press Enter]')
        ilen = len(u.trajectory)
        if nf and not fs:  # ???? ??????? ?????? ?????????? ??????
            fs = (ilen // nf) + 1
        olen = 0
        start_n_atoms = a_sln.n_atoms
        with MDAnalysis.Writer(str(self.mda_xtc)) as W:
            for ts in u.trajectory[start:stop:fs]:
                print(
                    f"\rstart_n_atoms: {start_n_atoms}. Processing {ts.frame} frame of {ilen} ({self.mdFit_xtc}->{self.mda_xtc})",
                    end="")
                a_sln = u.select_atoms(sln)
                # if a_sln.n_atoms != start_n_atoms:
                #     print(f'Detected wrong number of atoms: {a_sln.n_atoms}')
                #     continue
                W.write(a_sln)
                olen += 1
        pymol_loading_speed = 3814598  # bites per second
        if db: print(
            f"\nSize: {self.mdFit_xtc.hsize} -> {self.mda_xtc.hsize} \t Len: {humanize.intcomma(ilen)} -> {humanize.intcomma(olen)}. Pymol loading ET: {datetime.timedelta(seconds=os.stat(self.mda_xtc).st_size / pymol_loading_speed)}")
        return 1

    def make_visu(self, *args, **kwargs):
        self.par['npt.gro'].copy_to(self['mda.gro'])
        self['']
        pmtr(mdirs=[self], *args, **kwargs)

    def load_visu(self):
        t = time.time()
        mda_pse_path = os.path.join(self, "mda.pse")
        mda_size = humanize.naturalsize(os.stat(mda_pse_path).st_size)
        print(f'Loading pse ({mda_size})...', end='')
        pma(f'load {mda_pse_path}')
        print(f"\rLoaded file ({mda_size}) in  {datetime.timedelta(seconds=time.time() - t)}")

    def unlink_gmx_backups(self, depth=1):
        directories = self.iterdir()
        [i.unlink() for i in directories if i.name.startswith('#') and i.name.endswith('#')]
        if depth == 2:
            for directory in self.chd:
                directory.unlink_gmx_backups()

    def make_good_frames(self):
        for h_idx, h_data_piece in enumerate(self.h_data):
            if h_idx not in [0, 1, 2, 3, 5, 6, 7]:
                continue
            if h_idx == 7:  # we add only 100 -200 bad frames to ~2000 -it's normal
                continue
            frame_set = set(h_data_piece[:, 0][(h_data_piece[:, 1] == 1)])
            if h_idx == 0:
                good_frames = frame_set
            good_frames = good_frames.intersection(frame_set)
            print(len(good_frames), end=' -> ')
        self.good_frames = [int(frame_n) for frame_n in good_frames]

    def to_res(self):
        suggested_res_path = mdires.joinpath(self.name)
        if suggested_res_path.exists():
            print('ERROR: path exists1!!!!!')
        else:
            self.move_to(suggested_res_path)
            for t1 in d__tdir.rglob(f'{self.name}*'):
                print(f'moving {t1}')
                t1.move_to(tdires)


    def rm_tmp(self, db=True):
        for tmp_file in d__tdir.rglob(f'{self.name}*'):
            tmp_file.rm()
            if db:
                print(f'Removed: {tmp_file}')

    def rm(self, remove_tmp=True):
        super().rm()
        self.rm_tmp()

    @print_universe
    def get_md_universe(self):
        while not (self.md_xtc.exists() and self.md_tpr.exists()):
            print(f' not self.md_xtc.exists() {self.md_xtc.exists()} and self.md_tpr.exists() {self.md_tpr.exists()}!')
            sl(.5)
        while True:
            try:
                u = mda.Universe(self.md_tpr, self.md_xtc)
                return u
            except Exception as e:
                print(f'get_md_universe (maybe  XDR read error = endoffile) Exception: {str(e)}')

    @print_universe
    def get_mdFit_universe(self):
        return mda.Universe(str(self.md_tpr), str(self.mdFit_xtc))

    @print_universe
    def get_trr_universe(self):
        return mda.Universe(self['topol.tpr'].s, self['traj.trr'].s)

    @print_universe
    def get_md_xtc_universe(self):
        return mda.Universe(str(self.md_tpr), str(self.md_trr))
        # return mda.Universe(str(self.md_tpr), str(self.md_xtc))
    @print_universe
    def get_aaa_bb_rmsd_universe(self):
        return mda.Universe(str(self.md_tpr), str(self.aaa_bb_rmsd_xtc))

    @print_universe
    def get_md_center_universe(self):
        # return mda.Universe(self.md_tpr, self.md_center_xtc)
        # return mda.Universe(self.md_center_xtc)
        if isinstance(self, MM):
            return mda.Universe(self['EX.gro'].s, self['1/md_center.xtc'].s)
        if isinstance(self, M):
            return mda.Universe(self['../EX.gro'].s, self['md_center.xtc'].s)

    # @status
    # def get_trr_universe(self):
    #     return mda.Universe(self['traj.trr'].s)

    @print_universe
    def get_solv_ions_universe(self):
        return mda.Universe(self.md_tpr, self.solv_ions_gro)

    def get_names_from_pdb_str(self, refu):
        refStr = sdir.joinpath('amp2.pdb')
        ligtr = rdpdb(refStr.str, sanitize=False)
        ligtr2 = Chem.RemoveHs(ligtr)
        self._tmp.mkdir()
        self._tmp.joinpath('litu').mkdir()
        litu_file = self._tmp.joinpath(
            f'litu/litu{random.randint(1, 10 ** 7)}.pdb'
        )
        refu.select_atoms("resname UNL").write(litu_file)
        litu3 = PDBParser().get_structure("", litu_file)
        litu = rdpdb(litu_file.str, sanitize=False)
        selAts = []
        rdMatch = litu.GetSubstructMatch(ligtr2)
        for atIdx, atName in enumerate(litu3[0].child_list[0].child_list[0].child_list):
            if atIdx in rdMatch:
                selAts.append((atIdx, atName.name))
        print(selAts)
        return (selAts)

    def make_md_center(self, b=0, e=10 ** 20, dt=1):
        self.md_center_xtc.rm()
        # gromacs.trjconv(
        #     s=self.md_tpr.name, f=self.md_xtc.name, o='md_center.xtc',
        #     center=True, pbc='mol', ur='compact',
        #     b=b, e=e, dt=dt,
        #     input=('4', '0'),
        #     cwd=self,
        #     stdout=subprocess.DEVNULL,
        #     stderr=subprocess.DEVNULL
        # )
        import subprocess
        gromacs.trjconv(
            s=self.md_tpr.name, f='traj_comp.xtc', o='md_center.xtc', n='../EX.ndx',
            # s=self.md_tpr.name, f='traj_comp.xtc', o='md_center.xtc',
            center=True, pbc='mol', ur='compact',
            b=b, e=e, dt=dt,
            input=('22', '22'),
            cwd=self,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # print(f'Finished: md.xtc ({self.md_xtc.hsize}) -> md_center.xtc({self.md_center_xtc.hsize})')


    def make_snapshots(self, b=0, e=10 ** 20, dt=1):
        import subprocess
        print("hello")
        self['snapshots'].rwdir()
        self['1/snapshots.err'].rm()
        self['1/snapshots.out'].rm()
        self.run(f"{args.gmx_path} trjconv -f traj.trr -s topol.tpr -o snapshots/frame.pdb -conect -skip 5 -sep -n ../prot_UNL.ndx -center -pbc mol -ur compact")
        gromacs.trjconv(
            s=self.md_tpr.name, f='traj.trr', o='snapshots/frame.pdb',
            n='../prot_UNL.ndx',
            center=True, pbc='mol', ur='compact',
            conect=True,
            sep=True,
            b=b, e=e, dt=dt,
            input=('1', '22'),
            cwd=self,
            stdout=open(self['snapshots.out'].s, "w"),
            stderr=open(self['snapshots.err'].s, "w")
        )
        # print(f'Finished: md.xtc ({self.md_xtc.hsize}) -> md_center.xtc({self.md_center_xtc.hsize})')


    def make_mdFit(self):
        refIds = [138, 140, 147, 149, 17, 20]
        md_center_u = self.get_md_center_universe()
        solv_u = self.get_solv_ions_universe()
        ligSel = "backbone and (resid {})".format(" ".join(str(x) for x in refIds))
        alignment = align.AlignTraj(
            md_center_u,
            solv_u,
            filename=self.mdFit_xtc.str,
            select=ligSel)
        alignment.run(
            step=1
        )
        print(
            f"Finished: md.xtc ({self.md_xtc.hsize}) -> mdFit.xtc ({self.mdFit_xtc.hsize}) alignment length: {len(alignment.rmsd)}")

    def make_rmsd(self, md_fit_step=1, start=0, stop=-1, append=False):
        solv_u = self.get_solv_ions_universe()
        md_center_u = self.get_md_center_universe()
        ampAts = self.get_names_from_pdb_str(solv_u)
        ligSels = [
            ("rmsd", "name {}".format(" ".join([j[1] for j in ampAts]))),
            ("cat_bb_rmsd", "backbone and (resid {})".format(" ".join(str(x) for x in liuCatRes))),
            ("cat_sc_rmsd", "(not backbone) and (resid {})".format(" ".join(str(x) for x in liuCatRes))),
            ("aaa_bb_rmsd", "backbone and (resid {})".format(" ".join(str(x) for x in aaaV1))),
            ("aaa_sc_rmsd", "(not backbone) and (resid {})".format(" ".join(str(x) for x in aaaV1))),
        ]
        ligSels = [ligSels[0], ligSels[3]]  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! BAD CODE
        self.wrv(md_fit_step, 'mdFit_step')
        for rmsName, ligSel in ligSels:
            print(f"Calculating rmsd of {rmsName} ...")
            alignment = align.AlignTraj(
                md_center_u,
                solv_u,
                filename=self.joinpath(f'{rmsName}.xtc'),
                select=ligSel
            )
            alignment.run(
                step=md_fit_step,
                start=start,
                stop=stop
            )
            print(f"Calculating rmsd of {rmsName} ...: OK: len = {len(alignment.rmsd)}")
            rmsd_array = np.array([
                (md_center_u.trajectory[rmsI * alignment.step].time, rmsEl)
                for rmsI, rmsEl in enumerate(alignment.rmsd)
            ])
            if append:
                old_rmsd_array = self.rdv(rmsName, if_not_exists=np.array([[0, 1]]))
                print(
                    f'Appending rmsd_array of len ({len(rmsd_array)}) to existing rmsd_array (len: {len(old_rmsd_array)}')
                rmsd_array = np.append(old_rmsd_array, rmsd_array, axis=0)

            self.wrv(
                rmsd_array,
                rmsName
            )

    def make_h_data(self, cnt_step=1, append=False, term_o_h=False):  # APEND is silly and simple function...
        aaa_bb_rmsd_u = self.get_aaa_bb_rmsd_universe()
        ml = rdpdb(self.unl_NEW_pdb.str, sanitize=False)
        lgd = prepLig(ln, extraLinkAt=0, term_o_h=term_o_h, ref=cur_ser.f__REF.str)
        ml = AllChem.AssignBondOrdersFromTemplate(lgd.lig, ml)
        match = ml.GetSubstructMatch(lgd.laq_core)
        unl_file_list = [i.split()[2] for i in self.unl_NEW_pdb.open().readlines() if i.startswith("ATOM")]
        kw_list1 = kw_list_gen(laq_hydr_cnts, match, unl_file_list)
        flur_smi = eval(f'{ln}_flur_smi')
        ba = Chem.MolFromSmiles(flur_smi)
        AllChem.EmbedMolecule(ba)
        match = ml.GetSubstructMatch(ba)
        hydr_cnts = eval(f'{ln}_hydr_cnts')
        kw_list2 = kw_list_gen(hydr_cnts, match, unl_file_list)  # ONLY NE* interaction!
        h_data = []
        t0 = time.time()
        kw_list1 += kw_list2
        if append:
            old_h_data = self.rdv('h_data', if_not_exists=[])
            if len(old_h_data) != len(kw_list1):
                append = False
                print(f'Update aborted. len(old_h_data): {len(old_h_data)}, len(kw_list1): {len(kw_list1)}')
        for kw_idx, kws in enumerate(kw_list1):
            sel_kws = [aaa_bb_rmsd_u.select_atoms(kws[f"selection{j}"]) for j in range(1, 3)]
            dst = mda.analysis.distances.dist(*sel_kws)[-1] if len(sel_kws[0]) == len(sel_kws[1]) else ''
            print(kw_idx, kws, [len(sel_kw) for sel_kw in sel_kws], dst)
            if kw_idx in [4, 8, 9]:  # BAD INTERACTIONS
                h_data.append(np.array([]))
                continue
            print('v declaration')
            v = MDAnalysis.analysis.hbonds.HydrogenBondAnalysis(aaa_bb_rmsd_u, **kws)
            print('v.run started')
            v.run(start=1, stop=len(aaa_bb_rmsd_u.trajectory), step=cnt_step)
            h_data_piece = np.array([list(j) for j in list(v.count_by_time())])
            if append:
                h_data_piece = np.append(old_h_data[kw_idx], h_data_piece, axis=0)
            h_data.append(h_data_piece)
            print(kws, np.mean(h_data_piece[:, 1]), "time: ", time.time() - t0)
        #         break
        self.wrv(h_data, f'h_data')

    def visu_traj(self, png=None):
        histStepRel = 0.05
        if 'get_ipython' in globals().keys():
            display(Image(ddir.joinpath('rpr/cnt1.png').str, width=300))
        cnt_data = self.rdv('h_data')
        assert len(cnt_data[0]) > 0, "cnt_data[0] is empty!"
        amp_cnt_data, flur_cnt_data = cnt_data[:8], cnt_data[8:]
        fig = plt.figure(figsize=(20, 20))
        gs = gridspec.GridSpec(4, 4)
        ax2 = fig.add_subplot(gs[0, 0:8 - len(flur_cnt_data)])
        rms_label = f'Dir: {self.name}'
        for rmsdName, rmsdColor, make_plot in [("rmsd", "r", 1), ('cat_bb_rmsd', '', 1), ('cat_sc_rmsd', '', 1),
                                               ('aaa_bb_rmsd', 'c', 1), ('aaa_sc_rmsd', '', 1)]:
            rmsdsI = self.rdv(rmsdName)
            if rmsdsI == "IOError":
                continue
            if make_plot:
                ax2.plot(rmsdsI[:, 0], rmsdsI[:, 1], rmsdColor, linewidth=1.0)
                ax2.hlines(np.median(rmsdsI[:, 1]), 0, rmsdsI[:, 0][-1], rmsdColor, linewidth=2.0)
            rms_label += "\n {}: {}| {}".format(rmsdName, round(np.median(rmsdsI[:, 1]), 2),
                                                round(np.std(rmsdsI[:, 1]), 2))
        ax2.text(0.5, 1.75, rms_label, **{'horizontalalignment': 'center', 'verticalalignment': 'top'})
        ax2.axes.axes.set_ylim(-0.05, 2.05)
        ax2.set(title="RMSD", xlabel="Time, ps", ylabel="RMSD")
        for k, cnt_dat in enumerate(amp_cnt_data + flur_cnt_data):
            if len(cnt_dat) == 0:
                continue
            if k < 4:
                k2 = k
                ax1 = fig.add_subplot(gs[1, k2:k2 + 1])
            elif k < 8:
                k2 = k - 4
                ax1 = fig.add_subplot(gs[2, k2:k2 + 1])
            else:
                k2 = k - 8
                ax1 = fig.add_subplot(gs[3, k2:k2 + 1])
            ax1.set(title="{})  mean={:.2f}, std={:.2f}".format(str(k + 1), np.mean(cnt_dat[:, 1]),
                                                                np.std(cnt_dat[:, 1])), xlabel="Time, ps",
                    ylabel="Average contacts")
            ax1.plot(cnt_dat[:, 0], cnt_dat[:, 1], 'r-', alpha=0.2)
            histStep = int(histStepRel * cnt_dat[:, 0][-1])
            histStep = 1 if histStep < 1 else histStep
            x = cnt_dat2bins(cnt_dat, histStep)

            m, e = map(lambda p: np.mean(p[1]), x), map(lambda p: np.std(p[1]), x)
            m, e = [*m], [*e]
            ax1.errorbar([j[0] for j in x], [*m], [*e], marker='^');
            ax1.hlines(np.mean(m), 0, cnt_dat[:, 0][-1], "b")

            m, e = map(lambda p: np.mean(p[1]), x), map(lambda p: np.std(p[1]), x)
            m, e = [*m], [*e]
            ax1.errorbar([j[0] for j in x], m, e, marker='.');
            ax1.hlines(np.mean(m), 0, cnt_dat[:, 0][-1], "y")

            ax1.axes.axes.set_ylim(-0.2, 1.3)
            if png:
                fig.savefig(self.joinpath('fig.png').str)
        return fig

    @staticmethod  # SEVERE SAMPLE BIAS !!!!
    def _get_cnt_average(cnt_slice):  # cnt = array([  [10, 1.0], [11, 0.0], .... ])
        cnt_average = sum(cnt_slice[:, 1]) / len(cnt_slice)
        return cnt_average

@for_all_methods(status)
class MM(M):
    def __new__(cls, path=''):
        return PosixPath.__new__(cls, path)

    def __init__(self, path, x=''):
        super().__init__(path)

    @property
    def chd1(self):
        return self['1']

    @property
    def mm_idx(self):
        return self.name_prefix.split('_')[-1]

    @property
    def par_pld(self):
        if self.stage == 2:
            rnd_sigmoid_Y = self.rdv('rnd_sigmoid_Y', if_not_exists=[])
            if not len(rnd_sigmoid_Y):
                raise('HR directories of stage #2 must have rnd_sigmoid_Y!')
            return self.par['plds'][rnd_sigmoid_Y[0]]

    @property
    def par_mol2_idx(self):
        return self.par_pld.mol2s[self.rdv('rnd_sigmoid_Y')[1]]

    @property
    def par_mol2(self):
        return self.par_pld['plds'][self.par_mol2_idx]

    # @property
    # def par_mol3(self):
    #     # par_mol2s =
    #     return [i for i in self.par.mol2s if i.name_prefix.split('_')[-1].lstrip('0') == self.mm_idx]

    @property
    def ed_dirs(self):
        return [P(i) for i in self.ls if i.name.startswith('ed')]

    @property
    def par_ed(self):
        if self.stage ==4:
            return self.par.par
        else:
            raise("not nested hrx")

    @property
    def chd(self):
        children_md_directories = [int(child.name) for child in self.iterdir() if
                                   child.name.isdigit()]
        children_md_directories.sort()
        return Ps([self.joinpath(str(child)) for child in children_md_directories])

    def initialize(self):
        super().initialize()

        # for number in range(1,17):
        #     child_path = self.joinpath(str(number))
        #     if child_path.exists():
        #         setattr(self, f'_{number}', P(child_path))

    def make_visu(self, *args, **kwargs):
        pmtr(mdirs=self.chd, *args, **kwargs)

    # PROCESS itself:

    def _get_lig_center_str(self):
        E_u = mda.Universe(self['EX.gro'].s, self['1/md_center.xtc'].s)

        lit_at_pairs = t.rdv('lit_at_pairs')
        lit_E_cat_at_resids = [str(i[0][0]) for i in lit_at_pairs]
        lit_E_cat_at_resids_str = " ".join(lit_E_cat_at_resids)
        lig_center = E_u.select_atoms(f'resid {lit_E_cat_at_resids_str}').center_of_mass()
        lig_center_str = '   '.join([*map(str, lig_center)])
        return lig_center_str

    def launch(self):
        self.load_structures()
        self.parametrize_ligand()
        if self.stage == 2:
            self.mutate_protein()
            # self['par_prot.pdb'].copy_to(self['prot.pdb'])
        else:
            self['par_prot.pdb'].copy_to(self['prot.pdb'])
        self.run_pdb2gmx()
        self.prep_lig_restraints()
        self.prep_complex_structure()
        self.append_ligand()
        self.create_solvent_box()
        self.solvate()
        # self.visu_solvate()
        # ?????????: MAYBE IMPORTANT! gmx make_ndx -f npt.gro

        self.grompp4genion()
        self.genion()
        # self.visu_genion()

        self.grompp_em()

        self.em()
        # self.visu_em_nrj()
        # self.visu_em()
        self.ndx_nvt()
        self.gpp_nvt()
        self.nvt()

        # self.visu_nvt_nrj()
        # self.visu_nvt()

        self.gpp_npt()
        self.npt()
        # self.visu_npt_nrj()
        # self.visu_npt()

        self.gpp_md()

        self.heat_atoms()

        self.write_distance_restraints()
        self.make_chd()
        self.prepare_plumed()
        # self.stop_hrx()
        self.run_hrx()

        self.monitor_hrx_traj()

        # if self.rdv(f'stage_{self.stage}_verdict'):
        #     self.prepare_kids()
        #     self.launch_kids()

    def load_structures(self):
        if self.stage == 4:
            self.unl_acpype.rm()
            self.par.par.unl_acpype.copy_to(self)
            # mdir['CRY_1_1/unl.acpype'].copy_to(self)
            unl_NEW_pdb_res = self.unl_acpype['unl_NEW_res.pdb']
            self.unl_NEW_pdb.copy_to(unl_NEW_pdb_res)
            unl_GMX_gro_res = self.unl_acpype['unl_GMX_res.gro']

            self.unl_GMX_gro.copy_to(unl_GMX_gro_res)
            dec_uni = mda.Universe(self.par.decoys[0].s)
            unl = dec_uni.select_atoms('resname UNL')
            prot = dec_uni.select_atoms('protein and not type H')
            prot.segments.segids = 'SYSA'
            prot.write(self['par_prot'].s) # mda writes random chain ID!
            # prot.write(args.E_pdb.s) # WHAT??
            unl_from_dec_pdb = self.unl_acpype['unl_from_dec.pdb']
            unl.write(unl_from_dec_pdb.s)

            changeMolIdxs(inp=unl_from_dec_pdb.s, otp=self.unl_NEW_pdb.s, ref=unl_NEW_pdb_res.s, db=0)
            u = mda.Universe(self.unl_NEW_pdb.s)
            u.atoms.write(self.unl_GMX_gro.s)
        else:
            self.par_mol2.copy_to(self.j('lig.mol2'))
            self.root.E_pdb.copy_to(self['par_prot.pdb'])

    def parametrize_ligand(self):
        # mdir['CRY_1_1/unl.acpype'].copy_to(self)
        # self.unl_acpy
        if self.stage == 2:
            # running acpype
            self.unl_acpype.rm()
            cmd_ = f'acpype -b unl -i {self["lig.mol2"]} --charge_method user --net_charge {args.net_charge} -a gaff'
            self.run(cmd_)
            # WE COULD REWRITE PDB file unl_NEW.pdb to be connected (representable in pymol), but it is unnecessary
            # lgd = Chem.MolFromSmiles(args.lig_smiles)
            # lgd = Chem.AddHs(lgd)
            # AllChem.EmbedMolecule(lgd)
            # wr = Chem.rdmolfiles.PDBWriter(self.root.sdir['lig.H.pdb'].s)
            # wr.write(lgd)

            # # Fixing atom numbering from O1' (pdb like) to O1 (rdkit like):
            # self.unl_GMX_gro.copy_to(self.unl_GMX_gro.append2prefix("res"))
            # # ERROR: changeMolIdxs -> no match. Because lig.mol2 has hydrogens in wrong place !!! Spores error.
            # unl_NEW_pdb_res = self.unl_acpype['unl_NEW_res.pdb']
            # self.unl_NEW_pdb.copy_to(unl_NEW_pdb_res)
            # unl_GMX_gro_res = self.unl_acpype['unl_GMX_res.gro']
            # self.unl_GMX_gro.copy_to(unl_GMX_gro_res)
            # t = self.root
            # pdb_rdkit_a_names = t.sdir['lig.H.pdb']
            # # t.sdir['lig.H.pdb'].l
            # changeMolIdxs(inp=unl_NEW_pdb_res.s, otp=self.unl_NEW_pdb.s, ref=pdb_rdkit_a_names.s, db=0)

    def get_complex_universe(self):
        prot_universe = mda.Universe(self.j('par_prot.pdb').s)

        if self.stage == 4:
            lig_universe = mda.Universe(self.unl_NEW_pdb.s)
        else:
            lig_universe = mda.Universe(self.j('lig.mol2').s)


        complex_universe = mda.Merge(prot_universe.atoms, lig_universe.atoms)
        return complex_universe

    def mutate_protein(self, db=0):
        # BAD CODE: 1. cat_atoms -> cat_aa, automatically assumed as catalitic (polar contacts with cat_atoms_names) -> set join with cat_aa from literature
        # 2. Ca-Cb vector to ligand !
        # 3. residues that form (any) polar contacts with cataltic residues are also banned
        t = self.root
        complex_universe = self.get_complex_universe()

        resname = mda.Universe(self['lig.mol2'].s).atoms[0].resname
        args.plants2hrx_thresh = 2.5
        cat_aa = ' '.join([*map(str, args.cat_aa)])
        mut_selection = f'byres (around {args.plants2hrx_thresh} resname {resname}) and not resname PRO ALA GLY and not resid {cat_aa}'
        residues_to_mutate = complex_universe.select_atoms(mut_selection).residues.resids
        self.wrv(residues_to_mutate, "residues_to_mutate")
        if db:
            print("Residues to mutate", residues_to_mutate, 'len: ', len(residues_to_mutate))
        print(residues_to_mutate)
        self['par_prot.pdb'].copy_to(self['prot.pdb'])

        for residue_to_mutate_idx in residues_to_mutate:
            print('mutating model: ', residue_to_mutate_idx)
            # mutate_protein(self['prot'].s, str(residue_to_mutate_idx), 'ALA', chain)
            Mutate_model(self['prot'].s, str(residue_to_mutate_idx), 'ALA', chain)

        # E_pdb_u = mda.Universe(t.E_pdb.s)
        # prot_u = mda.Universe(self['prot.pdb'].s)
        # prot_u_seq = [aaMap[res.resname] for res in prot_u.residues]
        # E_pdb_u_seq = [aaMap[res.resname] for res in E_pdb_u.residues]
        # assert len(E_pdb_u_seq) == len(prot_u_seq)
        # E_muts = [[i, from_, to_] for i, [from_, to_] in enumerate(zip(E_pdb_u_seq, prot_u_seq)) if from_ != to_]

    def run_pdb2gmx(self):
        from biobb_md.gromacs.pdb2gmx import Pdb2gmx

        # Create inputs/outputs

        prop = {
            # 'gmx_path': args.gmx_path,
            'force_field': 'amber99sb-ildn',
            'water_type': 'spce',
        }

        # Create and launch bb
        Pdb2gmx(input_pdb_path=self['prot.pdb'].s,
                output_gro_path=self['prot_pdb2gmx.gro'].s,
                output_top_zip_path=self['prot_pdb2gmx_top.zip'].s,
                properties=prop).launch()

    def prep_lig_restraints(self):
        gromacs.make_ndx(f='unl.acpype/unl_GMX.gro', o='index_lig.ndx',
                         input=('0 & ! a H*', 'q'), cwd=self)

        gromacs.genrestr(f='unl.acpype/unl_GMX.gro', n='index_lig.ndx', o='lig_posres.itp',
                         fc=1000, input='3', cwd=self)

    def prep_complex_structure(self):
        # biobb analysis module
        from biobb_analysis.gromacs.gmx_trjconv_str import GMXTrjConvStr
        from biobb_structure_utils.utils.cat_pdb import CatPDB


        # Convert gro (with hydrogens) to pdb (PROTEIN)
        # TODO use args gmx
        prop = {
            'gmx_path': args.gmx_path,
            'selection': 'System'
        }

        # Create and launch bb
        GMXTrjConvStr(input_structure_path=self['prot_pdb2gmx.gro'].s,
                      input_top_path=self['prot_pdb2gmx.gro'].s,
                      output_str_path=self['prot_complex_H.pdb'].s,
                      properties=prop).launch()

        # Convert gro (with hydrogens) to pdb (LIGAND)
        prop = {
            # TODO use args gmx
            'gmx_path': args.gmx_path,
            'selection': 'System'
        }
        # Create and launch bb
        GMXTrjConvStr(input_structure_path=self.unl_GMX_gro.s,
                      input_top_path=self.unl_GMX_gro.s,
                      output_str_path=self['lig_complex_H.pdb'].s,
                      properties=prop).launch()

        # Concatenating both PDB files: Protein + Ligand

        # Create and launch bb
        CatPDB(input_structure1=self['prot_complex_H.pdb'].s,
               input_structure2=self['lig_complex_H.pdb'].s,
               output_structure_path=self['prot_lig_H.pdb'].s).launch()

    def append_ligand(self):

        # AppendLigand: Append a ligand to a GROMACS topology
        # Import module
        from biobb_md.gromacs_extra.append_ligand import AppendLigand

        # Create prop dict and inputs/outputs
        posresifdef = 'POSRES_UNL'
        prop = {
            'gmx_path': args.gmx_path,
            'posres_name': posresifdef
        }

        # Create and launch bb
        AppendLigand(input_top_zip_path=self['prot_pdb2gmx_top.zip'].s,
                     input_posres_itp_path=self['lig_posres.itp'].s,
                     input_itp_path=self.unl_acpype['unl_GMX.itp'].s,
                     output_top_zip_path=self['complex.top.zip'].s,
                     properties=prop).launch()

    def create_solvent_box(self):
        # Editconf: Create solvent box
        # Import module
        from biobb_md.gromacs.editconf import Editconf

        # Create prop dict and inputs/outputs

        os.popen("echo $PATH").read()
        prop = {
            'gmx_path': args.gmx_path,
            'box_type': 'octahedron',
            'distance_to_molecule': 0.8
        }

        # Create and launch bb
        Editconf(input_gro_path=self['prot_lig_H.pdb'].s,
                 output_gro_path=self['complex_editconf.gro'].s,
                 properties=prop).launch()

    def solvate(self):
        # Solvate: Fill the box with water molecules
        from biobb_md.gromacs.solvate import Solvate

        # Create prop dict and inputs/outputs

        # Create and launch bb
        props = {
            'gmx_path': args.gmx_path,
            'input_solvent_gro_path': 'spc216.gro'
        }
        Solvate(input_solute_gro_path=self['complex_editconf.gro'].s,
                output_gro_path=self['solvate.gro'].s,
                input_top_zip_path=self['complex.top.zip'].s,
                output_top_zip_path=self['solvate_top.zip'].s,
                properties=props).launch()

    def visu_solvate(self):
        # Show protein
        # view = nv.show_file(self['solvate.gro'].s)
        # view.clear_representations()
        # view.add_representation(repr_type='cartoon', selection='protein', color='sstruc')
        # view.add_representation(repr_type='licorice', radius='.5', selection='UNL')
        # view.add_representation(repr_type='line', linewidth='1', selection='SOL', opacity='.7')
        # view._remote_call('setSize', target='Widget', args=['', '600px'])
        # view.camera = 'orthographic'
        # display(view)
        # return view
        print("visu solvate didn't produce, because emacs")
        # get_ipython().ex('display(view)')

    def grompp4genion(self):
        # Grompp: Creating portable binary run file for ion generation
        from biobb_md.gromacs.grompp import Grompp

        # Create prop dict and inputs/outputs
        prop = {
            'gmx_path': args.gmx_path,
            'mdp': {
                'type': 'minimization',
                'nsteps': '5000'
            }
        }

        # Create and launch bb
        Grompp(input_gro_path=self['solvate.gro'].s,
               input_top_zip_path=self['solvate_top.zip'].s,
               output_tpr_path=self['complex_genion.tpr'].s,
               properties=prop).launch()

    def genion(self):
        # Genion: Adding ions to reach a 0.05 molar concentration
        from biobb_md.gromacs.genion import Genion

        # Create prop dict and inputs/outputs
        prop = {
            'gmx_path': args.gmx_path,
            'neutral': True,
            'concentration': 0.05
        }

        # Create and launch bb
        Genion(input_tpr_path=self['complex_genion.tpr'].s,
               output_gro_path=self['genion.gro'].s,
               input_top_zip_path=self['solvate_top.zip'].s,
               output_top_zip_path=self['genion_top.zip'].s,
               properties=prop).launch()

    def visu_genion(self):
        # Show protein
        view = nv.show_file(self['genion.gro'].s)
        view.clear_representations()
        view.add_representation(repr_type='cartoon', selection='protein', color='sstruc')
        view.add_representation(repr_type='licorice', radius='.5', selection='UNL')
        view.add_representation(repr_type='ball+stick', selection='NA', radius=1)
        view.add_representation(repr_type='ball+stick', selection='CL', radius=1)
        view.add_representation(repr_type='line', linewidth='1', selection='SOL', opacity='.7')
        view._remote_call('setSize', target='Widget', args=['', '600px'])
        view.camera = 'orthographic'
        display(view)

    def grompp_em(self):
        # Grompp: Creating portable binary run file for mdrun
        from biobb_md.gromacs.grompp import Grompp

        # Create prop dict and inputs/outputs
        prop = {
            'gmx_path': args.gmx_path,
            'mdp': {
                'type': 'minimization',
                'nsteps': '5000',
                'emstep': 0.01,
                'emtol': '500'
            },
                "remove_tmp": False
        }

        # Create and launch bb
        Grompp(input_gro_path=self['genion.gro'].s,
               input_top_zip_path=self['genion_top.zip'].s,
               output_tpr_path=self['gppmin.tpr'].s,
               properties=prop).launch()

    def em(self):
        # Mdrun: Running minimization
        from biobb_md.gromacs.mdrun import Mdrun

        # Create prop dict and inputs/outputs

        prop = {
            'gmx_path': args.gmx_path,
            # "num_threads": args.num_threads_em_nvt_npt,
                  "gpu_id": t.rdv("gpu_id", if_not_exists=2),
                "remove_tmp": False
        }
        # Create and launch bb
        Mdrun(input_tpr_path=self['gppmin.tpr'].s,
              output_trr_path=self['min.trr'].s,
              output_gro_path=self['min.gro'].s,
              output_edr_path=self['min.edr'].s,
              output_log_path=self['min.log'].s,
              properties=prop).launch()

    def visu_em_nrj(self):
        # GMXEnergy: Getting system energy by time
        from biobb_analysis.gromacs.gmx_energy import GMXEnergy

        # Create prop dict and inputs/outputs

        prop = {
            'gmx_path': args.gmx_path,
            'terms': ["Potential"]
        }

        # Create and launch bb
        GMXEnergy(input_energy_path=self['min.edr'].s,
                  output_xvg_path=self['min_ene.xvg'].s,
                  properties=prop).launch()

    def visu_em(self):
        import plotly
        import plotly.graph_objs as go

        # Read data from file and filter energy values higher than 1000 Kj/mol^-1
        with open(self['min_ene.xvg'].s, 'r') as energy_file:
            x, y = map(
                list,
                zip(*[
                    (float(line.split()[0]), float(line.split()[1]))
                    for line in energy_file
                    if not line.startswith(("#", "@"))
                    if float(line.split()[1]) < 1000
                ])
            )

        plotly.offline.init_notebook_mode(connected=True)

        fig = ({
            "data": [go.Scatter(x=x, y=y)],
            "layout": go.Layout(title="Energy Minimization",
                                xaxis=dict(title="Energy Minimization Step"),
                                yaxis=dict(title="Potential Energy KJ/mol-1")
                                )
        })

        plotly.offline.iplot(fig)

    def ndx_nvt(self):
        # MakeNdx: Creating index file with a new group (protein-ligand complex)
        from biobb_md.gromacs.make_ndx import MakeNdx

        # Create prop dict and inputs/outputs
        prop = {
            'gmx_path': args.gmx_path,
            'selection': "\"Protein\"|\"Other\"",
            'container_shell_path': '/bin/bash'
        }

        # Create and launch bb
        MakeNdx(input_structure_path=self['min.gro'].s,
                output_ndx_path=self['index.ndx'].s,
                properties=prop).launch()

    def gpp_nvt(self):
        # Grompp: Creating portable binary run file for NVT System Equilibration
        from biobb_md.gromacs.grompp import Grompp
        Grompp("a", "a", "a")
        # Create prop dict and inputs/outputs
        posresifdef = 'POSRES_UNL'

        prop = {
            'gmx_path': args.gmx_path,
            'mdp': {
                'type': 'nvt',
                'nsteps': '5000',
                'tc-grps': 'Protein_Other Water_and_ions',
                'define': '-DPOSRES -D' + posresifdef,
                'lincs-warnangle': '180'
            },
                "remove_tmp": True
        }

        # Create and launch bb
        Grompp(input_gro_path=self['min.gro'].s,
               input_top_zip_path=self['genion_top.zip'].s,
               input_ndx_path=self['index.ndx'].s,
               output_tpr_path=self['gppnvt.tpr'].s,
               output_mdp_path=self['gppnvt.mdp'].s,
               properties=prop).launch()

    def nvt(self):
        # Mdrun: Running NVT System Equilibration
        from biobb_md.gromacs.mdrun import Mdrun

        # Create prop dict and inputs/outputs
        props = {
            'gmx_path': args.gmx_path,
            # "num_threads": args.num_threads_em_nvt_npt,
                  "gpu_id": t.rdv("gpu_id", if_not_exists=2)

                 }
        # Create and launch bb
        Mdrun(input_tpr_path=self['gppnvt.tpr'].s,
              output_trr_path=self['nvt.trr'].s,
              output_gro_path=self['nvt.gro'].s,
              output_edr_path=self['nvt.edr'].s,
              output_log_path=self['nvt.log'].s,
              output_cpt_path=self['nvt.cpt'].s,
              properties=props).launch()

    def visu_nvt_nrj(self):
        # GMXEnergy: Getting system temperature by time during NVT Equilibration
        from biobb_analysis.gromacs.gmx_energy import GMXEnergy

        # Create prop dict and inputs/outputs

        prop = {
            'gmx_path': args.gmx_path,
            'terms': ["Temperature"]
        }

        # Create and launch bb
        GMXEnergy(input_energy_path=self['nvt.edr'].s,
                  output_xvg_path=self['nvt_temp.xvg'].s,
                  properties=prop).launch()

    def visu_nvt(self):
        import plotly
        import plotly.graph_objs as go

        # Read temperature data from file
        with open(self['nvt_temp.xvg'].s, 'r') as temperature_file:
            x, y = map(
                list,
                zip(*[
                    (float(line.split()[0]), float(line.split()[1]))
                    for line in temperature_file
                    if not line.startswith(("#", "@"))
                ])
            )

        plotly.offline.init_notebook_mode(connected=True)

        fig = ({
            "data": [go.Scatter(x=x, y=y)],
            "layout": go.Layout(title="Temperature during NVT Equilibration",
                                xaxis=dict(title="Time (ps)"),
                                yaxis=dict(title="Temperature (K)")
                                )
        })

        plotly.offline.iplot(fig)

    def gpp_npt(self):
        # Grompp: Creating portable binary run file for (NPT) System Equilibration
        from biobb_md.gromacs.grompp import Grompp
        posresifdef = 'POSRES_UNL'
        # Create prop dict and inputs/outputs
        prop = {
            'gmx_path': args.gmx_path,
            'mdp': {
                'type': 'npt',
                'nsteps': '5000',
                'tc-grps': 'Protein_Other Water_and_ions',
                'define': '-DPOSRES -D' + posresifdef
            }
        }

        # Create and launch bb
        Grompp(input_gro_path=self['nvt.gro'].s,
               input_top_zip_path=self['genion_top.zip'].s,
               input_ndx_path=self['index.ndx'].s,
               output_tpr_path=self['gppnpt.tpr'].s,
               input_cpt_path=self['nvt.cpt'].s,
               properties=prop).launch()

    def npt(self):
        # Mdrun: Running NPT System Equilibration
        from biobb_md.gromacs.mdrun import Mdrun

        # Create prop dict and inputs/outputs
        prop = {
            'gmx_path': args.gmx_path,
                  # "num_threads": args.num_threads_em_nvt_npt,
                  "gpu_id": t.rdv("gpu_id", if_not_exists=2)
              }
        # Create and launch bb
        Mdrun(input_tpr_path=self['gppnpt.tpr'].s,
              output_trr_path=self['npt.trr'].s,
              output_gro_path=self['npt.gro'].s,
              output_edr_path=self['npt.edr'].s,
              output_log_path=self['npt.log'].s,
              output_cpt_path=self['npt.cpt'].s,
              properties=prop).launch()

    def visu_npt_nrj(self):
        # GMXEnergy: Getting system pressure and density by time during NPT Equilibration
        from biobb_analysis.gromacs.gmx_energy import GMXEnergy

        # Create prop dict and inputs/outputs

        prop = {
            'gmx_path': args.gmx_path,
            'terms': ["Pressure", "Density"]
        }

        # Create and launch bb
        GMXEnergy(input_energy_path=self['npt.edr'].s,
                  output_xvg_path=self['npt_PD.xvg'].s,
                  properties=prop
                  ).launch()

    def visu_npt(self):
        import plotly
        from plotly import tools
        import plotly.graph_objs as go

        # Read pressure and density data from file
        with open(self['npt_PD.xvg'].s, 'r') as pd_file:
            x, y, z = map(
                list,
                zip(*[
                    (float(line.split()[0]), float(line.split()[1]), float(line.split()[2]))
                    for line in pd_file
                    if not line.startswith(("#", "@"))
                ])
            )

        plotly.offline.init_notebook_mode(connected=True)

        trace1 = go.Scatter(
            x=x, y=y
        )
        trace2 = go.Scatter(
            x=x, y=z
        )

        fig = tools.make_subplots(rows=1, cols=2, print_grid=False)

        fig.append_trace(trace1, 1, 1)
        fig.append_trace(trace2, 1, 2)

        fig['layout']['xaxis1'].update(title='Time (ps)')
        fig['layout']['xaxis2'].update(title='Time (ps)')
        fig['layout']['yaxis1'].update(title='Pressure (bar)')
        fig['layout']['yaxis2'].update(title='Density (Kg*m^-3)')

        fig['layout'].update(title='Pressure and Density during NPT Equilibration')
        fig['layout'].update(showlegend=False)

        plotly.offline.iplot(fig)

    def gpp_md(self):
        gromacs.make_ndx(f='npt.gro', o='EX.ndx', input=('1 | 19', 'q'), cwd=self)
        gromacs.editconf(
            f='npt.gro',
            o='EX.gro',
            n='EX.ndx',
            input=('22'),
            cwd=self.s
        )

        gromacs.editconf(
            f='npt.gro',
            o='EX.pdb',
            n='EX.ndx',
            input=('22'),
            cwd=self.s
        )
        posresifdef = 'POSRES_UNL'
        nsteps_gpp_md = '1500000000000000'
        # nsteps_gpp_md = '1500'
        from biobb_md.gromacs.grompp import Grompp
        prop = {
            'gmx_path': args.gmx_path,
            'mdp': {
                'constraints': 'h-bonds',
                'type': 'free',
                # 'nsteps':'500000' # 1 ns (500,000 steps x 2fs per step)
                # 'nsteps':'5000' # 10 ps (5,000 steps x 2fs per step)
                'nsteps': nsteps_gpp_md,
                'nstxout': nsteps_gpp_md,
                'nstvout': nsteps_gpp_md,
                'nstfout': nsteps_gpp_md,
                'nstlog': '100',
                # 'nstxtcout': 500,
                'nstxout-compressed': 500,
                #  nstxout-compressed - the same thing in diff versions of gromacs
                'compressed-x-grps': 'Protein_UNL',
            },
            'remove_tmp': False,
        }

        gpp = Grompp(input_gro_path=self['npt.gro'].s, # just to get hrex_2molecules.top without includes!
                    input_top_zip_path=self['genion_top.zip'].s,
                    output_tpr_path=self['md.tpr'].s,
                    input_cpt_path=self['npt.cpt'].s,
                    input_ndx_path=self['EX.ndx'].s,
                    properties=prop)


        gpp.create_mdp(self['md.mdp'].s)
        self.run('rm -rf genion_top_zip/')
        self.run('unzip genion_top.zip -d genion_top_zip') # system toplology: ligand.top
        gromacs.grompp(f='md.mdp', c='npt.gro', p='genion_top_zip/ligand.top', n='EX.ndx',# we want .top file without indcludes!
                    pp='hrex_2molecules.top', t='npt.cpt', o='hrex_2molecules.tpr', cwd=self)

    def heat_atoms2(self, db=0):
        hrx_u = mda.Universe(self['npt.gro'].s)

        import parmed as pmd
        hrx_u = mda.Universe(self['npt.gro'].s)
        hrx_top = pmd.gromacs.GromacsTopologyFile(self['hrex_2molecules.top'].s)

        # Preparation to heating
        args.hrx_atom_selection_thresh = 5
        cat_aa = ' '.join([*map(str, args.cat_aa)])
        hrx_selection = f'resname UNL or (not backbone and byres (protein and (around {args.hrx_atom_selection_thresh} resname UNL))) '
        # Actual heating:
        hot_atoms = hrx_u.select_atoms(hrx_selection)
        db=True
        assert all(hrx_u.residues.resnames == np.array([i.name for i in hrx_top.residues], dtype=object))
        for hot_atom_res in hot_atoms.residues:
            if db:
                print(f'heating {hot_atom_res} residue')
            parmed_residue = hrx_top.residues[hot_atom_res.ix]
            residue_hot_atoms = hot_atoms.select_atoms(f'resid {hot_atom_res.resid}')
            for parmed_atom in parmed_residue.atoms:
                if parmed_atom.name in residue_hot_atoms.names:
                    parmed_atom.type = parmed_atom.type + "_"

        assert list(hrx_top.molecules)[0] == 'UNL'
        assert list(hrx_top.molecules)[1] == 'Protein_chain_A'
        hrx_top.write(self['hrex_.top'].s, [[0,1]])

    def heat_atoms3(self, db=0):

        # Getting hot atoms:

        cat_aa = ' '.join([*map(str, args.cat_aa)])
        hrx_selection = f'resname UNL or (not backbone and byres (around {args.sc_heating_atom_selection_thresh} resname UNL)) '
        prot_u = mda.Universe(self['prot_pdb2gmx.gro'].s)
        lig_u = mda.Universe(self['unl.acpype/unl_GMX.gro'].s)
        merg_u = mda.Merge(prot_u.atoms, lig_u.atoms)
        hot_atoms = merg_u.select_atoms(hrx_selection)
        # Actual heating:
        import parmed as pmd
        hrx_top = pmd.gromacs.GromacsTopologyFile(self['hrex.top'].s)

        assert hrx_top.residues[len(merg_u.residues) - 1].name == 'UNL'
        assert merg_u.residues[len(merg_u.residues) - 1].resname == 'UNL'

        for hot_atom_res in hot_atoms.residues:
            if db:
                print(f'heating {hot_atom_res} residue')
            parmed_residue = hrx_top.residues[hot_atom_res.ix]
            residue_hot_atoms = hot_atoms.select_atoms(f'resid {hot_atom_res.resid}')
            for parmed_atom in parmed_residue.atoms:
                if parmed_atom.name in residue_hot_atoms.names:
                    parmed_atom.type = parmed_atom.type + "_"

        assert list(hrx_top.molecules)[0] == 'UNL'
        assert list(hrx_top.molecules)[1] == 'Protein_chain_A'
        hrx_top.write(self['hrex_.top'].s)



    def heat_atoms(self, db=0):
        # TODO REMOVE HYDROGENS (maybe)
        import parmed as pmd
        hrx_u = mda.Universe(self['npt.gro'].s)
        hrx_top = pmd.gromacs.GromacsTopologyFile(self['hrex_2molecules.top'].s)

        # Preparation to heating
        args.hrx_atom_selection_thresh = 5
        cat_aa = ' '.join([*map(str, args.cat_aa)])
        hrx_selection = f'resname UNL or (not backbone and byres (protein and (around {args.hrx_atom_selection_thresh} resname UNL))) '
        # Actual heating:
        # byres not working
        hot_atoms = hrx_u.select_atoms(hrx_selection).residues.atoms
        db=True
        assert all(hrx_u.residues.resnames == np.array([i.name for i in hrx_top.residues], dtype=object))
        for hot_atom_res in hot_atoms.residues:
            if db:
                print(f'heating {hot_atom_res} residue')
            parmed_residue = hrx_top.residues[hot_atom_res.ix]
            residue_hot_atoms = hot_atoms.select_atoms(f'resid {hot_atom_res.resid}')
            for parmed_atom in parmed_residue.atoms:
                if parmed_atom.name in residue_hot_atoms.names:
                    parmed_atom.type = parmed_atom.type + "_"

        assert list(hrx_top.molecules)[0] == 'unl'
        assert list(hrx_top.molecules)[1] == 'Protein_chain_A'
        hrx_top.write(self['hrex_unrestr.top'].s, [[0,1]])

    def write_distance_restraints(self):
        dist_restr = [
            "",
            "[ distance_restraints ]",
            "; ai aj type index type? low up1 up2 fac"
        ]
        args.cat_dist_restr_selection_thresh = 5.5
        restr_idx = 0

        hrx_u = mda.Universe(self['npt.gro'].s)
        X_resname_hrx = 'UNL'
        lit_at_pairs = t.rdv('lit_at_pairs')
        E_and_X_id_list = []
        for cat_E_at_lst, cat_X_at_lst in lit_at_pairs:
            cat_X_at_id = cat_X_at_lst[2]
            cat_X_at_id_corred = cat_X_at_id + hrx_u.select_atoms(f'resname UNL').ids[0] - 1
            cat_E_at_name = cat_E_at_lst[1]
            cat_E_at_resid = cat_E_at_lst[0]
            E_at = hrx_u.select_atoms(f'resid {cat_E_at_resid} and name {cat_E_at_name}').atoms
            X_at = hrx_u.select_atoms(f'resname {X_resname_hrx} and bynum {cat_X_at_id_corred}').atoms
            hrx_u.select_atoms(f'resname UNL').ids
            assert len(E_at) == 1 and len(X_at) == 1
            E_at, X_at = E_at[0], X_at[0]
            self.wrv([E_at.id, X_at.id], 'dist_restrs_HREX')
            dist_restr.append(f'{E_at.id} {X_at.id} 1 {restr_idx} 2 0.28 0.3 0.35 1.0')
            E_and_X_id_list.append([E_at.id, X_at.id])
            restr_idx += 1
        restr_text = "\n".join(dist_restr) + "\n"
        E_atms = hrx_u.select_atoms(' or '.join([f'bynum {E_i}' for E_i, X_i in E_and_X_id_list]))
        X_atms = hrx_u.select_atoms(' or '.join([f'bynum {X_i}' for E_i, X_i in E_and_X_id_list]))

        E_atms.write(self['vi_E_atoms.pdb'].s)
        X_atms.write(self['vi_X_atoms.pdb'].s)
        pml_cmd = [
            f'load {self["npt.gro"].s}',
            f'remove r. sol',
            f"load {self['vi_E_atoms.pdb'].s}",
            f"load {self['vi_X_atoms.pdb'].s}",
            f'label vi_E_atoms, "%s-%s" % (resi, name)',
            f'label vi_X_atoms, "%s-%s" % (resi, name)',
            f'hide sticks',
            f'show lines, polymer',
            f'show sticks, vi_E_atoms',
            f'show spheres, vi_X_atoms',
            f'show sticks, org'
        ]
        pm(*pml_cmd)
        pm(f'save {self["vi_cnt.pse"].s}', r=False)
        print("DIST RESTR READY! :")
        print(restr_text)
        from textwrap import dedent
        hrex_top_fragment = dedent("""
            [ moleculetype ]
            ; Name            nrexcl
            SOL          3
        """)

        # lines = self['hrex_unrestr.top'].read_text()
        # before_fragment_coord = lines.index(hrex_top_fragment)
        # new_lines = lines[:before_fragment_coord] + restr_text + lines[before_fragment_coord:]
        self['hrex_unrestr.top'].copy_to(self['hrex_.top'])
        # self['hrex_.top'].write_text(new_lines)

    def make_chd(self):
        # TODO fitting (zlobin)
        # TODO stop criteria
        nrep = 4
        tmin, tmax = (300, 1000)

        self.run(f"""
        # build geometric progression
        list=$(
        awk -v n={nrep} \
            -v tmin={tmin} \
            -v tmax={tmax} \
        'BEGIN{{for(i=0;i<n;i++){{
            t=tmin*exp(i*log(tmax/tmin)/(n-1));
            printf(t); if(i<n-1)printf(",");
        }}
        }}'
        )
        echo $list
        # clean directory
        rm -fr \#*
        rm -fr topol*

        for((i=1;i<{nrep}+1;i++))
        do
        echo "PROCESSING DIRECTORY: $i"
        lambda=$(echo $list | awk 'BEGIN{{FS=",";}}{{print $1/$'$((i))';}}')
        echo "lambda = $lambda." # Instruct PLUMED to bias the Hamiltonian in hrex.top using the # current $lambda and writethe biased topology in topol_$i.top
        rm -r $i; mkdir $i; mkdir $i/.tmp
        /home/domain/anur/progs/plumed2.5/bin/plumed partial_tempering $lambda < hrex_.top > $i/topol.top
        {args.gmx_path} grompp -maxwarn 1 -f md.mdp -c npt.gro -p $i/topol.top -t npt.cpt -o $i/topol.tpr -n EX.ndx
        done
        """)

    def prepare_plumed(self, db=1):
        from textwrap import dedent
        # TODO make visu to make sure it plumed is working with atoms as expected (or some other way ensure, maybe assert)
        # uni = self['1'].get_md_center_universe()
        # # cpx = uni.select_atoms('protein or resname UNL')
        # E_for_plumed = self['E_for_plumed.pdb']
        # cpx = uni.select_atoms('protein')
        # cpx.write(E_for_plumed.s)
        # # cpx.write(E_for_plumed)
        # txt = E_for_plumed.read_text().replace("0.00", "1.00"); E_for_plumed.write_text(txt)


        uni = mda.Universe(self['EX.pdb'].s)
        E_ats_ids = uni.select_atoms('protein').ids
        X_ats_ids = uni.select_atoms('resname UNL').ids
        f1, f2 = min(E_ats_ids), max(E_ats_ids)
        f3, f4 = min(X_ats_ids), max(X_ats_ids)

        E_for_plumed = uni.select_atoms('protein')
        E_for_plumed_file = self['1/E_for_plumed_chd1.pdb']
        E_for_plumed.write(E_for_plumed_file)

        txt = E_for_plumed_file.read_text().replace("0.00", "1.00")
        E_for_plumed_file.write_text(txt)

        f5, f6 = self.rdv('dist_restrs_HREX')

        # self['1/plumed_vars.dat'].rm()


        self['1/plumed.dat'].write_text(dedent (f"""
            x: DISTANCE ATOMS={f5},{f6} NOPBC
            rmsd: RMSD REFERENCE={E_for_plumed_file} TYPE=OPTIMAL
            c: COORDINATION GROUPA={f1}-{f2} GROUPB={f3}-{f4} R_0=0.3 NLIST NL_CUTOFF=0.5 NL_STRIDE=100

            RESTRAINT ARG=x SLOPE=1.0 AT=3.0
            bias: REWEIGHT_BIAS

            PRINT FILE=plumed_vars.dat ARG=bias,x,rmsd,c STRIDE={args.cv_table_stride*500}
        """))

    def run_hrx(self):
        import signal
        # TODO module is different from ...
        self.wrv(None, 'min_array')

        run_hrx_env = os.environ.copy()
        run_hrx_env["PLUMED_MAXBACKUP"] = '-1'
        print("Starting gromacs...")
        gpu_id = t.rdv('gpu_id', if_not_exists=2)
        cmd_ = f"""
cd {self.s}; module load gmx/2019.6-threadedmpi-cuda-single-pm2.6-gpu; mpirun -np 4 /home/domain/data/prog/gromacs_2018.6_mpi_single_gpu_pm2.5.1/bin/gmx_mpi mdrun -v -plumed {self.chd1}/plumed.dat -multidir {{1..4}} -dlb no -hrex  -replex 100 -gpu_id {gpu_id}
        """
        proc = subprocess.Popen(
            cmd_,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=isinstance(cmd_, str),
            preexec_fn=os.setsid,
            cwd=self,
            env=run_hrx_env
        )
        self.wrv(proc.pid, 'proc_pid')

    def flush_analyzed_wrv(self):
        pass

    def stop_hrx(self):
        import signal
        proc_pid = self.rdv('proc_pid')
        if proc_pid != 'IOError':
            os.killpg(os.getpgid(proc_pid), signal.SIGTERM)

    def monitor_hrx_traj(self):
        self.wrv(0, 'chd_1_md_center_traj_len_already_reached')
        self.wrv(0, 'plumed_vars_len_already_reached')
        while True:
            try:
                self.make_and_analyze_cv_table()
                conv_len = self.rdv("conv_len", if_not_exists=0)
                B_len = self.rdv("B_len", if_not_exists=0)
                print(f"GOT:conv_len: {conv_len}, B_len: {B_len}")
                if conv_len > args.conv_len_thres:
                    self.stop_hrx()
                    self.wrv(1, f'stage_{self.stage}_verdict')
                    return
                if B_len > args.B_len_thres:
                    self.stop_hrx()
                    self.wrv(0, f'stage_{self.stage}_verdict')
                    return
            except Exception as e:
                print(f"monitor_hrx_traj: Exception! {str(e)}")
            sl(5)
        # sl(120)

    def make_cv_table(self, db=1, make_md_center=1, do_length_check=True):
        # TODO make visu to make sure it plumed is working with atoms as expected (or some other way ensure, maybe assert)
        if make_md_center:
            self['1'].make_md_center()
        uni = self['1'].get_md_center_universe() # must be equivalent
        # uni = mda.Universe(self['EX.pdb'].s)
        E_ats_ids = uni.select_atoms('protein').ids
        X_ats_ids = uni.select_atoms('resname UNL').ids
        f1, f2 = min(E_ats_ids), max(E_ats_ids)
        f3, f4 = min(X_ats_ids), max(X_ats_ids)
        f1,f2,f3,f4 = 1,5,10,15
        f5, f6 = self.rdv('dist_restrs_HREX')
                      # GRID_MIN_1, GRID_MAX_1
            # MOLINFO STRUCTURE=E_for_plumed.pdb
        import csv

        plumed_vars_text = self['1/plumed_vars.0.dat'].read_text()
        plumed_vars_text_split = plumed_vars_text.split('\n')[:-1]
        plumed_vars_text = "\n".join(plumed_vars_text_split)
        self['1/plumed_vars_stable.dat'].write_text(plumed_vars_text)
        lines_list = [[float(k) for k in i.split()] for i in plumed_vars_text_split if i and not i.startswith('#') and len(i.split()) == 5]
        arr = np.array(lines_list, dtype=float)

        # do a check of restrained atom check:
        f5, f6 = self.rdv('dist_restrs_HREX')

        # length check (files are written and in process they are short)
        if do_length_check:
            if self.rdv('chd_1_md_center_traj_len_already_reached', if_not_exists=0) > len(uni.trajectory):
                return("TRY_ONE_MORE_TIME")
            if self.rdv('plumed_vars_len_already_reached', if_not_exists=0) > arr.shape[0]:
                return("TRY_ONE_MORE_TIME")
        self.wrv(len(uni.trajectory), 'chd_1_md_center_traj_len_already_reached')
        self.wrv(arr.shape[0], 'plumed_vars_len_already_reached')

        # timestep check
        arr_ts = float(arr[1][0] - arr[0][0])
        uni_ts = float(uni.trajectory[1].time - uni.trajectory[0].time)

        print(f"Comparing timesteps of trajectory and plumed file: uni_ts: {uni_ts}, plumed_bias_ts = arr_ts: {arr_ts}")
        assert uni_ts - arr_ts < 0.02 # some bias is acceptable, we can correct it by rounding

        # plumed have accumulating bias in time! We need corrections, but some frames will be ommited.
        arr[:,0] = np.round(arr[:,0])
        # if there is a "deletion" in jjkjkj
        uni_atoms = uni.atoms
        tailored_xtc = self["1/uni_atoms_tailored.xtc"]
        arr_idx = 0
        arr_tailored = []
        uni.trajectory[0]
        with MDAnalysis.Writer(tailored_xtc.s, uni_atoms.n_atoms) as W:
            for tsi in range(len(uni.trajectory)):
                ts = uni.trajectory[tsi]
                if arr_idx >= arr.shape[0]:
                    break
                if ts.time == arr[arr_idx][0]:
                    W.write(uni_atoms)
                    arr_tailored.append(arr[arr_idx])
                    arr_idx = arr_idx + 1
                elif ts.time == arr[arr_idx+1][0]:
                    arr_tailored.append(arr[arr_idx])
                    W.write(uni_atoms)
                else:
                    break
        arr_tailored = np.vstack(arr_tailored)

        uni_tailored = mda.Universe(self['EX.gro'].s, tailored_xtc)

        print(f'Tailoring trajectory: {len(uni.trajectory)} ---> {len(uni_tailored.trajectory)}')
        print(f'Tailoring and de-deletion-ing plumed vars array: {arr.shape} ---> {arr_tailored.shape}')

        assert len(uni_tailored.trajectory) == arr_tailored.shape[0]
        self.wrv(arr_tailored, 'arr_tailored')
        GRID_MIN_1, GRID_MAX_1 = arr_tailored[:, 3].min(), arr_tailored[:, 3].max()
        GRID_MIN_2, GRID_MAX_2 = arr_tailored[:, 4].min(), arr_tailored[:, 4].max()

        self['fes1.dat'].rm()
        # TODO just 10 times and you have an array of time
        # for fes_idx in range(10):
        f5, f6 = self.rdv('dist_restrs_HREX')
        self['1/plumed_post_factum.dat'].write_text(textwrap.dedent(f"""
            rmsd: READ FILE={self.chd1}/plumed_vars_stable.dat  VALUES=rmsd IGNORE_FORCES 
            c:    READ FILE={self.chd1}/plumed_vars_stable.dat  VALUES=c IGNORE_FORCES 
            x:    READ FILE={self.chd1}/plumed_vars_stable.dat  VALUES=x IGNORE_FORCES 

            RESTRAINT ARG=x SLOPE=1.0 AT=3.0
            bias: REWEIGHT_BIAS TEMP=300

            HISTOGRAM ...
            ARG=rmsd,c CLEAR=0
            GRID_MIN={GRID_MIN_1},{GRID_MIN_2}
            GRID_MAX={GRID_MAX_1},{GRID_MAX_2}
            GRID_BIN=100,100
            BANDWIDTH=0.1,0.1
            LABEL=hB
            ... HISTOGRAM

            fes1: CONVERT_TO_FES TEMP=300 GRID=hB
            DUMPGRID GRID=fes1 FILE={self.chd1}/fes1.dat STRIDE=0 FMT=%8.4f
        """))

            # c: COORDINATION GROUPA=2477-2513 GROUPB=1-2476 R_0=0.3 NLIST NL_CUTOFF=0.5 NL_STRIDE=100
        run_hrx_env = os.environ.copy()
        run_hrx_env["PLUMED_MAXBACKUP"] = '-1'
        print("Starting gromacs...")
        self.run(f'/home/domain/rustam/miniconda3/envs/py37/bin/plumed driver --plumed=1/plumed_post_factum.dat --noatoms', env=run_hrx_env)

    def visu_cv_table(self):
#         min_coords_values = self.rdv('min_coords_values')
#         # Visualization of min and max boundaries of CV1 (3, 4) and CV2 (5, 6)

#         my_plot(slopes_centered, "slopes_centered", "Номер результата докинга", "Скор программы PLANTS", "Результаты PLANTS, cluster_structures=10, ~2 000 запусков")

#         self.plt_plot(min_coords_values[:, 4], "CV1_мин" , "Время, ps", "Мин. и Макс. значения CV1")

#         plt_plot
# # def my_plot(arrs, suptitle, ):
# # plt.plot(np.arange(min_coords_values.shape[0]), min_coords_values[:,3], label="CV1_мин")
#         my_plot(np.arange(min_coords_values.shape[0]), min_coords_values[:, 4], label="CV1_макс")
        plt.xlabel("Время, ps")
        plt.ylabel("Мин. и Макс. значения CV1")
        fig = plt.gcf()
        fig.savefig(d_u['img/CV1_boundaries_traj_0_0.png'].s, dpi=300)
        plt.legend()
        plt.show()


        plt.plot(np.arange(min_coords_values.shape[0]), min_coords_values[:,3], label="CV1_мин")
        my_plot(np.arange(min_coords_values.shape[0]), min_coords_values[:, 4], label="CV1_макс")
        plt.xlabel("Время, ps")
        plt.ylabel("Мин. и Макс. значения CV1")
        fig = plt.gcf()
        fig.savefig(d_u['img/CV1_boundaries_traj_0_0.png'].s, dpi=300)
        plt.legend()
        plt.show()

        plt.plot(np.arange(min_coords_values.shape[0]), min_coords_values[:,5], label="CV2_мин")
        plt.plot(np.arange(min_coords_values.shape[0]), min_coords_values[:, 6], label="CV2_макс")
        plt.xlabel("Время, ps")
        plt.ylabel("Мин. и Макс. значения CV2")
        fig = plt.gcf()
        fig.savefig(d_u['img/CV2_boundaries_traj_0_0.png'].s, dpi=300)
        plt.legend()
        plt.show()


        plt.plot(np.arange(min_coords_values.shape[0]), min_coords_values[:, 2], label="Координата№2 лид. бина")
        plt.xlabel("Время, ps")
        plt.ylabel("Координаты лидирующего бина в решетке 100х100.")
        fig = plt.gcf()
        fig.savefig(d_u['img/leader_bin_coords_traj_0_0.png'].s, dpi=300)
        plt.legend()
        plt.show()

    def make_and_analyze_cv_table(self, if_make_cv_table=True):
        # TODO change do_login_check=True in production!!!
        if if_make_cv_table and self.make_cv_table(do_length_check=False) ==  "TRY_ONE_MORE_TIME":
            return("TRY_ONE_MORE_TIME")
        fes1_dat = self['1/fes1.dat']
        """#! FIELDS rmsd c hB dhB_rmsd dhB_c"""
        fes1_dat_arr = parse_hist_like_data_from_plumed(fes1_dat)
        if not fes1_dat_arr.shape[0] == 101 or not all([k.shape[0] == 101 for k in fes1_dat_arr]):
            print("Plumed is rewriting file right now....")
            return("TRY_ONE_MORE_TIME")

        arr_for_bins = fes1_dat_arr[:, :, 0:2]
        bins_cv_1, bins_cv_2 = arr_for_bins[0, :, 0], arr_for_bins[:, 0, 1]
        bins_cv_1_corred = np.array([bins_cv_1[i: i + 2].mean() for i in range(bins_cv_1[:-1].shape[0])])
        bins_cv_2_corred = np.array([bins_cv_2[i: i + 2].mean() for i in range(bins_cv_2[:-1].shape[0])])

        arr_99_99_3 = fes1_dat_arr[1:-1, 1:-1, 0:3]
        X_99 = arr_99_99_3[0, :, 0]
        Y_99 = arr_99_99_3[:, 0, 1]
        Z_99 = arr_99_99_3[:, :, 2]
        # TODO not full code, it is a tuple of minima
        Z_min_idx_99 = np.unravel_index(Z_99.argmin(), Z_99.shape)

        # Z_mins = self.rdv('Z_mins', if_not_exists=[])
        # Z_mins = Z_mins + Z_min_idx_99

        assert Z_99.min() == Z_99[Z_min_idx_99[0], Z_min_idx_99[1]]
        assert bins_cv_1_corred[0] < X_99[0] < bins_cv_1_corred[1]
        assert bins_cv_2_corred[0] < Y_99[0] < bins_cv_2_corred[1]
        assert bins_cv_1_corred[-2] < X_99[-1] < bins_cv_1_corred[-1]
        assert bins_cv_2_corred[-2] < Y_99[-1] < bins_cv_2_corred[-1]
        min_bin_coords = [
            [
                bins_cv_1_corred[Z_min_idx_99[0] - 1],
                bins_cv_1_corred[Z_min_idx_99[0] + 2],
            ],
            [
                bins_cv_2_corred[Z_min_idx_99[1] - 1],
                bins_cv_2_corred[Z_min_idx_99[1] + 2],
            ]
        ]

        t_bias_x_rmsd_c = self.rdv('arr_tailored')

        x, y = t_bias_x_rmsd_c[:, 3], t_bias_x_rmsd_c[:, 4]
        matching_frames = []
        # kara = np.array(kar)
        # np.count_nonzero(kara[:, 0])
        # np.count_nonzero(kara[:, 1])
        kar = []
        # TODO: erroneous code !
        for t, bias, x, rmsd, c in t_bias_x_rmsd_c:
            # kar.append([(min_bin_coords[0][0] < rmsd < min_bin_coords[0][1]), (min_bin_coords[1][0] < c < min_bin_coords[1][1])])
            if (min_bin_coords[0][0] < rmsd < min_bin_coords[0][1]) and (min_bin_coords[1][0] < c < min_bin_coords[1][1]):
                matching_frames.append([t, rmsd, c])
        self.wrv('matching_frames', matching_frames)

        # len(matching_frames)
        # recorded_min_coords = self.rdv('recorded_min_coords', if_not_exists=[])

        # for i, min_coords in enumerate(recorded_min_coords):
        #     xi, yi = x[:i], y[:i]
        #     bins = np.histogram2d(x, y, bins=(bins_cv_1_corred, bins_cv_2_corred))[0]
        #     # bins = np.histogram2d(xi, yi, 100)[0]
        #     min_x, min_y = np.unravel_index(bins.argmax(), bins.shape)
        #     min_coords = np.array([[bins[min_x, min_y], min_x, min_y, min(xi), max(xi), min(yi), max(yi)]])
        #     min_coords_values = np.append(min_coords_values, min_coords, axis=0)

        # leader_min_coords_last = min_coords_values[:, [1,2]][-1]
        # leader_min_coords_bool_2 = min_coords_values[:, [1, 2]] == leader_min_coords_last
        # B = np.all(leader_min_coords_bool_2, axis=1)
        # i_inv = np.argmax(B[::-1] != True)
        # i_conv = len(B) - i_inv
        # B_len = len(B)
        # conv_len = len(B[i_conv:])
        # conv_ratio = conv_len/B_len
        # self.wrv(min_coords_values, 'min_coords_values')
        # self.wrv(conv_len, 'conv_len')
        # self.wrv(B_len, 'B_len')
        # self.wrv(conv_ratio, 'conv_ratio')

    def get_complex_dG(self):
        # TODO get complex inferred energy
        return 5.0

    def run_affbio (self, b=0, e=10**20, skip=args.cv_table_stride, db=0):
        if db:
            e = 200
        self['1/snapshots'].rwdir()
        # gromacs.make_ndx(f='../npt.gro', o='EX.ndx', input=('1 | 19', 'q'), cwd=self['1'])
        gromacs.trjconv(
            s='topol.tpr', f='md_center.xtc',
            n='../EX.ndx',
            o='snapshots/frame.pdb',
            conect=True,
            skip=skip,
            sep=True,
            b=b,
            e=e,
            # input=('22'),
            input=('22'),
            cwd=self['1'].s
        )
        # gmx trjconv -n EX.ndx -f md_center.xtc -s topol.tpr -o snapshots/frame.pdb -conect -skip 5 -sep -b 0 -e 100
        srv.run("affbio -m aff_matrix.hdf5 -t cluster  -f *.pdb --verbose --nopbc", cwd=self['1/snapshots'].s)


    # UNUSED FUNCTIONS:
    def make_visu_hrx_1(self):
        #BAD code from A1.ipynb
        protein = u.select_atoms("all")
        # protein.write("protein1.gro")
        fn = self["md_snapshot.pdb"].s
        protein.write(fn)
        print(fn)

        for chd in self.chd:
        #     chd['traj.trr'].move_to(chd['md.trr'])
        #     chd['topol.tpr'].move_to(chd['md.tpr'])
        #     chd['traj_comp.xtc'].move_to(chd['md.xtc'])
            self['npt.gro'].copy_to(chd['solv_ions.gro'])

        # self['1'].make_md_center()
        self['1'].make_rmsd()
        self['1'].make_visu()
        self['1'].load_visu()

        cat_aa = ' '.join([*map(str, args.cat_aa)])
        cat_aa

        complex_universe = self.get_complex_universe()

        prot_universe = mda.Universe(self.j('par_prot.pdb').s)

        prot_universe2 = mda.Universe(self.j('prot.pdb').s)

        print([*prot_universe2.residues])

        print([*prot_universe.residues])

        [*self.rglob('*.tpr')]

        mdt = self['hrex.tpr']

        u = mda.Universe(mdt)

        hrx_u = mda.Universe(self['npt.gro'].s)

    def make_min_cluster_centroids(self, db=1):
        print("started")
        colv = self['RMSD_c'].read_text()
        rmsd_c = np.array([[float(i.split()[0]), float(i.split()[1]), float(i.split()[2])] for i in colv.split('\n') if not i.startswith('#') and i])
        X = rmsd_c[:, 1:]
        import sklearn
        from sklearn.mixture import GaussianMixture
        # Fit the data
        gmm = GaussianMixture(n_components=2)
        gmm.fit(X)
        # Distribution parameters; print(gmm.means_); print(gmm.covariances_)

        step = 100
        x = np.linspace(X[:, 0].min(), X[:, 0].max(), step)
        y = np.linspace(X[:, 1].min(), X[:, 1].max(), step)
        xx, yy = np.meshgrid(x, y)
        pos = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis = 1)
        z = gmm.score_samples(pos) # Note that this method returns log-likehood
        # z = np.exp(gmm.score_samples(pos)) # e^x to get likehood values
        z = z.reshape(xx.shape)

        fig = plt.gcf()
        fig.clear()
        plt.contourf(x, y, z, 50, cmap="viridis")
        # plt.scatter(X.T[0], X.T[1])
        plt.draw()
        fig.subplots_adjust()
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel('CV2: Координация, лиганд-фермент')
        ax1.set_title('CV1: RMSD относительно стартовой структуры')
        fig.savefig(self['plt.png'].s, dpi=300)
        plt.show()

        # Making ed dirs:
        hrex_scores = gmm.score_samples(X)
        # many dirs, SIMPLIFICATION :
        peak_id = hrex_scores.argmax()
        peak_id_frame = int(rmsd_c[peak_id][0])
        # u = self["1"].get_trr_universe()
        self.wrv(peak_id_frame, 'peak_id')


    # def prepare_kids(self):
    #     peak_id = self.rdv('peak_id')
    #     u = mda.Universe(self["1/topol.tpr"].s, self["1/traj.trr"].s)
    #     if type(self.kid) == Pl:
    #         for i, peak_id in enumerate([peak_id]):
    #             print(f'peak # {i}: {peak_id}. writing...')
    #             pl = self['plants'].mkdir()
    #             pld = pl[f'pld_{i}']
    #     if type(self.kid) == E:
    #         for i, peak_id in enumerate([peak_id]):
    #             print(f'peak # {i}: {peak_id}. writing...')
    #             edd = P(self[f'ED_{i}__3'])
    #             u.trajectory[peak_id]
    #             protein = u.select_atoms('protein')
    #             for res in protein.residues:
    #                 res.resid += 1
    #             #     protein.residues[16].resname
    #             # TODO ATOM NUMBERING MAY BE WRONG!!!
    #             protein.write(edd.ss.joinpath('prot_from_traj.pdb'))
    #             u.select_atoms('resname UNL') \
    #                 .write(edd.cg.joinpath('lig_from_traj.pdb'))

    def prepare_kids(self): # KIDS: PL or E
        n_dirs, dir_nam = args.stage2kids[self.stage]
        for dir_idx in range(n_dirs):
            kid = self[f'{dir_nam}_{dir_idx}__{self.stage + 1}']
            kid.initialize()
        u = mda.Universe(self['EX.gro'].s, self['1/md_center.xtc'].s)
        # TODO bad code
        import random as rnd
        # matching_frames = self.rdv('matching_frames', if_not_exists=[])
        # random_matching_frame = rnd.choice(matching_frames)
        # u.trajectory(random_matching_frame[0])
        u.trajectory[-1]


        # recorded_min_coords = self.rdv('recorded_min_coords', if_not_exists=[])

        protein = u.select_atoms('protein') # TODO ATOM NUMBERING MAY BE WRONG!!!
        protein.segments.segids = 'SYSA'
        ligand = u.select_atoms('resname UNL')
        # if type(self.kid) == Pl:
        #     # PROTEIN
        #     protein.write(self['prot_after_4.pdb'].s)
        #     self['prot_after_4.pdb'].l
        #     self['spores'].rwdir()
        #     # self.sdir.joinpath("lig.pdb").copy_to(self.spores_dir)
        #     self['prot_after_4.pdb'].copy_to(self['spores/E.pdb'])
        #     self['spores'].run(f'{cur_ser.e__spores_exe} --mode complete E.pdb sporesed_prot.mol2')
        #     self['spores/sporesed_prot.mol2'].copy_to(self.kid['str/prot.mol2'])


        #     # LIGAND
        #     ligand.write(self.unl_acpype['ligand_after_4.gro'].s)
        #     u1 = mda.Universe(self.unl_acpype['lig.mol2'].s)
        #     u2 = mda.Universe(self.unl_acpype['ligand_after_4.gro'].s)
        #     u1.atoms.positions = u2.atoms.positions
        #     u1.atoms.write(self.unl_acpype['ligand_after_4_charged.mol2'].s)
        #     self.unl_acpype['ligand_after_4_charged.mol2'].copy_to(self.kid['str/lig.mol2'])
        if type(self.kid) == E:
            for edd in self.kids:
                protein.write(edd.ss['prot_from_traj.pdb'])
                ligand.write(edd.cg['lig_from_traj.pdb'])

    # @status
    # def run_eds(self):
    #     procs = {}
    #     for edd in self.ed_dirs:
    #         edd.prepare_ed_run()
    #         p = edd.run_ed()
    #         edd.run_hrxs()

            # print(f'Launched ED in {frame_dir} with pid: {p.pid}')
            # procs[edd] = p

        # for edd, p in procs.items():
        #     print(f'Waiting for proc: {p.pid} of {edd}')
        #     p.wait()

    def run_plds(self):
        pass
class Ps(list):
    def __init__(self, mdir_list):
        if isinstance(mdir_list[0], str):
            mdir_list = [P(i) for i in mdir_list]
        if all([lambda x: x.__class__ == M for x in mdir_list]):
            self.__class__ = Ms
        # mdir_list = [*set(mdir_list)]
        super().__init__(mdir_list)
        # MDDirSetter().md_dirs_setter(self, [str(i) for i in self])

    @property
    def pd_sl_paths(self):
        return [i.str for i in self]

    def __add__(self, other):
        some_list = super().__add__(other)
        new_list = Ps(some_list)
        # MDDirSetter().md_dirs_setter(new_list, [str(i) for i in new_list])
        return new_list

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pd_sl_paths})'

    def unlink(self):
        for file in self:
            print(f'unlinking {file}')

    def rm(self):
        for file in self:
            file.rm()

    def rwdir(self):
        for path in self:
            path.rwdir()

    def rwfile(self):
        for path in self:
            path.rwfile()

    def mkdir(self):
        for item in self:
            if item.is_dir:
                item.mkdir()

class Ms(Ps):
    def __init__(self):
        super().__init__()

    def make_good_frames(self):
        for p in self:
            p.make_good_frames()
def remove_term_o_h(mpc_uncl_str, db=False, flur_smi=bff_flur_smi):
    mpc_uncl_str2 = Chem.Mol(mpc_uncl_str)
    flur = Chem.MolFromSmiles(flur_smi)
    term_o = [mpc_uncl_str2.GetSubstructMatch(flur)[j] for j in [6]]
    term_o_h = [a.GetIdx() for a in mpc_uncl_str2.GetAtomWithIdx(term_o[-1]).GetNeighbors() if a.GetSymbol() == "H"]
    emol = rdkit.Chem.rdchem.EditableMol(mpc_uncl_str2)
    emol.RemoveAtom(term_o_h[0])
    deprot_mol = emol.GetMol()
    if db: loadCnf2Pml("deprot_mol", deprot_mol, srv_md=False)
    return deprot_mol
def rewrite_nitro_hydro(mpc_uncl_str):
    mpc_uncl_str2 = Chem.Mol(mpc_uncl_str)
    mpc_uncl_str2.RemoveAllConformers()
    lgd = prepLig("rrf")
    n_c_n = [mpc_uncl_str.GetSubstructMatch(lgd.laq_core)[j] for j in [13, 18, 19]]
    nitro_hydros = [a.GetIdx() for a in mpc_uncl_str.GetAtomWithIdx(n_c_n[-1]).GetNeighbors() if a.GetSymbol() == "H"]
    for cid in range(mpc_uncl_str.GetNumConformers()):
        cnf = mpc_uncl_str.GetConformer(cid)
        cnf.SetAtomPosition(nitro_hydros[0], [i - 0.5 for i in list(cnf.GetAtomPosition(nitro_hydros[0]))])
        cid2 = mpc_uncl_str2.AddConformer(cnf, assignId=True)
        cnf3 = mpc_uncl_str2.GetConformer(cid2)
        print("<")
        # print([list(cnf.GetAtomPosition(nitro_hydros[i])) for i in range(2)])
        print(">")
    return mpc_uncl_str2
def cntRouting(trjPath):
    trjPath = mdDir(trjPath, pd=True, crop=True, trSl=True)
    if trjPath in cntRoutingDict.keys():
        return cntRoutingDict[trjPath]
    else:
        return cntRoutingDict[""]
def get_tmou(tdir):
    dec_str = PDBParser().get_structure("ref", "{}dec.pdb".format(tdir))
    if dec_str.child_list[0].child_list[0].child_list[0].resname in ["RRF", "BFF", "UNL"]:
        str_l = dec_str.child_list[0].child_list[1].child_list
    else:
        str_l = dec_str.child_list[0].child_list[0].child_list
    mts = map(lambda x: aaMap2[x.resname] if x.resname not in ["RRF", "BFF", "UNL"] else "", str_l)
    rfs = list(lplaSeq)
    return {ridx + 1: mit for ridx, (rit, mit) in enumerate(zip(rfs, mts)) if rit != mit}
def cnt_dat2bins(cnt_dat, histStep, db=0):
    rng = range(0, int(cnt_dat[:, 0][-1]), int(histStep))
    rng_hist = np.digitize(cnt_dat[:, 0], [*rng])
    zp_hist = *zip(rng_hist, cnt_dat),
    x = []
    for ki, kp in enumerate(rng):
        binItems = *filter(lambda y: y[0] == ki + 1, zp_hist),
        xs, ys = [y[1][0] for y in binItems], [y[1][1] for y in binItems]
        xs = np.mean(xs)
        x.append((xs, ys))
        # print(ki, binItems)
    return x
def get_ref_dist(r, r0, dist=5.0):
    return min(r0)
def cnt_fn_gen(ref_dist):
    def cnt_fn(r, r0, dist=5.0):
        return MDAnalysis.analysis.contacts.radius_cut_q(r, r0, ref_dist)

    return cnt_fn
def is_any_closer(r, r0, dist=5.0):
    # print(r, r0)
    print(MDAnalysis.analysis.contacts.radius_cut_q(r, r0, [min(2.9, ri * 1.2) for ri in r0]))
    return MDAnalysis.analysis.contacts.radius_cut_q(r, r0, [min(2.9, ri * 1.2) for ri in r0])
def pmal(x, name='', parital=0):
    prl = ', partial=1' if str(x).endswith('pse') and parital else ''
    name = f', {name}' if name else ''
    pma(f'load {x}{prl}{name}')
def pmrl(x, name='', parital=0):
    prl = ', partial=1' if str(x).endswith('pse') and parital else ''
    name = f', {name}' if name else ''
    pmr(f'load {x}{prl}{name}')
def mutatePdb(inp="ss/bff_relax/lpla_lg2_0001.pdb", muts={}, otp="out_str.pdb", db=False):
    pm("load {}".format(inp))
    for pos, aa in muts.items():
        if db: note(aa, aaRevMap[aa])
        pm("zoom i. {}".format(pos), "show sticks, i. {}".format(pos), r=False)
        if db: prepareImage()
        cmd.wizard("mutagenesis")
        cmd.do("refresh_wizard")
        cmd.do('cmd.get_wizard().do_select("i. {}")'.format(pos))
        cmd.do('cmd.get_wizard().set_mode("{}")'.format(aaRevMap[aa]))
        cmd.do("refresh_wizard")
        cmd.do("cmd.get_wizard().apply()")
        cmd.do("cmd.set_wizard()")
        pm("orient i. {}".format(pos), r=False)
        if db: prepareImage()
    # Sometimes the temp mutant is not removed
    try:
        cmd.remove("/_tmp_mut")
    except:
        pass

    pm("save {}".format(otp), r=False)
def prepareImage(width=300, height=300, sleep=2, filename='/tmp/pymolimg.png'):
    ## To save the rendered image
    cmd.ray(width, height)
    cmd.png(filename)
    # display(filename)
    display(Image('/tmp/pymolimg.png'))
    sl(sleep)

# def decVisuNew():


def decVisu(inp="", pmName=".tmp/d2.pse", srv_md=True, r=False, label=True, pmal=False, repr="spheres"):
    vd(r);
    mkDir('.tmp')
    if inp:
        pm("load {}".format(inp), srv_md=srv_md, r=r)
    else:
        pm("", srv_md=srv_md, r=r, db=True)
    pm("select aaaV1_crys, {}".format(pymol_expr(aaaV2)), "select cat, {}".format(pymol_expr(liuCatRes)),
       "select catAmp, {}".format(pymol_expr(catAmpRes)),
       "util.cbao aaaV1", "util.cbaw cat", "util.cbam catAmp", "show {}, cat | aaaV1 | org".format(repr),
       "set label_size, 37", "zoom org", "set label_position, (0,0,3)",
       r=False, srv_md=srv_md)
    pm("select flu, n. F1", "color flurine, flu", "alter flu, vdw=1.47", "rebuild", "delete flu", r=False,
       srv_md=srv_md)
    if label:
        pm('label (aaaV1_crys | cat | catAmp) & n. CA, "%s-%s" % (resi, resn)',
           r=False, srv_md=srv_md)
    if pmName and srv_md: pm("save {}".format(pmName), r=False, srv_md=srv_md)
    if pmal:
        pma("load {}".format(pmName))
def waitFun(func):
    def decorated_function(*args, **kwargs):
        ok = False
        while not ok:
            try:
                print("trying...")
                result = func(*args, **kwargs)
                ok = True
                say("Ok")
                return result
            except Exception:
                sl(1)
                clear_output()
                pass

    return decorated_function
def scancel(job_id=None, all=False):
    if all:
        idxs = sqdf().index
        [run(f'scancel {i}') for i in idxs]
    if job_id:
        lmsp("scancel {}".format(job_id))
        note("{} cancelled".format(job_id))
def md_analyse(trjPath=None):
    # if not trjPath: return
    hrex = trjPath.split("/")[1].startswith("HRX")
    if hrex:
        tr_id = int(trjPath.split("/")[2])
    hydDirs = ['../BFF_1_5/', '../2B8_10_46/', '../BFF_1_11/', '../BFF_1_4/', '../BFF_1_10/', '../BFF_1_8/',
               '../4TV_1_2/', '../LIU_1_1/', '../4TV_1_4/', '../BFF_1_2/', '../BFF_1_9/', '../BFF_1_3/', '../BFF_1_6/',
               '../2B8_10_5/', '../BFF_1_7/', '../BFF_1_1/']
    test_run, cnf_run, db2, db3, review = 0, 1, 1, 1, 0
    md_fit_step, cnt_step = [1000, 1000] if test_run else [1, 1] if cnf_run else [5, 5]
    make_md_center, make_rmsd, make_h_data = 0, 0, 1
    make_mdFit, make_visu, load_visu = 0, 0, 0
    if cnf_run: warn("ATTENTION!: md_fit_step: 5 -> 1\nATTENTION!: cnt_step: 5 -> 1")
    # FUNCTION
    print("md_analyse launched with {}. hrex = {}".format(trjPath, hrex))
    rev = {}
    for j in ["rmsd", "cat_bb_rmsd", "cat_sc_rmsd", "aaa_bb_rmsd", "aaa_sc_rmsd", "h_data"]:
        rev[j] = rdv(j, trjPath) != "IOError"
    for j in ["md_center.xtc", "mdFit.xtc", "rmsd.xtc", "md2.xtc"]:
        rev[j] = os.path.exists("{}{}".format(trjPath, j))
    note("Review: {}".format(rev))
    # if review: return
    #     make_rmsd = 1 if rdv("rmsd", trjPath) == "IOError" else make_rmsd
    make_md_center = 1 if not os.path.exists("{}md_center.xtc".format(trjPath)) else make_md_center
    #     make_cnt_data = 1 if rdv("cnt_data", trjPath) == "IOError" else make_cnt_data
    # gmx trjconv -s npt.gro  -f npt.trr -o npt_center.trr -center -pbc mol -ur compact
    if make_md_center:
        sp(
            """ printf "4\n0\n" | {args.gmx_path} trjconv -s {0}md.tpr -f {0}md.xtc -o {0}md_center.xtc -center -pbc mol -ur compact 2>/dev/null """.format(
                trjPath), silent=True)
    if make_rmsd or make_mdFit:
        trju = mda.Universe("{}md.tpr".format(trjPath), "{}md_center.xtc".format(trjPath))
        print("{} has traj length of {}".format(trjPath, len(trju.trajectory)))
        refu = mda.Universe("{}md.tpr".format(trjPath), "{}solv_ions.gro".format(trjPath))
    if make_rmsd:
        ampAts = get_names_from_pdb_str(refu, "../../str/amp2.pdb")
        ligSels = [
            ("rmsd", "name {}".format(" ".join([j[1] for j in ampAts]))),
            ("cat_bb_rmsd", "backbone and (resid {})".format(" ".join(str(x) for x in liuCatRes))),
            ("cat_sc_rmsd", "(not backbone) and (resid {})".format(" ".join(str(x) for x in liuCatRes))),
            ("aaa_bb_rmsd", "backbone and (resid {})".format(" ".join(str(x) for x in aaaV1))),
            ("aaa_sc_rmsd", "(not backbone) and (resid {})".format(" ".join(str(x) for x in aaaV1))),
        ]
        if cnf_run:
            ligSels = [ligSels[0], ligSels[3]]
        # if not l2: wrv(md_fit_step, "mdFit_step", trjPath)
        for rmsName, ligSel in ligSels:
            print("Calculating rmsd of {} ...".format(rmsName))
            alignment = align.AlignTraj(trju, refu, filename='{}{}.xtc'.format(trjPath, rmsName), select=ligSel)
            alignment.run(step=md_fit_step);
            print("{} alignment length: {}".format(trjPath, len(alignment.rmsd)))
            wrv(np.array([(rmsI * alignment.step, rmsEl) for rmsI, rmsEl in enumerate(alignment.rmsd)]), rmsName,
                trjPath)
    if make_h_data:
        print("Making h DATA");
        vd(trjPath)
        cont_trju = mda.Universe("{}md.tpr".format(trjPath), "{}aaa_bb_rmsd.xtc".format(trjPath))
        ml = rdpdb("{}unl.acpype/unl_NEW.pdb".format(trjPath), sanitize=False)
        ln_type = "bff" if "B" in [ml.GetAtomWithIdx(j).GetSymbol() for j in range(ml.GetNumAtoms())] else "rrf"
        extraLinkAts = 0 if trjPath not in ["../BFF_1_8/"] else 1
        term_o_h = trjPath in hydDirs
        lgd = prepLig(ln_type, extraLinkAt=extraLinkAts, term_o_h=term_o_h, ref=REF)
        ml = AllChem.AssignBondOrdersFromTemplate(lgd.lig, ml)
        match = ml.GetSubstructMatch(lgd.laq_core)
        unl_file_list = [i.split()[2] for i in open("{}unl.acpype/unl_NEW.pdb".format(trjPath)).readlines() if
                         i.startswith("ATOM")]
        kw_list1 = kw_list_gen(laq_hydr_cnts, match, unl_file_list)
        ba = Chem.MolFromSmiles(eval("{}_flur_smi".format(ln_type)))
        AllChem.EmbedMolecule(ba)
        match = ml.GetSubstructMatch(ba)
        vd(match)
        kw_list2 = kw_list_gen(eval("{}_hydr_cnts".format(ln_type)), match, unl_file_list)
        h_data = [];
        t0 = time.time()
        if trjPath not in ["../BFF_1_8/"]:
            kw_list1 += kw_list2
        #         if cnf_run:
        #             kw_list1.del(4); kw_list1.del(-2); kw_list1.del(-2)
        for kw_idx, kws in enumerate(kw_list1):
            # ATTTTTTTTTTTTTTTEEEEEEEEEEEEEEEEEEEEEENNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTIIIIIIIIIIIiion!!!!!!!!!!!!!!!!!!!!!!!!!! bad code!:
            if kw_idx != 7:
                continue
            sel_kws = [cont_trju.select_atoms(kws["selection{}".format(j)]) for j in range(1, 3)]
            dst = MDAnalysis.analysis.distances.dist(*sel_kws)[-1] if len(sel_kws[0]) == len(sel_kws[1]) else ''
            print(kw_idx, kws, [len(sel_kw) for sel_kw in sel_kws], dst)
            if kw_idx in [4, 8, 9] and cnf_run: print("skipped due to cnf_run"); h_data.append(np.array([])); continue
            #             if test_run: continue
            v = MDAnalysis.analysis.hbonds.HydrogenBondAnalysis(cont_trju, **kws)
            v.run(start=1, stop=len(cont_trju.trajectory), step=cnt_step)
            h_data_piece = np.array([list(j) for j in list(v.count_by_time())])
            h_data.append(h_data_piece)
            print(kws, np.mean(h_data_piece[:, 1]), "time: ", time.time() - t0)
        #         break
        wrv(h_data, "h_data", trjPath, db=1)
    if make_mdFit:
        refIds = [138, 140, 147, 149, 17, 20];
        ligSel = "backbone and (resid {})".format(" ".join(str(x) for x in refIds))
        alignment = align.AlignTraj(trju, refu, filename='{}mdFit.xtc'.format(trjPath), select=ligSel)
        alignment.run(step=md_fit_step);
        print("{} alignment length: {}".format(trjPath, len(alignment.rmsd)))
    if make_visu:
        trju2 = mda.Universe("{}md.tpr".format(trjPath), "{}mdFit.xtc".format(trjPath));
        print("len(trju2.trajectory) = {}".format(len(trju2.trajectory)))

        ligSel = "resname UNL or (resid {})".format(" ".join(str(x) for x in aaaV2 + liuCatRes))
        ligSel = "not (resname SOL)"

        trjSolv = mda.Universe("{}solv_ions.gro".format(trjPath))
        protein = trjSolv.select_atoms(ligSel)
        W = MDAnalysis.Writer("{}so.gro".format(trjPath), protein.n_atoms)
        W.write(protein)
        protein = trju2.select_atoms(ligSel)
        with MDAnalysis.Writer("{}md2.xtc".format(trjPath), protein.n_atoms) as W:
            for ts in trju2.trajectory:
                W.write(protein)
        # trju2.trajectory[0]; W.write(protein)
    if load_visu:
        pm("load ../LIU_1_1/.tmp/d2.pse")
def ms2df(txt):
    open("/tmp/ms2df.txt", "w").write(txt)
    return read_table("/tmp/ms2df.txt", sep=r"\s+")
def prntFun(func):
    def decorated_function(*args, **kwargs):
        pat = "{0}{1}{2}{1}{3}".format("<-", "-" * 25, "{}", "->")
        print(pat.format("{} started.".format(func.__name__)))
        t = time.time()
        result = func(*args, **kwargs)
        print(pat.format("{} finished. Time: {}s".format(func.__name__, round(time.time() - t, 2))))
        return result

    return decorated_function
def ffp(fg, depth=4, db=False):  # e.g. *.pyc; lib.* etc.
    matches = []
    for root, dirnames, filenames in os.walk('../../'):
        if db: print(filenames)
        for filename in fnmatch.filter(filenames, fg):
            matches.append(os.path.join(root, filename))
    if len(matches) == 1:
        return matches[0]
    else:
        return matches
def se_exp(x):
    return (" r. " + x[0] + " and  n. " + x[1] + " and  i. " + x[2] + " ")
def se_count(sel_name, srv_md=True):
    pm_eng = cmd if srv_md else pml
    stored.count = 0
    pm_eng.iterate(sel_name, "stored.count += 1")
    return stored.count
def se2list(p, srv_md=True):
    stored.list = [];
    pm_eng = cmd if srv_md else pml
    pm_eng.iterate(p, "stored.list.append((resn, name, resi))")
    return (stored.list)
def print_pym_selection(selName="sele", selVal="name"):
    pml_pri_sel_expr = """stored.list=[]
cmd.iterate("({})","stored.list.append(({}))")
print stored.list""".format(selName, selVal)
    pml.do(pml_pri_sel_expr)
def pairwise_dist(sel1, sel2, max_dist, output="N", sidechain="N", show="N"):
    """
    usage: pairwise_dist sel1, sel2, max_dist, [output=S/P/N, [sidechain=N/Y, [show=Y/N]]]
    sel1 and sel2 can be any to pre-existing or newly defined selections
    max_dist: maximum distance in Angstrom between atoms in the two selections
    --optional settings:
    output: accepts Screen/Print/None (default N)
    sidechain: limits (Y) results to sidechain atoms (default N)
    show: shows (Y) individual distances in pymol menu (default=N)
    """
    cmd.delete("dist*")
    extra = ""
    if sidechain == "Y": extra = " and not name c+o+n"

    # builds models
    m1 = cmd.get_model(sel2 + " around " + str(max_dist) + " and " + sel1 + extra)
    m1o = cmd.get_object_list(sel1)
    m2 = cmd.get_model(sel1 + " around " + str(max_dist) + " and " + sel2 + extra)
    m2o = cmd.get_object_list(sel2)

    # defines selections
    cmd.select("__tsel1a", sel1 + " around " + str(max_dist) + " and " + sel2 + extra)
    cmd.select("__tsel1", "__tsel1a and " + sel2 + extra)
    cmd.select("__tsel2a", sel2 + " around " + str(max_dist) + " and " + sel1 + extra)
    cmd.select("__tsel2", "__tsel2a and " + sel1 + extra)
    cmd.select("IntAtoms_" + max_dist, "__tsel1 or __tsel2")
    cmd.select("IntRes_" + max_dist, "byres IntAtoms_" + max_dist)

    # controlers-1
    if len(m1o) == 0:
        print
        "warning, '" + sel1 + extra + "' does not contain any atoms."
        return
    if len(m2o) == 0:
        print
        "warning, '" + sel2 + extra + "' does not contain any atoms."
        return

    # measures distances
    s2 = []
    s = ""
    counter = 0
    for c1 in range(len(m1.atom)):
        for c2 in range(len(m2.atom)):
            distance = math.sqrt(sum(map(lambda f: (f[0] - f[1]) ** 2, zip(m1.atom[c1].coord, m2.atom[c2].coord))))
            if distance < float(max_dist):
                if m1.atom[c1].name[0] != 'H' and m2.atom[c2].name[0] != 'H':
                    s2.append([[tuple([m1.atom[c1].resn, m1.atom[c1].name, m1.atom[c1].resi]),
                                tuple([m2.atom[c2].resn, m2.atom[c2].name, m2.atom[c2].resi])], distance])
                s += "%s/%s/%s/%s/%s to %s/%s/%s/%s/%s: %.3f\n" % (
                    m1o[0], m1.atom[c1].chain, m1.atom[c1].resn, m1.atom[c1].resi, m1.atom[c1].name, m2o[0],
                    m2.atom[c2].chain, m2.atom[c2].resn, m2.atom[c2].resi, m2.atom[c2].name, distance)
                counter += 1
                if show == "Y": cmd.distance(
                    m1o[0] + " and " + m1.atom[c1].chain + "/" + m1.atom[c1].resi + "/" + m1.atom[c1].name,
                    m2o[0] + " and " + m2.atom[c2].chain + "/" + m2.atom[c2].resi + "/" + m2.atom[c2].name)

    # controler-2
    if counter == 0:
        print
        "warning, no distances were measured! Check your selections/max_dist value"
        return
    print
    "Number of distances calculated: %s" % (counter)
    cmd.hide("lines", "IntRes_*")
    if show == "Y": cmd.show("lines", "IntRes_" + max_dist)
    cmd.deselect()
    return s2  # [[('RRF', '1', 'O6'), ('ASN', '121', 'ND2')], 2.9359434415975265]
def get_dists(cont_trju, tupleCnts):
    """VERY SLOW DISTANCE MEASUREMENT
    ptr = cntRouting(trjPath)
    tupleCnts = [map(lambda x: tuple([x]) if isinstance(x,str) else x, pt) for pt in ptr]
    dists = []
    for tsi, ts in enumerate(cont_trju.trajectory):
        ts_dists = get_dists(cont_trju,tupleCnts)
        if db3: print(tsi, ts_dists)
        dists.append(ts_dists)
    """
    dists = []
    for cntIdx, (tidx, t0, t1) in enumerate(tupleCnts):
        tupleDists = []
        db4 = False
        for t00 in t0:
            for t11 in t1:
                if db4: print(tidx, t11, t00, t1, t0)
                s1_, s2_ = ["resid {} and name {}".format(tidx, t00), "resname UNL and name {}".format(t11)]
                s1, s2 = map(cont_trju.select_atoms, [s1_, s2_])
                if db4: print(s1, s2, s1_, s2_)
                dst = MDAnalysis.analysis.distances.dist(s1, s2)
                tupleDists.append(dst[2])
        r = min(tupleDists)
        r0 = [r[1] for r in natCntDists][cntIdx]
        if min(tupleDists) < min(2.9, 1.2 * r0):
            dists.append(1)
        else:
            dists.append(0)
    return dists
def getScore(fileName, silent=True, prmsPath="cg/BFF.params", liuw=True):
    if liuw:
        prmsPath = "cg/prms/prms_0/RRF.params"
    rmFile("score.sc")
    sp(
        "/home/domain/data/prog/rosetta_2017/main/source/bin/score_jd2.default.linuxgccrelease -in:file:s {} -extra_res_fa {}".format(
            fileName, prmsPath), silent=silent)
    try:
        sc1 = float(re.findall(r'(?<=SCORE:)\s*\S*', open("score.sc").read())[-1])
    except Exception:
        sc1 = 0.0
    note(fileName, sc1)
    return sc1
def rlxRun(liuw=False, rlxRun=False, score=False, scoreNat=False, silent=True, db=False, prefix="rl"):
    if db: pmr("")
    if liuw:
        for chainIdx, chain in enumerate(["A", 'B', 'C', 'D'])[1:]:
            if rlxRun:
                rwdir('ss/liuw_rlx_{}/'.format(chainIdx))
                tsp(
                    "/home/domain/data/prog/rosetta_2017/main/source/bin/relax.default.linuxgccrelease @ri/msc/rlx_{}.flags".format(
                        chainIdx),
                    cwd="../", name="rlx_{}_{}".format(ed_id, chainIdx))
            if score:
                if scoreNat:
                    # sp("/home/domain/data/prog/rosetta_2017/main/source/bin/score_jd2.default.linuxgccrelease -in:file:s  -extra_res_fa cg/prms/prms_0/RRF.params".format(chainIdx), silent=silent) # SCORING INIT_STR W/O LIG
                    # sp("/home/domain/data/prog/rosetta_2017/main/source/bin/score_jd2.default.linuxgccrelease -in:file:s ss/liuw_{0}.pdb -extra_res_fa cg/prms/prms_0/RRF.params".format(chainIdx), silent=silent) # SCORING INIT_STR W/O LIG
                    # sp("/home/domain/data/prog/rosetta_2017/main/source/bin/score_jd2.default.linuxgccrelease -in:file:s .res/ss_2_res_11_APR/liuw_rlx_0/lpla_lg_0_0001.pdb -extra_res_fa cg/prms/prms_0/RRF.params".format(chainIdx), silent=silent) # SCORING INIT_STR W/O LIG
                    sc1 = getScore("ss/ss_batch/lpla_lg2_0.pdb")
                else:
                    sc1 = 0
                sc2 = getScore("ss/liuw_rlx_{0}/lpla_lg_{0}_0001.pdb".format(chainIdx))
                note("Scores for {}".format(chain), vd(sc1), vd(sc2))
            if db: pma("load ss/liuw_{0}.pdb".format(chainIdx),
                       "load ss/ss_batch/lpla_lg_{0}.pdb, ss_batch_{0}".format(chainIdx),
                       "load ss/liuw_rlx_{0}/lpla_lg_{0}_0001.pdb, ss_liuw_rlx_{0}".format(chainIdx),
                       "align ss_liuw_rlx_{0}, liuw_{0}".format(chainIdx),
                       "align ss_batch_{0}, liuw_{0}".format(chainIdx))
    else:
        tsp("/home/domain/data/prog/rosetta_2017/main/source/bin/relax.default.linuxgccrelease @ri/msc/rlx.flags",
            cwd="../", name="{}_{}".format(prefix, ed_id))
def get_names_from_pdb_str(refu, refStr="../../str/amp2.pdb"):
    ligtr = rdpdb(refStr, sanitize=False)
    ligtr2 = Chem.RemoveHs(ligtr)
    mkDir(".tmp/litu/")
    litu_name = ".tmp/litu/litu_{}.pdb".format(random.randint(1, 10 ** 7))
    refu.select_atoms("resname UNL").write(litu_name)
    litu3 = PDBParser().get_structure("", litu_name)
    litu = rdpdb(litu_name, sanitize=False)
    selAts = []
    rdMatch = litu.GetSubstructMatch(ligtr2)
    for atIdx, atName in enumerate(litu3[0].child_list[0].child_list[0].child_list):
        if atIdx in rdMatch:
            selAts.append((atIdx, atName.name))
    print(selAts)
    return (selAts)
def nor(unicode_string):
    return unicodedata.normalize('NFKD', unicode_string).encode('ascii', 'ignore')
def plms(path_):
    a = path_.split("/")
    new_path_ = "/home/domain/rustam/l/" + "/".join(a[6:])
    # print(new_path_)
    return new_path_
def psl(path_):
    path_ = os.path.abspath(path_)
    for path_i in ["/home/domain/data/rustam/", "/home/domain/rustam/", "/mnt/storage/rustam/", "/mnt/scratch/users/fbbstudent/work/rustam/"]:
        path_ = path_.replace(path_i, "/r/")
    new_path_ = path_
    # new_path_ = "/r/" + "/".join(a[5:])
    return new_path_
def pls(path_):
    a = path_.split("/")
    new_path_ = "/r/" + "/".join(a[2:])
    print(new_path_)
    return new_path_
def ringFinished(job_id, name, clo=True):
    while True:
        sl(1)
        queue = lmsp("squeue -u fbbstudent", silent=True)
        if clo: clear_output()
        queueStr = re.search(r'(?<={}).*'.format(job_id), queue)
        if not queueStr:
            for _ in range(100):
                say("JOB FINISHED")
            break
def ringStarted(job_id, name="some job", clo=True):
    while True:
        sl(1)
        queue = lmsp("squeue -u fbbstudent", silent=True)
        if clo: clear_output()
        queueStr = re.search(r'(?<={}).*'.format(job_id), queue).group()
        if queueStr.strip().split()[-1].strip("()")[0] == "n":
            print("Got nodes!:\n{}".format(queueStr))
            say("{} started".format(name))
            break
def say(phrase):
    os.system(""" ssh rustam@localhost -p 22222 " espeak {}" """.format(phrase))
def notif(phrase):
    open("/home/domain/rustam/.tmp/notif.txt", "w").write(phrase)
    sl(0.3)
def reprMol(list_):
    cmd_ = ""

    def str_(array_):
        return str(",   ".join([str(i) for i in list(array_)]))

    for lit in list_:
        atSym, lit = lit
        cmd_ += """spherelist = [
   COLOR,    {},
   SPHERE,   {},{},
    ]
cmd.load_cgo(spherelist, "{}",   1)
""".format(str_(pmlCpk[atSym]), str_(lit), 0.5, list_.index([atSym, lit]))
    cmd_ = """
from pymol.cgo import *
from pymol import cmd

{}
""".format(cmd_)
    open(".tmp/pmSph.py", "w").write(cmd_)
    pm("run .tmp/pmSph.py", "set pse_export_version, 1.721", "save .tmp/pmSph.pse", srv_md=True)
    pm(".tmp/pmSph.pse", srv_md=False)
def pymol_repr(array_, name):
    def str_(array_):
        return str(",   ".join([str(i) for i in list(array_)]))

    com_ = """python
from pymol.cgo import *
from pymol import cmd

spherelist = [
   COLOR,    {},
   SPHERE,   {},{},
    ]
cmd.load_cgo(spherelist, "{}",   1)
python end""".format(str_(nprandom.rand(1, 3)[0]), str_(array_), 0.5, name)
    for line in com_.split("\n"):
        print(line)
        pml.do(line)
def run_alt(cmd_, *args, **kwargs):
    return run(cmd_, *args, **kwargs)
def sp(command, silent=False, cwd="", ipk=True, sch=False, com=None):
    if cwd:
        cwd = os.path.abspath(cwd)
    my_env = os.environ.copy()
    print("Executing command:\n\t{}".format(command))
    if ipk:
        if "/home/domain/rustam/" in os.getcwd():
            command = ". /home/domain/rustam/miniconda3/bin/activate ipykernel_py2; " + command
        else:
            command = ". /home/rustam/anaconda3/envs/py27/bin/activate py27;" + command
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.abspath(cwd),
                         env=my_env, executable='/bin/bash')

    ser = ""
    so, ser = p.communicate()
    if not silent:

        if isinstance(so, bytes):
            so = so.decode('utf8')
        if isinstance(ser, bytes):
            ser = ser.decode('utf8')
        if ser: ser = "\nSTDERR:\n{}".format(ser)
        # print("STDOUT:\n{}{}".format(so, ser))
        print('STDOUT:')
        print(so)
        return so
def pymol_expr(list_, sel="NA", sc=False):
    if sel == "NA":
        try:
            int(list_[0])
            sel = "i"
        except Exception:
            sel = "n"
    c = ""
    for i in list_:
        c += "or {}. {} ".format(sel, str(i))
    if sc:
        return "(( {} ) &! n. n+c+o) ".format(c[3:])
    else:
        return "( {} )".format(c[3:])
def lmsp(command_, ipk=True, silent=False):
    import paramiko
    host = 'lomonosov2.parallel.ru'
    user = 'fbbstudent'
    secret = " "
    port = 22
    print("Command: \n" + command_)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, password=secret, port=port)
    stdin, stdout, stderr = client.exec_command(command_)
    err = stderr.read()
    data = stdout.read()
    data += err
    client.close()
    data = data.decode()
    if not silent:
        print(data)
    return (data)
def ssp(command_, ipk=True, silent=False):
    import paramiko
    ipk_activate = ""
    if ipk:
        ipk_activate = 'source /home/domain/rustam/miniconda3/bin/activate ipykernel_py2; '
    host = 'vsb.fbb.msu.ru'
    user = 'rustam'
    secret = " "
    port = 22022
    command_ = ipk_activate + command_
    print("Command: \n" + command_)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, password=secret, port=port)
    stdin, stdout, stderr = client.exec_command(command_)
    err = stderr.read()
    data = stdout.read()
    data += err
    client.close()
    if not silent:
        print(data)
    return (data)
def glob_or_file_exists(glob_or_path, db=True):
    if os.path.exists(glob_or_path):
        return True
    else:
        gl = glob.glob(glob_or_path)
        if gl:
            if db: note("found glob: {}".format(gl))
            return True
        else:
            return False
def file_or_glob_readable(glob_or_path):
    gl = glob.glob(glob_or_path)
    if gl:
        glob_or_path = gl[0]
    return open(glob_or_path).read()
def wait4file(filenames, cwd=os.path.curdir, silent=True, boolFilenamesFunc=all, ring=True):
    if isinstance(filenames, str):
        filenames = [filenames]
    boolFilenames = [os.path.exists(os.path.realpath(os.path.join(cwd, filename))) for filename in filenames]
    vd(boolFilenames)
    while not boolFilenamesFunc(boolFilenames):
        # boolFilenames = [os.path./exists(os.path.realpath(os.path.join(cwd, filename))) for filename in filenames]
        boolFilenames = [glob_or_file_exists(os.path.realpath(os.path.join(cwd, filename))) for filename in filenames]
        mapFilenames = zip(filenames, boolFilenames)
        if not silent:
            vd(mapFilenames)
            sl(1)
            clear_output()
    while not map(lambda x: file_or_glob_readable(x), filenames):
        continue
    sl(1)  # big files are written slowly
    if ring:
        say("Files appeared")
def ltsp(command_, ipk=False, silent=False, name="MD", log=True, wait=False):
    logPath = ".log/tsp.txt"
    ipk_activate = ""
    if log:
        if not os.path.exists(".log"):
            os.mkdir(".log")
        vd(logPath)
        if os.path.exists(logPath):
            os.remove(logPath)
        command_ += " 2>&1 > .log/tsp.txt"
    if ipk:
        ipk_activate = 'source /home/domain/rustam/miniconda3/bin/activate ipykernel_py2; '
    bash_command = """SESSION='{}'
tmux kill-session -t $SESSION
tmux -2 new-session -d -s $SESSION
tmux new-window -t $SESSION':1' -n 'Logs1'
tmux send-keys '{}' C-m""".format(name, ipk_activate + command_)
    sp(bash_command, ipk=False, silent=silent)
    if wait:
        wait4file(logPath)
    if log:
        print(open(logPath).read())
def tssp(command_, ipk=True, silent=False, name="rename_me", cwd=''):
    ipk_activate = ""
    if ipk:
        ipk_activate = 'source /home/domain/rustam/miniconda3/bin/activate ipykernel_py2; '
    bash_command = """SESSION='{}'
tmux kill-session -t $SESSION
tmux -2 new-session -d -s $SESSION
tmux new-window -t $SESSION':1' -n 'Logs1'
tmux send-keys '{}' C-m""".format(name, ipk_activate + command_)
    ssp(bash_command, ipk=False, silent=silent)
def pat(ar, pat=(), occ=0, i_s=0, r_s=0):
    global df
    if pat[-1] == 1:
        strict = True
    elif pat[-1] == 0:
        strict = False
    else:
        print("Enter correct pattern: 0 = non strict, 1 = strict" + "!" * 20)
        raise ()
    pat = pat[:-1]
    df = copy.deepcopy(ar)
    for k, v in {"occ": u' Occurance', "i_s": "{}-amp ifE".format(ln.lower()), "r_s": "TotScore"}.items():
        if eval(k):
            if k == "occ":
                df = df.loc[to_numeric(df[v], errors='coerce') > eval(k)]
            else:
                df = df.loc[to_numeric(df[v], errors='coerce') < eval(k)]

    if isinstance(pat[0], dict):
        for k4, j4 in pat[0].items():
            df2 = df[df[list([k4])].apply(lambda x: all([j4 in i for i in x]), axis=1)]
    elif isinstance(pat[0], int):
        df2 = df[df[list(pat)].apply(lambda x: all(["+" in i for i in x]), axis=1)]
        if strict:
            antipat = list(set([i for i in df.keys() if type(i) == int]) - set(pat))
            df2 = df[df[list(antipat)].apply(lambda x: not any(["+" in i for i in x]), axis=1)]
    print("Now your dataframe in 'df' variable")

    # with option_context('display.max_rows', None, 'display.max_columns', None):
    #     display(df2)
    return df2
def clustDec(array_of_mut_elm, type_="a"):
    d = defaultdict(list)
    final_list = []
    for item in array_of_mut_elm:
        if type_ == "a":
            d[tuple(item.data_a)].append(copy.deepcopy(item))
        elif type_ == "b":
            d[tuple(item.data_p)].append(copy.deepcopy(item))
        elif type_ == "c":
            return array_of_mut_elm
    for k, v in d.items():  # k = ('G', 'G', 'L', 'T'), v = [<mut...1>, .....]
        final_item = copy.deepcopy(v[0])
        final_item.qua = len(v)

        rv = lambda param, item: setattr(item, param, round((1.0 / len(v)) * sum([getattr(i6, param) for i6 in v]),
                                                            2))  # we exclude non - number-values from

        map(functools.partial(rv, item=final_item),
            list(set(final_item.__dict__.keys()) - set(['ide', 'name', 'qua', 'data_a', 'data_p', 'ed_id'])))
        final_list.append(final_item)
    return final_list

""" CLASSES """
class mut:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # def __init__(self, data_p, data_a, **kwargs):
        # def __init__(self, data_p, data_a, **kwargs):

        # for k5, v5 in kwargs.items():
        self.qua = 0
        self.ide = ""
class decoy:
    def __init__(self, ori, muts, id="NA", name="", i_s=0.0, r_s=0.0):
        self.ori = ori
        self.muts = muts
        self.id = id
        self.name = name
        self.i_s = i_s
        self.r_s = r_s
def all_mutations_p(ult_mut):
    a = []
    for i in ult_mut:
        a.extend(list(i.data_p))
    b = list(set(a))
    return (b)
def sort_list(a):
    return (sorted([int(i) for i in list(set(a))]))
def print_to_df(list_, ult_mut, default_seq, ar="a", conf=-1):
    if ar == "a":
        df_pretitle = [[" ", " ", " Wild type", " "] + [r1[1] for r1 in default_seq]]
        if conf != -1:
            list_ = sorted(filter(lambda x: int(x.name[0]) == conf, list_), key=lambda x1: x1.r_s)
        # for liuIdx, liuPos in range
        liu_list = zip(*filter(lambda k: all([i == k[0] for i in k]), zip(*[x.liu for x in list_])))
        # print(list_)
        df = [([x.i_s, x.r_s, x.ide, x.qua] + list(liu_list[xidx])) for xidx, x in enumerate(list_)]
        # df[0:0] = df_pretitle
        df2 = DataFrame(data=array(df, dtype=str),
                        columns=["{}-amp binding energy".format(ln), "TotScore", "SEQ ID", " Occurance"] + [
                            tuple3[0] + 1
                            for tuple3
                            in
                            default_seq])
    elif ar == "p":
        df_pretitle = [[" ", " ", " Wild type", " "] + [r1[1] for r1 in default_seq]]
        df = [([x.i_s, x.r_s, x.ide, x.qua] + x.liu) for x in ult_mut]
        df[0:0] = df_pretitle
        df2 = DataFrame(data=array(df, dtype=str),
                        columns=["{}-amp binding energy".format(ln), ln + "2-AMP binding energy", "SEQ ID",
                                 " Occurance"] + [tuple3[0] + 1
                                                  for tuple3
                                                  in
                                                  default_seq])
    elif ar == "r2":
        df_pretitle = [[" ", " ", " Wild type"] + [r1[1] for r1 in default_seq]]
        df = [([x.i_s, x.r_s, x.ide] + x.liu) for x in list_]
        df[0:0] = df_pretitle
        df2 = DataFrame(data=array(df, dtype=str),
                        columns=["{}-amp binding energy".format(ln), "TotScore", "SEQ ID"] + [tuple3[0] + 1 for tuple3
                                                                                              in
                                                                                              default_seq])
    global pymol_positions_visu
    pymol_positions_visu = [tuple3[0] for tuple3 in default_seq]
    df2.style.set_properties(**{'background-color': 'black', 'color': 'lawngreen', 'border': 1, 'border-color': 'white',
                                'border-width': '6'})
    return (df2)
def sc(command, silent=False):
    exp = "expect " + pr + "ex.sh " + " ' " + command + " '"
    if not silent:
        print("dddd", exp)
    os.chdir("/home/rustam/")
    os.system(exp)
def dpf(*args, **kwargs):
    label = ""
    if 'label' in kwargs.keys():
        label = kwargs['label']
    for filename in args:
        if os.path.exists(os.path.abspath(filename)):
            command = """<h6 style="margin-bottom: 3px; !important">%(but_name)s:</h6>_________________________________<button id="id%(but_id)s">
              %(s)s
            </button>
            <div id="id%(div_id)s" style=" font-family: monospace; height: 400px; overflow: scroll; display: none; border: 2px solid darkgreen; border-radius: 5px; padding: 5px; margin-top: 5px;">
            </div>
            <script src="https://code.jquery.com/jquery-1.11.0.min.js"></script>
            <style>
            #iframe_disp {
              width: 700px;
              height: 400px;
              }
            </style>
            <script type="text/javascript">
              $("#id%(but_id)s").click(function() {
              data = `%(data_)s`
              data = data.replace(/\\n/g, "<br>").replace(/ /g, "&nbsp;");
              $("#id%(div_id)s").html(data).toggle();
                });
            </script>""" % {'s': filename, 'div_id': random.randint(1, 3535), 'but_id': random.randint(1, 3535),
                            'but_name': label, 'data_': open(filename).read()}
            display(HTML(command))
        else:
            display("File {} not found".format(filename))
def list_selection(selection_name="sele", srv_md=srv_md, sel="NA", sc=False):
    if srv_md:
        tmp_file_name = "/home/domain/rustam/tmp/pml.txt"
    else:
        tmp_file_name = "/r/tmp/pml.txt"
    listCmd = """select %s
list=[]
iterate (%s),list.append((resi))
print(list)
import os
os.system('echo {}  > %s '.format(str(list)))""" % (selection_name, selection_name, tmp_file_name)
    if srv_md:
        cmd.do(listCmd)
    else:
        pml.do(listCmd)
    if srv_md == False: tmp_file_name = "/home/domain/rustam/tmp/pml.txt"
    print(tmp_file_name)
    selection_string = open(tmp_file_name).read()
    selection_list = [i.strip() for i in re.search(r'(?<=\[).*(?=\])', selection_string).group().split(",")]
    # print(selection_list)
    if not selection_list[0]:
        return []
    else:
        return sorted([int(i) for i in list(set(selection_list))])
def scheme_html(schemes_available):
    sa_html_pattern = ""
    img_name = ""
    av_sc_file = open("/r/dgfl/misc/available_schemes.html").read()
    for k, sa in schemes_available.items():
        img_name = sa["img"]
        sa_html_pattern += '<tr><td><center>{}</center></td><td><center>{}</center></td><td><center><img src="http://localhost:8899/files{}"></center></td></tr>'.format(
            k, sa["code"], img_name)
    but_id, tn_1, tn_2, st = [random.randrange(1, 999999) for i in range(4)]
    sch = av_sc_file % {"sch": sa_html_pattern, "tn_1": tn_1, "tn_2": tn_2, "but_id": but_id, "st": st}
    display(HTML(sch))
def scheme_name_gen(submit):
    new_scheme_name = ""
    if submit == True:
        new_scheme_name = "unk{}".format(random.randrange(1, 999999))
    elif submit != False:
        new_scheme_name = submit.replace(" ", "_")
    if new_scheme_name:
        return new_scheme_name
def scheme_set_view_clear(new_scheme):
    sv_reg = r'set_view \(.*\);'
    new_scheme = re.sub(sv_reg, "", new_scheme)
    return new_scheme
def av_values(rmsds):
    rmsds = [i[:, 1] for i in rmsds]  # ONLY FOR STEP = 1!!!!!!!!!!!!!!!!!!!!!!!!
    avRmsds = []
    for rmsIdx in range(max(map(len, rmsds))):
        av = []
        for rmsd in rmsds:
            if rmsIdx < len(rmsd): av.append(rmsd[rmsIdx])
        avRmsds.append(np.mean(av))
    return np.array(list(enumerate(avRmsds)))
def pmlView():
    yv = [round(i, 3) for i in pml.get_view()]
    mcell("pml.set_view({})".format(yv), md=False)
def pmlImg(srv_md=True, r=False, size=[1200, 1200], pilot=False, cap=""):
    pilot = "300, 300" if pilot else "{}, {}".format(*size)
    pm("bg_color white", "set ray_trace_mode, 1", "set ray_opaque_background, 1", "ray {}".format(pilot), srv_md=srv_md,
       r=r)
    imgDir = "../DGFL/imgs/"
    mkDir(imgDir)
    fileIdxs = sorted([int(os.path.splitext(f)[0]) for f in os.listdir(imgDir) if re.match(r'[0-9]+.png', f)],
                      reverse=False)
    fileIdx = 0 if not fileIdxs else fileIdxs[-1] + 1
    pdbName = "{}{}.png".format(imgDir, fileIdx)
    pm("save {}".format(pdbName), srv_md=srv_md, r=r)
    sp("convert {}{}.png -trim {}{}.png".format(imgDir, fileIdx, imgDir, fileIdx))
    mcell(u'<img width="350" src="{}"><figcaption>{}</figcaption>'.format(pdbName, cap))
def mol2graph(mol):
    admatrix = rdmolops.GetAdjacencyMatrix(mol)
    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    adlist = np.ndarray.tolist(admatrix)
    graph = igraph.Graph()
    g = graph.Adjacency(adlist).as_undirected()
    for idx in g.vs.indices:
        g.vs[idx]["smth"] = mol.GetAtomWithIdx(idx).GetPropNames()
        g.vs[idx]["AtomicNum"] = mol.GetAtomWithIdx(idx).GetAtomicNum()
        g.vs[idx]["AtomicSymbole"] = mol.GetAtomWithIdx(idx).GetSymbol()
    for bd in bondidxs:
        btype = mol.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble()
        g.es[g.get_eid(bd[0], bd[1])]["BondType"] = btype
    return g
def traj_from_selection(selection, trajectory, tprName="md.tpr", selTrjName=".tmp/tmp.xtc"):
    mkDir(".tmp")
    trjSelection = trju2.select_atoms(selection)
    with MDAnalysis.Writer(selTrjName, trjSelection.n_atoms) as W:
        for ts in trju2.trajectory:
            W.write(trjSelection)
    return mda.Universe(selTrjName)
def pdb_remove_duplicates(file_path):
    with open(file_path, "r+") as f:
        lines = f.readlines()
        new_l = []
        prev_ = ""
        for line in lines:
            if line.startswith("CON"):
                if prev_ != line:
                    new_l.append(line)
            else:
                new_l.append(line)
            prev_ = line
        f.seek(0)
        f.truncate()
        for j in new_l:
            f.write(j)
def slice_file(start_, end_, fileName):
    command_ = """sed -n '/%(s)s/ {
$! {
:loop
N
/%(e)s/p
/%(e)s/! b loop
}}' %(f)s""" % {'s': start_, "e": end_, "f": fileName}
    return os.popen(command_).read()
def getCoordMap(core2, match):
    coordMap = {}
    core2Conf = core2.GetConformer(-1)
    for i, idxI in enumerate(match):
        core2PtI = core2Conf.GetAtomPosition(i)
        coordMap[idxI] = core2PtI
    return coordMap
def ConstrainedEmbed(mol, core, db=False, randomseed=-1, mnz_tries=lg_mnz_tries, energyTol=lg_mnz_etol,
                     forceTol=lg_mnz_ftol, maxIts=lg_mnz_mxIts, lg_mnz_cst=lg_mnz_cst,
                     getForceField=AllChem.UFFGetMoleculeForceField, doMap=True, cst1d=False, ln="bff"):
    """ modified rdkit.AllChem.ConstrainedEmbed"""
    if db: note("Constrained Embed started")
    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(-1)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI
    coordMap = {} if not doMap else coordMap
    ci = AllChem.EmbedMolecule(mol, clearConfs=True, coordMap=coordMap, randomSeed=randomseed)
    if ci < 0:
        raise ValueError('Could not embed molecule.')
    elif db:
        note("Molecule successfully embedded")
    algMap = [(j, i) for i, j in enumerate(match)]
    # rotate the embedded conformation onto the core:
    rms = AllChem.AlignMol(mol, core, atomMap=algMap)
    ff = getForceField(mol, confId=0)
    conf = core.GetConformer()
    for i in range(core.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
        ff.AddDistanceConstraint(pIdx, match[i], 0, 0, lg_mnz_cst)
    if isinstance(cst1d, list):
        # print(cst1d)
        # tors_cst_atms = {'rrf': [22,20,19,18]}[ln]
        # ff.UFFAddTorsionConstraint(*tors_cst_atms + [False, 153.7, 205.0, 1000.0])
        ff.UFFAddTorsionConstraint(*cst1d)
    ff.Initialize()
    n = mnz_tries  # default = 40
    if db: vd(n, "number of minimization tries")
    ff.Minimize()
    more = ff.Minimize(energyTol=energyTol, forceTol=forceTol,
                       maxIts=maxIts)  # forceTol default = 1e-4 and energyTol default = 1e-06
    # print("more: {}".format(more))
    # if db: vd(more, "more before while")
    # while more and n:
    #     more = ff.Minimize(energyTol=energyTol, forceTol=forceTol, maxIts=maxIts)
    #     n -= 1
    #     if db: vd(n), vd(more)
    # if db: vd(more, "more after while")
    # # realign
    rms = AllChem.AlignMol(mol, core, atomMap=algMap)
    en = float(ff.CalcEnergy())
    if db: note("Constrained Embed finished, en: {}; rms: {} ".format(en, rms))
    return mol, en, rms
def rndTors(Core, cnf, mol, algMap=None):
    torsAtms = [45, 15, 16, 17, 18, 19, 20, 22][::-1]
    torsAtms = [torsAtms[i:i + 4] for i in range(len(
        torsAtms) - 3)]  # [[13, 15, 16, 17], [15, 16, 17, 18], [16, 17, 18, 19], [17, 18, 19, 20], [18, 19, 20, 21], [19, 20, 21, 22]]
    for torsAtm in torsAtms:
        rdkit.Chem.rdMolTransforms.SetDihedralDeg(cnf, *torsAtm + [random.randrange(0, 360)])
    return mol
def wrTrjMegFile(otp="d1.pse", st=True):
    constrAALiuw = ".tmp/constrAALiuw.pdb"
    pm("load {}".format(constrAALiuw), "hide cart", "show spheres", "zoom org", r=False)
    pm(*["create cns, constrAALiuw, 1, {}".format(j) for j in range(40)], r=False)
    if st: pm("hide spheres", "show sticks", r=False)
    pm("delete constrAALiuw", "save {}".format(otp), r=False)
    say("pse wrote")
def mnzTrj(ff, mol, pmName="mol", n=15, otp=False, maxIts=100, st=True, db=True):
    if db: loadCnf2Pml(pmName, mol, srv_md=True, r=False)
    for j in range(n):
        more = ff.Minimize(maxIts=maxIts)
        if db: vd(more)
        if db: loadCnf2Pml(pmName, mol, srv_md=True, r=False)
def getLiuwMol(ref="/home/domain/data/rustam/dgfl/ed/ed_4T0/ri/ss/liuw_rlx_0/lpla_lg_0_0001.pdb", rw=True, db=True,
               coreMol=False, extraLinkAt=0):
    rndIdx = random.randint(1, 5000)
    rndName = ".tmp/getLiuwMolTransient_{}.sdf".format(rndIdx)
    pm("load {}, liuw_ss".format(ref), "save {}, r. RRF & liuw_ss".format(rndName))
    bff = prepLig(ref=rndName, db=False, extraLinkAt=extraLinkAt)
    coreMol = coreMol if coreMol else bff.laq_core
    if rw:  # rw=False not ready!!!!
        while not 'rms' in locals().keys() or rms > 0.1:  # It's important to minimize rms because amp moiety will be fixed during further procedures
            mol, en, rms = ConstrainedEmbed(Chem.Mol(bff.lig), coreMol, randomseed=-1, db=False, lg_mnz_cst=100,
                                            doMap=False, maxIts=1000)
            if db: note("Generating starting conf", vd(rms), vd(en))
        wrpdb(mol, ".tmp/cstEm_{}.pdb".format(rndIdx), )
    note("Writing starting conf: .tmp/cstEm_{}.pdb".format(rndIdx))
    mol = Chem.rdmolfiles.MolFromPDBFile(".tmp/cstEm_{}.pdb".format(rndIdx),
                                         sanitize=False)  # Upload start. conf from file
    mol = AllChem.AssignBondOrdersFromTemplate(bff.lig, mol)
    match = mol.GetSubstructMatch(bff.laq_core)
    algMap = [(j, i) for i, j in enumerate(match)]
    return (bff, Chem.Mol(mol), match, algMap)
def parseFunArgs(args):
    args = 'f({})'.format(args)
    tree = ast.parse(args)
    funccall = tree.body[0].value
    rtArgs, rtKwargs = [], []
    for arg in funccall.args:
        if hasattr(arg, "id"):
            rtArgs.append("NOT_A_STR_{}".format(arg.id))
        else:
            rtArgs.append(ast.literal_eval(arg))

    for arg in funccall.keywords:
        if hasattr(arg.value, "id"):
            rtKwargs.append((arg.arg, "NOT_A_STR_{}".format(arg.value.id)))
        else:
            rtKwargs.append((arg.arg, ast.literal_eval(arg.value)))
    return rtArgs, rtKwargs
def who_am_i():
    stack = traceback.extract_stack()
    filename, codeline, funcName, text = stack[-2]
    return funcName
def call_n_def_args2str(callArgs, defaultArgs):
    """callArgs:
(['str4inp1', 'str4inp2', 'lig_B0.pdb'], [('db', 'NOT_A_STR_True')])
defaultArgs:
(['NOT_A_STR_inp1', 'NOT_A_STR_inp2'],
 [('inp', 'lig.pdb'),
  ('otp', 'NOT_A_STR_False'),
  ('ref',
   '/home/domain/data/rustam/liuw_acpype/BFF/ed_RD3_ri_cg_mp_mpc_uncl_str_1_molecule_4_AllChem_AssignBondOrdersFromTemplate.pdb'),
  ('db', 'NOT_A_STR_True')])"""
    db = True
    ultVarMap = []
    callPosArgs = callArgs[0][:len(defaultArgs[0])]
    callPosKWArgs = callArgs[0][len(defaultArgs[0]):]
    defPosKWArgs = defaultArgs[1][:len(callPosKWArgs)]
    preDefKwargs = defaultArgs[1][len(callPosKWArgs):]
    defKwargs = []
    for defKwarg, defKwargValue in preDefKwargs:
        if defKwarg in dict(callArgs[1]).keys():
            defKwargs.append((defKwarg, dict(callArgs[1])[defKwarg]))
        else:
            defKwargs.append((defKwarg, defKwargValue))
    if db: note("", vd(callPosArgs), vd(callPosKWArgs), vd(defPosKWArgs), vd(defKwargs))
    ultVarMap.extend(zip(defaultArgs[0], callPosArgs, ))
    ultVarMap.extend(zip([j[0] for j in defPosKWArgs], callPosKWArgs))
    ultVarMap.extend(defKwargs)
    if db: vd(ultVarMap)
    ultVarStrs = []
    for k, v in ultVarMap:
        if db: note("", vd(k), vd(v))
        formatStr = "{} = {}" if type(v) in [int, dict, list] or v.startswith("NOT_A_STR_") else "{} = \"{}\""
        if type(v) in [str]:
            ultVarStrs.append(formatStr.format(k.split("NOT_A_STR_")[-1], v.split("NOT_A_STR_")[-1]))
        elif type(v) in [int, dict, list]:
            ultVarStrs.append(formatStr.format(k.split("NOT_A_STR_")[-1], v))
    # RETURN STR: inp1 = "str4inp1"; inp2 = "str4inp2"; inp = "lig_B0.pdb"; otp = False; ref = "/home/domain/data/rustam/liuw_acpype/BFF/ed_RD3_ri_cg_mp_mpc_uncl_str_1_molecule_4_AllChem_AssignBondOrdersFromTemplate.pdb"; db = True
    return ("; ".join(ultVarStrs))
def extract_wrapped(decorated):
    closure = (c.cell_contents for c in decorated.__closure__)
    return next((c for c in closure if isinstance(c, FunctionType)), None)
def changeMolIdxs(inp="lig.pdb", otp=False,
                  ref="/home/domain/data/rustam/liuw_acpype/BFF/ed_RD3_ri_cg_mp_mpc_uncl_str_1_molecule_4_AllChem_AssignBondOrdersFromTemplate.pdb",
                  db=True, coef=1):
    refMol0 = rdpdb(ref)
    refMol = Chem.Mol(refMol0)
    refMol.RemoveAllConformers()
    refMol.AddConformer(refMol0.GetConformer(0), assignId=True)
    decMol = AllChem.AssignBondOrdersFromTemplate(refMol, rdpdb(inp, sanitize=False))

    assert decMol.GetNumAtoms() == refMol.GetNumAtoms()
    decCnf = decMol.GetConformer()
    refCnf = refMol.GetConformer(0)
    if db: loadCnf2Pml("decMol", decMol, srv_md=False); loadCnf2Pml("refMol", refMol, srv_md=False)
    for decIdx, refIdx in enumerate(decMol.GetSubstructMatch(refMol)):
        decPos = decCnf.GetAtomPosition(refIdx)
        refCnf.SetAtomPosition(decIdx, decPos / coef)
    if db: loadCnf2Pml("decMol", decMol, srv_md=False); loadCnf2Pml("refMol", refMol, srv_md=False); pma(
        "label org, rank", "set label_position, (0,0,2)")
    if otp:
        wrpdb(refMol, otp)
    return refMol
def getLiuwMolWrapper(rw=True, silent=False,
                      ref="/home/domain/data/rustam/dgfl/ed/ed_4T0/ri/ss/liuw_rlx_0/lpla_lg_0_0001.pdb"):
    go = False
    while not go:
        try:
            bff, mol, match, algMap = getLiuwMol(rw=rw, ref=ref)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
            note("minimizing...")
            ff.Initialize()
            ff.Minimize(maxIts=10)
            go = True
        except RuntimeError:
            warn("Runtime Error, repeating...")
        except ValueError:  # Sanitization: Explicit valence for atom # 72 H, 2, is greater than permitted
            warn("Sanitization Error, repeating...")
    if silent: clear_output()
    return ff, bff, mol, match, algMap
def setTor(cnf, FltTors, torsAtmsLists=torsAtmsLists):  # FltTors = [0,60,60,60,60]
    for torsAtmIdx, torsAtmList in enumerate(torsAtmsLists):
        torsVal = FltTors[torsAtmIdx]
        rdkit.Chem.rdMolTransforms.SetDihedralDeg(cnf, *torsAtmList + [torsVal])
def ecleaning(m, dizInp, efilter):
    diz = copy.copy(dizInp)
    diz.sort(key=lambda x: x[1])
    mini = float(diz[0][1])
    sup = mini + efilter
    n = Chem.Mol(m)
    n.RemoveAllConformers()
    newId = n.AddConformer(m.GetConformer(int(diz[0][0])), assignId=True)
    nid, ener = [], []
    nid.append(newId)
    ener.append(float(diz[0][1]))
    del diz[0]
    for x, y in diz:
        if y <= sup:
            newId = n.AddConformer(m.GetConformer(int(x)), assignId=True)
            nid.append(newId)
            ener.append(float(y))
        else:
            break
    diz2 = zip(nid, ener)
    diz2.sort()
    return n, diz2
def postrmsd(n, diz2Inp, rmspost):
    diz2 = copy.copy(diz2Inp)
    diz2.sort(key=lambda x: x[1])
    # print(diz2)
    o = Chem.Mol(n)
    o.RemoveAllConformers()
    confidlist = [diz2[0][0]]
    enval = [diz2[0][1]]
    nh = Chem.RemoveHs(n)
    del diz2[0]
    for w, z, in diz2:
        confid = int(w)
        p = 0
        for conf2id in confidlist:
            rmsd = AllChem.GetBestRMS(nh, nh, prbId=confid, refId=conf2id)
            if rmsd < rmspost:
                p = p + 1
                break
        if p == 0:
            confidlist.append(int(confid))
            enval.append(float(z))
    newConfidlist = []
    for id in confidlist:
        newId = o.AddConformer(n.GetConformer(id), assignId=True)
        newConfidlist.append(newId)
    diz3 = zip(newConfidlist, enval)
    diz3.sort()
    return o, diz3
def GetDist(l1, l2):
    return math.sqrt(sum(map(lambda i: (l1[i] - l2[i]) ** 2, range(3))))
def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))
def length(v):
    return math.sqrt(dotproduct(v, v))
def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
def IsNextTo(v1, v2, refDistV):
    for k in range(len(v1)):
        if GetDist(v1[k], v2[k]) > refDistV[k]:
            return False
    return True
def getRefCoords(sphCstAts=sphCstAts, ref="../../ed_4T0/ri/ss/liuw_rlx_0/lpla_lg_0_0001.pdb", rw=True):
    dec_str = PDBParser().get_structure("ref", ref)
    refCors = [list(dec_str[0]["X"].child_list[0].child_dict[j].get_coord()) for j in sphCstAts]
    return refCors
def sphFlt(cnf, refCors, sphCstRad=[2, 2, 2, 2, 2, 2], sphCstMolIdxs=sphCstMolIdxs):
    bffCors = [cnf.GetAtomPosition(sphCstMolIdxs[refCorIdx]) for refCorIdx, refCor in enumerate(refCors)]
    bffCors = [(i.x, i.y, i.z) for i in bffCors]
    return IsNextTo(bffCors, refCors, sphCstRad)
def DirGetSphFltTors(ref="../../ed_4T0/ri/ss/liuw_rlx_0/lpla_lg_0_0001.pdb", db=False, step=60, rw=False,
                     sphCstRad=[0.8, 1.0, 2, 2, 0.8, 0.8], sphCstAts=sphCstAts, sphCstMolIdxs=sphCstMolIdxs,
                     useSavedTors=False):
    chainIdx = int(ref.split("liuw_rlx_")[-1].split("lpla_lg2_")[-1][0])
    #     if rw and not useSavedTors: rmFile(".tmp/FltTors{}.txt".format(chainIdx))
    if db:
        pm("load ../../ed_4T0/ri/ss/liuw_rlx_0/lpla_lg_0_0001.pdb", )
        for atIdx, atName in enumerate(sphCstAts):
            pm("select sph_{0}, org & n. {0}".format(atName), "show spheres, sph_{}".format(atName), r=False)
            pm("alter sph_{}, vdw={}".format(atName, sphCstRad[atIdx]), r=False)
        pm("rebuild", r=False)
        decVisu(pmName=".tmp/d3.pse", r=False)
        pm("orient n. C25 & org", "save .tmp/d3.pse", r=False)
    refCors = getRefCoords(ref=ref, rw=True)
    ff, bff, mol, match, algMap = getLiuwMolWrapper(rw=True, silent=False, ref=ref)
    display(mol)
    cnf = mol.GetConformer()
    torsAtms = [22, 20, 19, 18, 17, 16, 15, 45]
    torsAtms = [torsAtms[i:i + 4] for i in range(len(
        torsAtms) - 3)]  # [[22, 20, 19, 18], [20, 19, 18, 17], [19, 18, 17, 16], [18, 17, 16, 15], [17, 16, 15, 45]]
    torsGen = range(0, 360, step)
    accMol = Chem.Mol(mol)
    accMol.RemoveAllConformers()
    acBffCors = []
    if useSavedTors:
        savedTors = eval(open(".tmp/FltTors{}".format(chainIdx)).read())
    numIts = (360 / step) ** 5 if not useSavedTors else len(savedTors)
    print("Chain idx: {}. Calculated runTime: {}m. MayBe {}->{} cnfs.".format(chainIdx,
                                                                              round((numIts / (248832 / 46.0) / 60.0),
                                                                                    2), numIts,
                                                                              (56 / 248832.0) * numIts))
    if useSavedTors:
        for (i0, i1, i2, i3, i4) in savedTors:
            setTor(cnf, [i0, i1, i2, i3, i4])
            if sphFlt(cnf, refCors, sphCstRad=sphCstRad):
                acBffCors.append([eval("i{}".format(j)) for j in range(len(torsAtms))])
                accMol.AddConformer(cnf, assignId=True)
    else:
        for i0 in torsGen:
            for i1 in torsGen:
                for i2 in torsGen:
                    for i3 in torsGen:
                        for i4 in torsGen:
                            setTor(cnf, [i0, i1, i2, i3, i4])
                            if sphFlt(cnf, refCors, sphCstRad=sphCstRad):
                                acBffCors.append([eval("i{}".format(j)) for j in range(len(torsAtms))])
                                accMol.AddConformer(cnf, assignId=True)
    #                     break
    #                 break
    #             break
    #         break
    if rw: open(".tmp/FltTors{}.txt".format(chainIdx), "w").write(str(acBffCors))
    if not rw: warn("not writing file!!!")
    say("Finished")
    print("ChainIdx {} Finished. {}  -> {}".format(chainIdx, numIts, len(acBffCors)))
    return acBffCors, accMol
@prntFun
def ConstrainedEmbedLiuw(db=False, randomseed=-1, mnz_tries=lg_mnz_tries, energyTol=lg_mnz_etol,
                         forceTol=lg_mnz_ftol, maxIts=lg_mnz_mxIts, lg_mnz_cst=lg_mnz_cst,
                         getForceField=AllChem.UFFGetMoleculeForceField, liuw=False,
                         rw=True, doMnz=True, numCnfs=0, wrPse="", sphCstMolIdxs=sphCstMolIdxs, rndGen=False):
    """ modified rdkit.AllChem.ConstrainedEmbed"""
    numCnfs = 10 ** 7 if not numCnfs else numCnfs
    refCors = getRefCoords(ref="../../ed_4T0/ri/ss/liuw_rlx_0/lpla_lg_0_0001.pdb", rw=True)
    ff, bff, mol, match, algMap = getLiuwMolWrapper(rw=rw, silent=True)
    Core = bff.laq_core
    conf = Core.GetConformer()
    for i in range(Core.GetNumAtoms()):  # Constrain LAQ:
        p = conf.GetAtomPosition(i)
        pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
        ff.AddDistanceConstraint(pIdx, match[i], 0, 0, 2000)
    pm("")
    molCnf = mol.GetConformer(0)
    display(mol)
    loadCnf2Pml("initMol", mol, srv_md=True)
    ff.Initialize()
    prot = Chem.rdmolfiles.MolFromPDBFile(".tmp/constrAALiuw.pdb")
    fltMol = Chem.Mol(mol)
    fltMol.RemoveAllConformers()
    t0 = time.time()
    nTries = 0
    j2 = 0
    if not rndGen: FltTorsList = eval(open(".tmp/FltTors.txt").read())
    while fltMol.GetNumConformers() < numCnfs:
        if not rndGen:
            setTor(molCnf, FltTorsList[nTries])
        else:
            mol = rndTors(Core, molCnf, mol, algMap)
            if not sphFlt(molCnf, refCors, sphCstMolIdxs):
                continue
        nTries += 1
        # ptr_dist = rdkit.Chem.rdShapeHelpers.ShapeProtrudeDist(mol, prot, 0, 0, vdwScale=0.75,
        #                                                        gridSpacing=0.00001, stepSize=0.0)
        # if ptr_dist != 1:
        #     continue
        if doMnz:
            mnzTrj(ff, mol, n=1, pmName="mol_{}".format(j2), maxIts=800)
            j2 += 1
            if not sphFlt(molCnf, refCors, sphCstMolIdxs):
                continue
            # ptr_dist = rdkit.Chem.rdShapeHelpers.ShapeProtrudeDist(mol, prot, 0, 0, vdwScale=0.75,
            #                                                        gridSpacing=0.00001, stepSize=0.0)
            # if ptr_dist != 1:
            #     continue
            fltMol.AddConformer(mol.GetConformer(0), assignId=True)
        else:
            fltMol.AddConformer(mol.GetConformer(0), assignId=True)
    note("ConfGen finished in {}. {} -> {}".format(time.time() - t0, nTries, fltMol.GetNumConformers()))
    if wrPse: wrTrjMegFile("d1.pse", st=True)
    return fltMol
    n = mnz_tries  # default = 40
    if db: vd(n, "number of minimization tries")
    more = ff.Minimize(energyTol=energyTol, forceTol=forceTol,
                       maxIts=5000)  # forceTol default = 1e-4 and energyTol default = 1e-06
    vd(more)
    # if db: vd(more, "more before while")
    # while more and n:
    #     more = ff.Minimize(energyTol=energyTol, forceTol=forceTol, maxIts=maxIts)
    #     n -= 1
    #     if db: vd(n), vd(more)
    # if db: vd(more, "more after while")
    # realign
    rms = AllChem.AlignMol(mol, Core, atomMap=algMap)
    en = float(ff.CalcEnergy())
    if db: note("Constrained Embed finished")
    return mol, en, rms
def sliceMol(Mol1, numCnfs):
    Mol2 = Chem.Mol(Mol1)
    Mol2.RemoveAllConformers()
    for i in range(numCnfs):
        Mol2.AddConformer(Mol1.GetConformer(i))
    return Mol2
def cnt2names(refPath):
    bas, acs = [], []
    for natCnt in cntRouting(refPath):
        print(natCnt)
        newNatCnt = []
        for cnt in natCnt:
            if isinstance(cnt, tuple):
                newNatCnt.append("name " + " ".join(cnt))
            elif isinstance(cnt, str):
                newNatCnt.append("name " + cnt)
            elif isinstance(cnt, int):
                newNatCnt.append("resid {}".format(cnt))
        sel_basic = "(({}) and ({}))".format(*newNatCnt[:2])
        sel_acidic = "(resname UNL) and ({})".format(newNatCnt[2])
        bas.append(sel_basic)
        acs.append(sel_acidic)
    return (zip(bas, acs))
def ConstrainedEmbedBAD(mol, Core, db=False, randomseed=-1, mnz_tries=lg_mnz_tries, energyTol=lg_mnz_etol,
                        forceTol=lg_mnz_ftol, maxIts=lg_mnz_mxIts, lg_mnz_cst=lg_mnz_cst,
                        getForceField=AllChem.UFFGetMoleculeForceField, liuw=False):
    """ modified rdkit.AllChem.ConstrainedEmbed"""
    if db: note("Constrained Embed started")
    match = mol.GetSubstructMatch(Core)
    if not match:
        raise ValueError("molecule doesn't match the Core")
    # coordMap = {}
    # coreConf = Core.GetConformer(-1)
    # for i, idxI in enumerate(match):
    #     if liuw and i in [39, 40, 41, 42, 43, 44]:
    #         continue
    #     corePtI = coreConf.GetAtomPosition(i)
    #     coordMap[idxI] = corePtI
    # # if liuw:
    # #     rndResidueIdxs = [20, 21, 24]
    # #     liuw_rlx = "../../ed_4T0/ri/ss/liuw_rlx_0/lpla_lg_0_0001.pdb"
    # #     bbNames = ['CA', 'C', 'O', 'N']
    # #     rndOxyCoords = []
    # #     dec_str = PDBParser().get_structure("dec", liuw_rlx)
    # #     for rndResidueIdx in rndResidueIdxs:
    # #         bbAtms = filter(lambda it: it[0] in bbNames,
    # #                         dec_str[0]["A"].child_list[rndResidueIdx - 1].child_dict.items())
    # #         for bbAtm in bbAtms:
    # #             rndOxyCoords.append(map(lambda x: round(x, 3), list(bbAtm[1].get_coord())))
    # # ci = AllChem.EmbedMolecule(mol, clearConfs=True, coordMap=coordMap, randomSeed=randomseed)
    #     # Adding constraints for trailing oxygen
    #     rndOxyPoint = random.choice(rndOxyCoords)
    #     pIdx = ff.AddExtraPoint(*rndOxyPoint, fixed=True) - 1
    #     ff.AddDistanceConstraint(pIdx, 6, 0, 0, 2.0)
    # ci = AllChem.EmbedMolecule(mol, clearConfs=True,randomSeed=randomseed)
    # constrAALiuw = ".tmp/constrAALiuw.pdb"
    # constrAALiuwMol = Chem.rdmolfiles.MolFromPDBFile(constrAALiuw)
    # constrAALiuwConf = constrAALiuwMol.GetConformer(0)
    # if ci < 0:
    #     raise ValueError('Could not embed molecule.')
    # elif db:
    #     note("Molecule successfully embedded")
    # algMap = [(j, i) for i, j in enumerate(match)]
    # # rotate the embedded conformation onto the Core:
    # rms = AllChem.AlignMol(mol, Core, atomMap=algMap)
    # # mol = rdpdb("cns2.pdb")
    mol = Chem.rdmolfiles.MolFromPDBFile("../../ed_RD3/ri/cg/BFF.pdb", sanitize=False)
    mol = AllChem.AssignBondOrdersFromTemplate(lgd.lig, mol)
    ff = getForceField(mol, confId=0)
    conf = Core.GetConformer()

    def constrainLaq(ff, Core=Core, conf=conf, ):
        for i in range(Core.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
            vd(pIdx)
            ff.AddDistanceConstraint(pIdx, match[i], 0, 0, lg_mnz_cst)
        return ff

    def rndTors(cnf):
        torsAtms = [13] + range(15, 23)
        torsAtms = [torsAtms[i:i + 4] for i in range(len(
            torsAtms) - 3)]  # [[13, 15, 16, 17], [15, 16, 17, 18], [16, 17, 18, 19], [17, 18, 19, 20], [18, 19, 20, 21], [19, 20, 21, 22]]
        for torsAtm in torsAtms:
            rdkit.Chem.rdMolTransforms.SetDihedralDeg(cnf, *torsAtm + [random.randrange(0, 180)])

    ff = constrainLaq(ff)
    for _ in range(10):
        rndTors(conf)

    pm("")
    # mnzTrj(ff, mol, n=5,)

    # mol = Chem.Mol(mol)
    # ff2 = AllChem.UFFGetMoleculeForceField(mol2, confId=0)
    # ff2 = getForceField(mol, confId=0)
    # ff2.Initialize()
    # ff2 = constrainLaq(ff)
    # mnzTrj(ff, mol, n=10,)
    # ff = getForceField(mol, confId=0)
    # ff.Initialize()
    # constrainLaq()
    # Adding constraints for backbone:
    # return constrAALiuwConf
    # for AtmIdx in range(mol.GetNumAtoms()):
    #     if AtmIdx not in match and AtmIdx == 1: # "Core" atoms are already fixed and don't need any constraints
    #         for cstAtmIdx in range(constrAALiuwConf.GetNumAtoms()):
    #             if cstAtmIdx == 32:
    #                 p = constrAALiuwConf.GetAtomPosition(cstAtmIdx)
    #                 pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
    #                 ff.AddDistanceConstraint(pIdx, AtmIdx, 3.0, 100.0, -2)
    #                 print(pIdx)
    #             # break
    #     # break
    ff.Initialize()
    mnzTrj(ff, mol, n=15, otp="d1.pse", maxIts=300)
    return
    #

    n = mnz_tries  # default = 40
    if db: vd(n, "number of minimization tries")
    more = ff.Minimize(energyTol=energyTol, forceTol=forceTol,
                       maxIts=5000)  # forceTol default = 1e-4 and energyTol default = 1e-06
    vd(more)
    # if db: vd(more, "more before while")
    # while more and n:
    #     more = ff.Minimize(energyTol=energyTol, forceTol=forceTol, maxIts=maxIts)
    #     n -= 1
    #     if db: vd(n), vd(more)
    # if db: vd(more, "more after while")
    # realign
    rms = AllChem.AlignMol(mol, Core, atomMap=algMap)
    en = float(ff.CalcEnergy())
    if db: note("Constrained Embed finished")
    return mol, en, rms
def makeVmpc():
    inds = [re.search(r'(?<=cg/mp/mpc_uncl_str_).*(?=.pdb)', i).group() for i in
            glob.glob('cg/mp/mpc_uncl_str_*.pdb')]  # inds = ['5', '0', '1', '2']
    pm("")
    for ind in inds:
        pm("load cg/mp/mpc_uncl_str_{}.pdb".format(ind), "load cg/cl_str/cl_str_{}.pdb".format(ind),
           "util.cbag mpc_uncl_str_{}".format(ind), "util.cbao cl_str_{}".format(ind),
           "create mpc_pair_{0}, *cl_str_{0}".format(ind), r=False)
    pm("delete *cl_str*", "remove e. h", "set pse_export_version, 1.721", "save ../rpr/vmpc.pse", r=False)
def mpc_frz(file_name, mpc_laq_constr_list=None):
    with open(file_name, "r+") as f:
        lines = f.readlines()
        newMop = []
        for line in lines:
            sp2 = line.split()
            if lines.index(line) > 2 and mpcOptimize:
                if mpc_laq_constr_list and not inCloseList(
                        map(lambda x: float(x), [sp2[1], sp2[3], sp2[5]]), [i[1] for i in mpc_laq_constr_list]):
                    continue
                p2 = line.split()
                newMop.append("{}  {} {} {} {} {:>8} {}".format(p2[0], p2[1], "0", p2[3], "0", p2[5], "0\n"))
            else:
                newMop.append(line)
        f.seek(0)
        f.truncate()
        f.write("".join(newMop))
        f.close()
def isclose(pair):
    a, b = pair
    return abs(a - b) <= 0.05
def inCloseList(a, list_):
    return any([all(map(isclose, zip(a, it))) for it in list_])
def float_r(x):
    return (round(float(x), 4))
def rmsFromRmsMat(data, nPts):
    dijs = {}
    dmIdx = 0
    for i in range(nPts):
        for j in range(i):
            dijs[(j, i)] = round(float(data[dmIdx]), 2)
            dmIdx += 1
    return dijs
def rms_fltr(fluro_core, lig_mol, confs, db=False, rprName="", btn_clst_dst_thrs=btn_clst_dst_thrs, visuThresh=2):
    # print("Recomendation: use rmsFromRmsMat for calibration")
    # if db:
    #     v = MolViewer()
    #     v.DeleteAll()
    #     for cf_idx in range(lig_mol.GetNumConformers()):
    #         v.ShowMol(lig_mol, 'mnz_conf_%s' % cf_idx, showOnly=False, confId=cf_idx)
    t5 = time.time()
    flr_rms_mtch = lig_mol.GetSubstructMatch(fluro_core)
    # RMS matrix
    print("rms_fltr: clustering started... (btn = {})".format(btn_clst_dst_thrs))
    pm("")
    rmsmat = AllChem.GetConformerRMSMatrix(lig_mol, prealigned=True, atomIds=flr_rms_mtch)
    # RMS clustering and efilteringj
    rms_clusters = Butina.ClusterData(rmsmat, len(confs), btn_clst_dst_thrs, isDistData=True, reordering=True)
    rms_clusters = [list(i) for i in rms_clusters]
    if db: note("clustering", vd(rmsmat), vd(rms_clusters), vd(confs))
    mnz_confs = []
    mnz_en_lig_mol = Chem.Mol(lig_mol)
    mnz_en_lig_mol.RemoveAllConformers()

    if rprName: rmsDict = rmsFromRmsMat(rmsmat, len(confs))
    # vd(rms_clusters)
    for clst_idx, rms_cluster in enumerate(rms_clusters):
        # rms_cluster.sort(key=lambda x: confs[x][1])
        mnz_en_item = rms_cluster[0]
        if rprName and len(rms_cluster) > visuThresh:
            # clear_output(); print("adding RMS with the first conformer (by idx) to name...")
            rms_cluster = sorted(rms_cluster)
            for cf_idx in rms_cluster:
                rms = 0.0 if rms_cluster[0] == cf_idx else rmsDict[(rms_cluster[0], cf_idx)]
                # print(cf_idx)
                loadCnf2Pml('clst_%s_%s_%s' % (clst_idx, cf_idx, rms), lig_mol, cf_idx)
            pm("color {}, clst_{}*".format(random.choice(pml_clrs), clst_idx), r=False)
        mnz_en, mnz_rms = [(i[1], i[2]) for i in confs if i[0] == mnz_en_item][0]
        new_id = mnz_en_lig_mol.AddConformer(lig_mol.GetConformer(mnz_en_item), assignId=True)
        mnz_confs.append((new_id, mnz_en, mnz_rms))
        if db: note("cluster filtration", vd(rms_cluster), vd(mnz_en_item), vd(mnz_confs))
    print("rms_fltr: {}->{} confs.\tTime: {} ".format(len(confs), len(mnz_confs), time.time() - t5))
    if rprName: pm("set pse_export_version, 1.721", "remove e. h", "save {}".format(rprName), r=False)
    return mnz_en_lig_mol, mnz_confs
def rms_fltr_wrapper(lig_mol_wrap):
    print("rms_flt_wrapper read  {} conformers. Launching filtering at pid: {}!".format(lig_mol_wrap.GetNumConformers(),
                                                                                        os.getpid()))
    chunkLigMol, chunkConfs = rms_fltr(anti_laq_core_mult, lig_mol_wrap, tupleConfs(lig_mol_wrap))
    return chunkLigMol, chunkConfs
def rar2():
    print('aaa')
@prntFun
def rms_fltr_multiprocessing(mol=None, chunkSize=3, recursion=False, inp="", otp="", rprName=""):
    if not recursion:
        if inp:
            mol = rdpdb(inp)
        global t_init, anti_laq_core_mult, shrinkingRatio, t2, numCnf
        t_init = time.time()
        numCnf = mol.GetNumConformers()
        print("Preparing to run rms_fltr_multiprocessing the first time ...")
        rwdir(".tmp4/")
        anti_laq_core_mult = prepLig().fluro_core
        display(anti_laq_core_mult)
        print("MPI Clusterization started with {} confs and btn: {} ".format(mol.GetNumConformers(), btn_clst_dst_thrs))
    t = time.time()
    TotalNumconfs = mol.GetNumConformers()
    if chunkSize == 1:
        print("Running last run with {} confs".format(mol.GetNumConformers()))
        mol, _ = rms_fltr(anti_laq_core_mult, mol, tupleConfs(mol))
        if otp:
            wrpdb(mol, otp)
        say("multiprocessed rms filtering complete")
        print("RESULT: {}->{}, time: {}".format(numCnf, mol.GetNumConformers(), time.time() - t_init))
        return mol
    else:
        chunks = chunkify(range(TotalNumconfs), chunkSize)
        octaMols = []
        for octaMerIdx, octaMer in enumerate(chunks):
            print("created chunck of len {}".format(len(octaMer)))
            transientMol = Chem.Mol(mol)
            transientMol.RemoveAllConformers()
            for j in octaMer:
                transientMol.AddConformer(mol.GetConformer(j), assignId=True)
            octaMols.append(transientMol)
        print("octaMols is a list of {} chem molecules".format(len(octaMols)))
        t2 = time.time()
        pool = Pool(14)
        print("Launching multiprocessing. {} conformers in {} chunks. Btn = {}.".format(TotalNumconfs, len(chunks),
                                                                                        btn_clst_dst_thrs))

        results = pool.map(rms_fltr_wrapper,
                           octaMols)  # [(<rdkit.Chem.rdchem.Mol object at 0x7f632a883260>, [(0, 0, 0), (1, 0, 0), (2 ......)]), (....), ]
        finalMol = Chem.Mol(mol)
        finalMol.RemoveAllConformers()
        for chunkMol, chunkConf in results:
            for chunkMolConfIdx in range(chunkMol.GetNumConformers()):
                finalMol.AddConformer(chunkMol.GetConformer(chunkMolConfIdx), assignId=True)
        print("time for all chuncks filtering: {}".format(time.time() - t2))
        chunckNumConfs = chunkMol.GetNumConformers()
        transientMolNumConfs = transientMol.GetNumConformers()
        conversion = chunckNumConfs / float(transientMolNumConfs)
        conversionCoef = (conversion / random.randrange(2, 3)) if conversion > 0.65 else 1.0
        # chunkSize = int(math.floor(
        #     chunkSize * chunckNumConfs) * conversionCoef / 700.0)  # e. g. 60 confs (average of 50 - 70 on each core) / 300.0 ("optimal number of cores for successful filtration.")

        chunkSize = 1 if not chunkSize else chunkSize
        note("Conversion params: ", vd(transientMolNumConfs), vd(chunckNumConfs), vd(conversionCoef), vd(chunkSize),
             "totNumConfs: ", chunkSize * chunckNumConfs)
        say("Enter Chunk Size")
        rawChunkSize = int(raw_input("Enter Chunk Size"))
        chunkSize = chunkSize if not rawChunkSize else rawChunkSize
        if chunkSize * chunckNumConfs < 20000:
            wrpdb(finalMol, '.tmp4/all_confs_teration_at_time_{}'.format(time.time() - t2))

        mol = rms_fltr_multiprocessing(finalMol, chunkSize, recursion=True, otp=otp)
        return mol
def ssGen(rw=False, leaveLiuCatSc=False):
    # pymol may not overwrite files. Cleaning directory:
    if rw:
        os.system("""find ss/ | grep -v "lpla_lg2_0001.pdb\|cst.py" | xargs rm""")
        pm("fetch 3a7r", "remove r. LAQ | r. so4 | r. hoh", "save ss/lpla.pdb")
        leaveLiuCatSc = "& ! {}".format(pymol_expr(liuCatRes)) if leaveLiuCatSc else " "
        if controlRunType != "NATIVE":
            pm("fetch 3a7r", "remove ( r. LAQ | r. so4 | r. hoh | e. mg | !( n. CA | n. O | n. C | n. N ) ) {} ".format(
                leaveLiuCatSc),
               "save ss/lpla_bb.pdb")
    lpla_str = Chem.rdmolfiles.MolFromPDBFile("ss/lpla_bb.pdb")
    print("Your structure contains {} atoms. Ref.: leaveLiuCatSc = False -> 1348 else -> 1414".format(
        lpla_str.GetNumAtoms()))
    return Chem.rdmolfiles.MolFromPDBFile("ss/lpla_bb.pdb")
def lsp(cmd_):
    sp(""" ssh fbbstudent@lomonosov2.parallel.ru "{}" """.format(cmd_))
def mopOpt(db=False, db2=False, preLom2=False, afterLom2="", mpcFile="cg/RCG_flt_str.pdb", pilot=False,
           ref="/home/domain/data/rustam/dgfl/ed/ed_RD3/ri/.tmp/test_laq.pdb", freeze="heavy", charge="-1"):
    # try:
    lom2MopList = []
    db2 = True if db else db2
    mkDir("cg")
    rwdir("cg/mp")
    rwdir("cg/cl_str")
    if afterLom2 == True:
        pass
        # prntFun(lscp)("{}ed_RD2/ri/cg/mp".format(lmpr), "/home/domain/data/rustam/dgfl/ed/ed_RD3/ri/cg/", toLom2=False, ed_id=False)
        # raise()
    elif afterLom2:
        if os.path.exists(afterLom2):
            shutil.rmtree("cg/mp/")
            shutil.copytree(afterLom2, "cg/mp/")
        else:
            warn("NO SUCH FILE OR DIRECTORY: {}".format(afterLom2))
    lgd = prepLig(ref=ref)
    mpc_uncl_supl = []
    os.popen("sed -i 's/^ENDMDL$/END/g' {}".format(mpcFile))
    for file_idx, file_ in enumerate(
            ["{}\nEND".format(fcont.strip()) for fcont in open(mpcFile, "r").read().split("END")[:-1]]):
        # GET MPC LAQ CONSTR LIST:
        try:
            pre_mpc = Chem.Mol(Chem.rdmolfiles.MolFromPDBBlock(file_, removeHs=False))
        except Exception as e:
            ### BBBBBBBBBBBBBAAAAAAAAAAAAd code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print("ERROR: {}".format(str(e)))
            lgd = prepLig('bff', extraLinkAt=0, term_o_h=False, ref=REF)
            lig_mol = rdpdb('mem/lig.pdb', sanitize=False)
            pre_mpc = AllChem.AssignBondOrdersFromTemplate(lgd.lig, lig_mol)
            print('OK!')
            # continue
        mpc_uncl_str = Chem.Mol(pre_mpc)
        print('mpc_uncl_str!!')
        mpc_uncl_str.RemoveAllConformers()
        if freeze == "all":
            pre_mpc2 = Chem.Mol(pre_mpc)
        elif freeze == "heavy":
            pre_mpc2 = Chem.RemoveHs(Chem.Mol(pre_mpc))
        else:
            pre_mpc2 = Chem.DeleteSubstructs(Chem.RemoveHs(pre_mpc), Chem.MolFromSmiles(lgd.lig_trunc_smiles))
        if db: display(pre_mpc, pre_mpc.GetNumAtoms())
        if db: display(pre_mpc2, pre_mpc2.GetNumAtoms())
        match = pre_mpc.GetSubstructMatch(pre_mpc2)
        algMap = [(j, i) for i, j in enumerate(match)]
        if db: note("match and algmap", vd(match), vd(algMap))
        ligCstConf = pre_mpc2.GetConformer(0)
        mpc_laq_constr_list = [[pre_mpc2.GetAtoms()[v2].GetSymbol(),
                                tuple([ligCstConf.GetAtomPosition(i) for i in range(ligCstConf.GetNumAtoms())][v2])]
                               for
                               k, v2 in algMap]
        # vd(mpc_laq_constr_list)
        if db:  # VIEW ALL FREEZED ATOMS:
            reprMol(mpc_laq_constr_list)
            pm("load {}, pre-M-opt ".format(mpcFile), "load .tmp/pmSph.pse", srv_md=False)
        # pdbBlock (file_) -> .mop file
        pybel.readstring('pdb', file_).write(format='mop', filename='cg/mp/mpc_uncl_str_{}.mop'.format(file_idx),
                                             opt={
                                                 'k': 'PM6 CHARGE={} PRECISE pKa EF THREADS=14'.format(charge)},
                                             overwrite=True)
        # .mop file -> .mop file with freezed atom positions
        vd(mpc_laq_constr_list)
        if db2: dpf("cg/mp/mpc_uncl_str_{}.mop".format(file_idx))
        mpc_frz('cg/mp/mpc_uncl_str_{}.mop'.format(file_idx), mpc_laq_constr_list)
        if db2: dpf("cg/mp/mpc_uncl_str_{}.mop".format(file_idx))
        if not preLom2:
            # if file_idx == 0:
            #     continue
            if not afterLom2:
                my_env = os.environ.copy()
                cmd_ = "/home/domain/data/prog/MOPAC2016_2019.exe cg/mp/mpc_uncl_str_{}.mop".format(file_idx)

                run(cmd_)
                # os.system(cmd_)
            try:
                wrter = list(pybel.readfile('mopout', 'cg/mp/mpc_uncl_str_{}.out'.format(file_idx)))[0]
            except IOError:
                print("NO SUCH MOP OUT FILE: {}. skipping...".format('cg/mp/mpc_uncl_str_{}.out'.format(file_idx)))
                continue
            # VISU BEGIN
            open("cg/cl_str/cl_str_{}.pdb".format(file_idx), "w").write(file_)
            wrter.write(format='pdb', filename='cg/mp/mpc_uncl_str_{}.pdb'.format(file_idx),
                        overwrite=True)
            # VISU END
            wrter.write(format='sdf', filename='cg/mp/mpc_uncl_str_{}.sdf'.format(file_idx),
                        overwrite=True)  # for rdkit (filtration. rdkit can't parse mol2 files correctly)
            wrter.write(format='mol2', filename='cg/mp/mpc_uncl_str_{}.mol2'.format(file_idx),
                        overwrite=True)  # for rosetta
            if db2: dpf(
                *["cg/cl_str/cl_str_{}.pdb".format(file_idx)] + ["cg/mp/mpc_uncl_str_{}.{}".format(file_idx, j4) for
                                                                 j4
                                                                 in ["out", "pdb", "sdf", "mol2"]])
            mpMol = Chem.Mol(
                Chem.rdmolfiles.SDMolSupplier('cg/mp/mpc_uncl_str_{}.sdf'.format(file_idx), removeHs=False,
                                              sanitize=False)[
                    0])  # ONLY GEOMETRY matters -> sanitization False
            mpc_uncl_str.AddConformer(mpMol.GetConformer(), assignId=True)
            if db2:
                return mpMol, pre_mpc
                loadCnf2Pml('cg_mp_%s' % file_idx, mpMol, srv_md=False, r=False)
                loadCnf2Pml('pre_mpc_%s' % file_idx, pre_mpc, srv_md=False, r=False)
            mpc_uncl_supl.append(tuple([file_idx, slice_file("FINAL HEAT OF FORMATION", "COMPUTATION TIME",
                                                             'cg/mp/mpc_uncl_str_{}.out'.format(file_idx)),
                                        file_idx]))
        else:
            lom2MopList.append('cg/mp/mpc_uncl_str_{}.mop'.format(file_idx))
        print("{}: OK".format(file_idx))
        # break
    # WRITING TO FILES
    wrcf(mpc_uncl_str, "cg/mpc_uncl_str.sdf", mpc_uncl_supl, "cg/mpc_uncl_supl")
    slg('mopOpt finished. Files mpc_uncl_str.sdf and mpc_uncl_supl generated ')
    if preLom2:
        return lom2MopList
def mopOptTest(inpSmiles=None, testMolFile=None, name=""):
    tN = "{}{{}}".format(name)
    vd(tN)
    if not testMolFile:
        mol = Chem.MolFromSmiles(inpSmiles)
        AllChem.Compute2DCoords(mol)
    else:
        mol = rdpdb(testMolFile)
    m3d = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(m3d)
    AllChem.MMFFOptimizeMolecule(m3d, maxIters=500, nonBondedThresh=200)

    Chem.rdmolfiles.PDBWriter(tN.format(".pdb")).write(m3d)
    list(pybel.readfile('pdb', tN.format(".pdb")))[0].write(format='mop', filename=tN.format(".mop"),
                                                            opt={
                                                                'k': 'PM6 CHARGE=0 PRECISE EF THREADS=14'},
                                                            overwrite=True)

    my_env = os.environ.copy()
    my_env["MOPAC_LICENSE"] = '/home/domain/anur/progs/mopac/'
    my_env["LD_LIBRARY_PATH"] = '/opt/mopac'
    subprocess.Popen(
        "/home/domain/anur/progs/mopac/MOPAC2016.exe {}".format(tN.format(".mop")),
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
        env=my_env).communicate("\n")
    wrter = list(pybel.readfile('mopout', tN.format(".out")))[0]

    wrter.write(format='pdb', filename=tN.format("2.pdb"),
                overwrite=True)
    wrter.write(format='mol2', filename=tN.format(".mol2"),
                overwrite=True)  # for rosetta
    dpf(tN.format(".out"))
    dpf(tN.format("2.pdb"))
    dpf(tN.format(".mol2"))
    print("python /opt/acpype/acpype.py -i {} --charge_method user --net_charge 0 -a gaff".format(tN.format(".mol2")))
    print("python /opt/acpype/acpype.py -i {}   ".format(tN.format(".pdb")))
def prepNativeLig(db=False, rw=False):
    rwdir('cg')
    rwdir('cg/mp')
    amp_smi = "CCCCC(=O)O[P@@](=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1cnc2c1ncnc2N)[O-]"
    if ln == "rrf":
        flur_smi = "C1=CC2=C(C=C1O)OC3=CC(=O)C(=CC3=N2)"
    elif ln == "laq":
        flur_smi = "C1CSS[C@@H]1"
    lig_smi = flur_smi + amp_smi
    if rw:
        pm("fetch 4tvw", "remove not c. A", "fetch 3a7r", "super 4tvw, 3a7r",
           'select sulfur, (r. 37P and e. S)', 'select nitr, sulfur around 2 and e. N', 'alter nitr, name="O33"',
           'alter nitr, elem="O"',
           'alter sulfur, name="P"', 'alter sulfur, elem="P"', "save .tmp/test_laq.sdf, r. LAQ",
           "save .tmp/test_rrf.sdf, r. 37P")
    if db: dpf("/home/domain/data/rustam/dgfl/ed/ed_RD3/ri/.tmp/test_laq.pdb", ".tmp/test_rrf.sdf")
    if ln == "laq":
        lig_raw = Chem.rdmolfiles.SDMolSupplier(".tmp/test_laq.sdf").next()
        lig_raw = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(amp_smi), lig_raw)
    elif ln == "rrf":
        lig_raw = Chem.rdmolfiles.SDMolSupplier(".tmp/test_rrf.sdf").next()
        lig_raw = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(lig_smi), lig_raw)
    lig_rawH = Chem.AddHs(lig_raw)
    if db:
        v = MolViewer()
        v.DeleteAll()
        v.ShowMol(lig_raw, "lig_raw", showOnly=False)
        v.ShowMol(lig_rawH, "lig_rawH", showOnly=False)
        display(lig_raw, "lig_raw")
    ligH, _, _ = ConstrainedEmbed(lig_rawH, lig_raw, lg_mnz_cst=600)
    if db: v.ShowMol(ligH, "ligH", showOnly=False)
    Chem.rdmolfiles.PDBWriter("cg/ligH.pdb").write(ligH)
    list(pybel.readfile('pdb', "cg/ligH.pdb"))[0].write(format='mop', filename='cg/ligH.mop',
                                                        opt={'k': 'PM6 NOOPT CHARGE=-1'}, overwrite=True)
    if db: dpf("cg/ligH.mop")
    my_env = os.environ.copy()
    my_env["MOPAC_LICENSE"] = '/home/domain/anur/progs/mopac/'
    my_env["LD_LIBRARY_PATH"] = '/opt/mopac'
    subprocess.Popen("/home/domain/anur/progs/mopac/MOPAC2016.exe cg/ligH.mop", shell=True, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=my_env).communicate("\n")
    if db: dpf('cg/ligH.out')
    writer = list(pybel.readfile('mopout', 'cg/ligH.out'))[0]
    writer.write(format='mol2', filename='cg/mp/mpc_uncl_str_0.mol2', overwrite=True)  # for rosetta
    writer.write(format='pdb', filename='cg/mpc_uncl_str.pdb', overwrite=True)  # just for dbging (optional)
    open("cg/mpc_uncl_supl", "w").write(str(
        [(0, slice_file("FINAL HEAT OF FORMATION", "COMPUTATION TIME", 'cg/ligH.out'), 0)]))  # mimic for mopOpt output
    if db: dpf('cg/mp/mpc_uncl_str_0.mol2')
    if db: dpf('cg/mpc_uncl_str.pdb')
    print("!!!!!!!!!!!!!!!!!!!!VERIFY YOUR mpc_uncl_str.pdb structure. It must NOT BE FRAGMENTARY!!!!!!!!!!")
def prep4tvwLigs(db=False, rw=False, db2=True, afterRlx=True, ssPath="../ri/", ssIdxs=[1, 1, 1, 1]):
    """ MAKE cg/mpc_uncl_str_{0,1,2,3}.pdb files"""
    global mpc_flt_str, chainIdx
    rwdir('cg')
    rwdir('cg/mp')
    amp_smi = "CCCCC(=O)O[P@@](=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1cnc2c1ncnc2N)[O-]"
    flur_smi = "C1=CC2=C(C=C1O)OC3=CC(=O)C(=CC3=N2)"
    lig_smi = flur_smi + amp_smi
    if rw:  # we are not removing ligand in structure. We will do it in sStrGen
        if not afterRlx:
            pm("load ../ri/ss/lpla_lg2_0001.pdb", "fetch 4tvw, liuw", 'select sulfur, (r. 37P and e. S)',
               'select nitr, sulfur around 2 and e. N', 'alter nitr, name="O33"', 'alter nitr, elem="O"',
               'alter sulfur, name="P"', 'alter sulfur, elem="P"', db=False)
        for chainIdx, chain in enumerate(["A", "B", "C", "D"]):
            if afterRlx:
                ss = "{0}ss/liuw_rlx_{1}/lpla_lg_{1}_000{2}.pdb".format(ssPath, chainIdx, ssIdxs[chainIdx])
                pm("load {}, liuw_{}".format(ss, chainIdx),
                   "save .tmp/n r. RRF & liuw_{0}".format(chainIdx, ))
                # shutil.copy("ss/liuw_rlx_{0}/lpla_lg_{0}_0001.pdb".format(chainIdx),
                #             "ss/liuw_{0}.pdb".format(chainIdx))
            else:
                pm("extract liuw_{}, liuw & c. {}".format(chainIdx, chain),
                   "super liuw_{}, lpla_lg2_0001".format(chainIdx),
                   "alter r. 37P, resn='RRF'",
                   "save .tmp/4tvw_rrf_{0}.sdf, r. RRF & liuw_{0}".format(chainIdx, ),
                   "save ss/liuw_{0}.pdb, liuw_{0}".format(chainIdx), r=False)
        if db: dpf(".tmp/4tvw_rrf_{}.sdf".format(chainIdx))
    rwdir('cg/prms')
    for chainIdx, chain in enumerate(["A", "B", "C", "D"]):
        def tryMopOpt():
            vd(chainIdx)
            lig_raw = Chem.rdmolfiles.SDMolSupplier(".tmp/4tvw_rrf_{}.sdf".format(chainIdx)).next()
            lig_raw = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(lig_smi), lig_raw)
            lig_rawH = Chem.AddHs(lig_raw)
            if db:
                display(lig_raw, "lig_raw")
                display(lig_rawH, "lig_rawH")
                loadCnf2Pml("lig_raw", lig_raw, 0, r=False, srv_md=False)
                loadCnf2Pml("lig_rawH", lig_rawH, 0, r=False, srv_md=False)

            ligH, _, _ = ConstrainedEmbed(lig_rawH, lig_raw, lg_mnz_cst=600)
            if db: loadCnf2Pml("ligH", ligH, 0, r=False, srv_md=False)

            Chem.rdmolfiles.PDBWriter("cg/ligH.pdb").write(ligH)
            list(pybel.readfile('pdb', "cg/ligH.pdb"))[0].write(format='mop', filename='cg/ligH.mop',
                                                                opt={'k': 'PM6 NOOPT CHARGE=-1'}, overwrite=True)
            if db: dpf("cg/ligH.mop")
            my_env = os.environ.copy()
            my_env["MOPAC_LICENSE"] = '/home/domain/anur/progs/mopac/'
            my_env["LD_LIBRARY_PATH"] = '/opt/mopac'
            subprocess.Popen("/home/domain/anur/progs/mopac/MOPAC2016.exe cg/ligH.mop", shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=my_env).communicate("\n")
            if db: dpf('cg/ligH.out')
            writer = list(pybel.readfile('mopout', 'cg/ligH.out'))[0]
            writer.write(format='mol2', filename='cg/mp/mpc_uncl_str_{}.mol2'.format(chainIdx),
                         overwrite=True)  # for rosetta
            writer.write(format='pdb', filename='cg/mpc_uncl_str_{}.pdb'.format(chainIdx),
                         overwrite=True)  # just for dbging (optional)
            if db: dpf('cg/mp/mpc_uncl_str_{}.mol2'.format(chainIdx))
            if db: dpf('cg/mpc_uncl_str_{}.pdb'.format(chainIdx))
            p = subprocess.Popen(
                "python2 /home/domain/data/prog/rosetta_2017/main/source/scripts/python/public/molfile_to_params.py -n RRF -p RRF --clobber mpc_uncl_str_{}.mol2 ".format(
                    chainIdx),
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="cg/mp/", executable='/bin/bash')
            so, ser = p.communicate()
            note("so, ser", vd(so), vd(ser))
            acGran = not ser
            return acGran

        acGran = False
        while acGran != True:
            acGran = tryMopOpt()
        note("Finished processing {}".format(chain))
    if db2:
        pmr(*['load cg/mpc_uncl_str_{}.pdb'.format(i) for i in range(4)])
class prepLig:
    def __init__(self, ln=ln, db=False, ref="/home/domain/data/rustam/dgfl/ed/ed_RD3/ri/.tmp/test_laq.pdb",
                 extraLinkAt=0,
                 term_o_h=True):
        self.db = db
        refName = "laq" if ref == "/home/domain/data/rustam/dgfl/ed/ed_RD3/ri/.tmp/test_laq.pdb" else "rrf"
        vd(refName)
        self.amp_smi = "CCCCC(=O)O[P@@](=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1cnc2c1ncnc2N)[O-]"
        # self.amp_smi2 = "CCCCC(=O)OP(=O)(OCC1OC(C(C1O)O)n1cnc2c1ncnc2N)[O-]"
        bff = ""
        if ln == "rrf":
            self.flur_smi = "C1=CC2=C(C=C1O)OC3=CC(=O)C(=CC3=N2)"
        elif ln == "bff":
            extraLinkAt = "C" * extraLinkAt
            bff = "[B-]1(F)(F)C2=CC(O)=CC=C2C=C3[N+]1=C(C)N(CCC{}CC(=O)O[P@@](=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1cnc2c1ncnc2N)[O-])C3(=O)".format(
                extraLinkAt)
            self.flur_smi = "[B-]1(F)(F)C2=CC(O)=CC=C2C=C3[N+]1=C(C)NC3(=O)"
        elif ln == "laq":
            self.flur_smi = "C1CSS[C@@H]1"
        if bff:
            self.lig_smi = bff
        else:
            self.lig_smi = self.flur_smi + self.amp_smi
        if not term_o_h and ln == "bff":
            self.lig_smi = self.lig_smi.replace("C2=CC(O)", "C2=CC([O-])")
        if not term_o_h and ln == "rrf":
            self.lig_smi = self.lig_smi.replace("(C=C1O)", "(C=C1[O-])")

        # pm("fetch 3a7r", "remove not r. LAQ", "save .tmp/test_laq.pdb", db=False)
        if db: dpf("/home/domain/data/rustam/dgfl/ed/ed_RD3/ri/.tmp/test_laq.pdb")
        self.amp_r = Chem.MolFromSmiles(self.amp_smi)
        if db: display('self.amp_r: maximal-constrained amp part from smiles', self.amp_r)

        # pdb_remove_duplicates(ref) ### ATTENTION: commented because we run only ag.py
        if refName == "rrf":
            while True:
                try:
                    laq_raw = next(Chem.rdmolfiles.SDMolSupplier(ref))
                    break
                except Exception:
                    print(' End of the supplier hit! ')
                    sl(1)

        elif refName == "laq":
            laq_raw = Chem.rdmolfiles.MolFromPDBFile("/home/domain/data/rustam/dgfl/ed/ed_RD3/ri/.tmp/test_laq.pdb")
        laq_raw = AllChem.AssignBondOrdersFromTemplate(self.amp_r, laq_raw)
        if db: display(
            'laq_raw: maximal-constrained amp part with 3A7R-like coords (raw .pdb file from RCSB:PDB) + assigned bond-orders from smiles (aromatic conjugate)',
            laq_raw)
        linker_list = ['C', 'C', 'C', 'C', 'C(=O)', 'O']
        if ln == "bff":
            bffLinkSmi = "" if NumLinkAt == 0 else '({})'.format("".join(linker_list[:NumLinkAt]))
            self.lig_trunc_smiles = self.flur_smi.replace("NC3(=O)", "N{}C3(=O)".format(bffLinkSmi))
        else:
            self.lig_trunc_smiles = self.flur_smi.replace("[C@@H]", "C") + "".join(linker_list[:NumLinkAt])
        fluroSmiles = 'C1CSSC1' if refName == "laq" else "C1=CC2=C(C=C1O)OC3=CC(=O)C(=CC3=N2)"
        self.laq_trunc_smiles = '{}{}'.format(fluroSmiles, "".join(linker_list[:NumLinkAt]))

        if db: display("You will delete from constrainable part of ligand this moiety", self.laq_trunc_smiles)
        self.laq_core = Chem.DeleteSubstructs(laq_raw, Chem.MolFromSmiles(self.laq_trunc_smiles))
        #
        if db:
            display('self.laq_core: Moiety, constrained in your ligand', self.laq_core)
            display('laq_trunc_similes:', self.laq_trunc_smiles)
        self.lig = Chem.MolFromSmiles(
            self.lig_smi)  # rdkit molecule for ConstrainedEmbed procedure (it will remove all conformers for convinience)
        self.fluro_core = Chem.DeleteSubstructs(self.lig, Chem.DeleteSubstructs(laq_raw, Chem.MolFromSmiles('C1CSSC1')))
        self.lig = Chem.AddHs(self.lig)
        if db:
            display('Your ligand:zoo', self.lig)
def confGen(NodeList=[], ln=ln, NumConf=10, max_ntries=1000000000, db=False,
            ref="/home/domain/data/rustam/dgfl/ed/ed_RD3/ri/.tmp/test_laq.pdb", doMap=False, cst1d=False):
    if isinstance(NodeList, str):
        NodeNumber = NodeList
    else:
        rndIdxs, NodeNumber, cst1d = NodeList
    cgDir = 'cg{}'.format(NodeNumber)
    rwdir(cgDir)
    slg('confGen started')
    t0 = timeit.default_timer()
    print(ln, ref)
    lgd = prepLig(ln=ln, ref=ref)
    # By default the force field used for the structure refinment is UFF. This can be changed to MMFF:
    GetFF = lambda x, confId=-1: AllChem.MMFFGetMoleculeForceField(x, AllChem.MMFFGetMoleculeProperties(x),
                                                                   confId=confId)
    # MMFF unable to handle BFF, so lets use UFF
    # display lgd.laq_core in pymol:
    if db:
        loadCnf2Pml("laq_core", lgd.laq_core, srv_md=False)
    lig_mol = Chem.Mol(lgd.lig)
    lig_mol.RemoveAllConformers()
    confs = []
    lig_mol_uncl = Chem.Mol(lgd.lig)
    lig_mol_uncl.RemoveAllConformers()
    confs_uncl = []
    ntries = 0
    cg_all, cg_acc = 0, 0
    pst_rq(proc='cg_clear')
    lenConfs = 0
    while len(confs_uncl) < NumConf:
        ntries += 1
        cg_all += 1
        tm_try = timeit.default_timer() - t0
        if NodeNumber:
            randomseed = rndIdxs[NodeNumber][ntries]
        else:
            randomseed = -1
        raw_conf, en, cst_rms = ConstrainedEmbed(Chem.Mol(lgd.lig), lgd.laq_core, randomseed=randomseed, db=db,
                                                 doMap=doMap, cst1d=cst1d, ln=ln)
        # if cst_rms > 0.04:
        if cst_rms > 0.6:
            continue
        # cf_idx = lig_mol.AddConformer(raw_conf.GetConformer(0),
        #                               assignId=True)  # Adding minimized conformer to lig_mol
        cf_idx_uncl = lig_mol_uncl.AddConformer(raw_conf.GetConformer(0),
                                                assignId=True)  # Adding minimized conformer to lig_mol
        # pst_rq(proc='cgp', data={'x': cg_all, 'y': cg_acc})

        # print(cg_all, cg_acc)
        # confs.append((cf_idx, en, cst_rms))
        confs_uncl.append((cf_idx_uncl, en))
        # print(len(confs_uncl))
        # note((cf_idx_uncl, en,
        # cst_rms))
        cg_acc += 1

        if db or cg_acc % 30 == 0:
            try:
                lig_mol_uncl2, confs_uncl2 = ecleaning(lig_mol_uncl, confs_uncl, 50)
                print(NodeNumber, ntries, len(confs_uncl), lig_mol_uncl.GetNumConformers())
                lig_mol_uncl, confs_uncl = postrmsd(lig_mol_uncl2, confs_uncl2, 0.1)
                rsl = NodeNumber, ntries, len(confs_uncl), lig_mol_uncl.GetNumConformers()
                print(rsl)
                open("cg{}/rslt.txt".format(NodeNumber), "a").write("{}\n".format(str(rsl)))
            except Exception:
                pass
        if db or cg_acc % 30 == 0:
            # dump of
            # backup
            if "lig_mol" in locals() and "confs" in locals():
                if "lig_mol_uncl" in locals() and "confs_uncl" in locals():
                    wrcf(lig_mol_uncl, "{}/uncl_str.pdb".format(cgDir), confs_uncl, "{}/uncl_supl".format(cgDir))
                    # wrcf(lig_mol, "{}/cl_str.pdb".format(cgDir), confs, "{}/cl_supl".format(cgDir))
                    # wrcf(lig_mol_uncl, "{}/uncl_str_tmp.pdb".format(cgDir), confs_uncl,
                    #      "{}/uncl_supl_tmp".format(cgDir))
                    # # wrcf(lig_mol_uncl, "{}/uncl_str.pdb".format(cgDir), confs_uncl, "{}/uncl_supl".format(cgDir))
                    # lig_mol_uncl.RemoveAllConformers()
                    # os.system("cat {0}/uncl_str_tmp.pdb >> {0}/uncl_str.pdb ".format(cgDir))
            if ntries > max_ntries:
                break
    if db: note("generated {} conformers".format(len(confs_uncl)))
    # slg("generated {} conformers".format(len(confs)))
    if db:
        dpf("{}/cl_supl".format(cgDir), "{}/uncl_supl".format(cgDir), "{}/cl_str.pdb".format(cgDir),
            "{}/uncl_str.pdb".format(cgDir))
    #
    # slg('ConfGen finished. Files cl and uncl generated ')
    return lig_mol_uncl, confs_uncl
def wrcf(lig_mol, cf_str_path, confs=None, cf_supl_path=None, db=False):
    if os.path.splitext(cf_str_path)[1] == '.pdb':
        wr = Chem.rdmolfiles.PDBWriter(cf_str_path)
    if os.path.splitext(cf_str_path)[1] == '.sdf':
        wr = Chem.rdmolfiles.SDWriter(cf_str_path)
    for cid in range(lig_mol.GetNumConformers()):
        wr.write(lig_mol, confId=cid)
    wr.close()
    if confs:
        open(cf_supl_path, "w").write(str(confs))
    if db: dpf(cf_str_path, cf_supl_path)
def wrpdb(lig_mol, cf_str_path, vbs=True):
    wr = Chem.rdmolfiles.PDBWriter(cf_str_path)
    wr.write(lig_mol, confId=-1)
    if vbs: print("Wrote {}".format(cf_str_path))
    wr.close()
def rdpdb(cf_str_path, vbs=True, sanitize=True):
    os.popen("sed -i 's/^END$/ENDMDL/g' {}".format(cf_str_path))
    molFromPdb = Chem.rdmolfiles.MolFromPDBFile(cf_str_path, removeHs=False, sanitize=sanitize)
    if vbs: print("Read {} cnfs from {}".format(molFromPdb.GetNumConformers(), cf_str_path))
    return molFromPdb

@prntFun
def gr_fltr(mol=None, prot=None, confs=None, inp="", gr_fltr_vdw=0.4, db=False, rprName="", leaveLiuCatSc=False):
    t = time.time()
    if rprName and os.path.exists(rprName): os.remove(rprName)
    if inp:
        mol = rdpdb(inp)
    if not prot:
        prot = ssGen(leaveLiuCatSc=leaveLiuCatSc, rw=False)
    mnz_confs = []
    mnz_en_mol, clash_mol, ok_mol = Chem.Mol(mol), Chem.Mol(mol), Chem.Mol(mol)
    mnz_en_mol.RemoveAllConformers()
    clash_mol.RemoveAllConformers()
    ok_mol.RemoveAllConformers()
    if rprName: loadCnf2Pml("prot_bb", prot, 0, r=True, srv_md=True)
    for k in range(mol.GetNumConformers()):
        ptr_dist = rdkit.Chem.rdShapeHelpers.ShapeProtrudeDist(mol, prot, k, 0, vdwScale=gr_fltr_vdw,
                                                               gridSpacing=0.00001, stepSize=0.0)
        # print(prot.GetNumAtoms(), deHydratedLigMol.GetNumAtoms())
        if ptr_dist == 1:
            new_id = mnz_en_mol.AddConformer(mol.GetConformer(k), assignId=True)
            if confs:
                mnz_en = [(i[1]) for i in confs if i[0] == k][0]
                mnz_confs.append((new_id, mnz_en))
            if rprName:
                ok_mol.AddConformer(mol.GetConformer(k), assignId=True)
                # loadCnf2Pml("ok_%s" % k, mol, k)
        elif rprName:
            clash_mol.AddConformer(mol.GetConformer(k), assignId=True)
            # loadCnf2Pml("clash_%s" % k, mol, k)
        if db: note("protrude dist between 3a7r and {}".format(k), ptr_dist)
    if rprName:
        loadCnf2Pml("clash", clash_mol, -1, srv_md=True)
        loadCnf2Pml("ok_mol", ok_mol, -1, srv_md=True)
        pm("load ss/lpla.pdb", "util.cbay clash*", "util.cbag ok*", "split_states ok_mol", "split_states clash",
           "set pse_export_version, 1.721",
           "select fltAtoms, i. {} or ( n. CA | n. O | n. C | n. N ) w. 10 of org".format(
               "+".join([str(i) for i in liuCatRes])), "remove e. h", "show spheres, fltAtoms | org",
           "alter *, vdw=1.0", "rebuild", "deselect",
           "save {}".format(rprName), r=False, srv_md=True, db=False)
    print("gr_fltr: {}->{}. Time: ".format(mol.GetNumConformers(), mnz_en_mol.GetNumConformers(), time.time() - t))
    if confs:
        return mnz_en_mol, mnz_confs
    else:
        return mnz_en_mol
def loadCnf2Pml(pymName="", ligMol=None, confId=-1, r=False, srv_md=True, inp="", otp=""):
    d__tdir = P('/tmp/')
    # if inp:
    #     if not ".pdb" in inp: raise ("not pdb inputs are temporarily not supported")
    #     vd(inp)
    #     ligMol = rdpdb(inp)
    #     if not pymName:
    #         pymName = os.path.basename(inp).replace(".pdb", "")
    Chem.rdmolfiles.PDBWriter(tdir[pymName].s).write(ligMol, confId=confId)
    cmd_ = f"load {tdir[pymName].s}"
    print(cmd_)
    pm(cmd_, r=r, srv_md=srv_md)

def plane_flt(uncl_str2, confs=[], planeCoords=[], db=False, prep=False, protPdb="ss/lpla.pdb", db2=True):
    # run "pl.py" in your pymol and click the atoms
    mnz_confs = []
    if prep:
        loadCnf2Pml("test", uncl_str2, 1, r=False, srv_md=False)
        pm("split_states test", "load {}".format(protPdb), "run pl.py", srv_md=False, r=False)
        return
    planEqs = []
    for planeCoord in planeCoords:
        [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), orientation] = planeCoord
        vector1 = [x2 - x1, y2 - y1, z2 - z1]
        vector2 = [x3 - x1, y3 - y1, z3 - z1]
        cross_product = [vector1[1] * vector2[2] - vector1[2] * vector2[1],
                         -1 * (vector1[0] * vector2[2] - vector1[2] * vector2[0]),
                         vector1[0] * vector2[1] - vector1[1] * vector2[0]]
        a = cross_product[0]
        b = cross_product[1]
        c = cross_product[2]
        d = - (cross_product[0] * x1 + cross_product[1] * y1 + cross_product[2] * z1)
        orientation = 1 if orientation else -1
        planEqs.append((a, b, c, d, orientation))
    # Checking the oxygen atom number:
    if db2: print("plane filtering will be on atom of element: '{}'".format(uncl_str2.GetAtomWithIdx(6).GetSymbol()))

    uncl_str3 = Chem.Mol(uncl_str2)
    uncl_str3.RemoveAllConformers()
    if db:
        print("Reinitialize your PYMOL if needed")
        badConfs = Chem.Mol(uncl_str2)
        badConfs.RemoveAllConformers()
    for j in range(uncl_str2.GetNumConformers()):
        cnf = uncl_str2.GetConformer(j)
        oxy = cnf.GetAtomPosition(6)
        # print (a * oxy.x + b * oxy.y + c * oxy.z + d)
        # vd(orientation)
        if not any([(a * oxy.x + b * oxy.y + c * oxy.z + d) * orientation > 0 for a, b, c, d, orientation in planEqs]):
            ncidx = uncl_str3.AddConformer(cnf, assignId=True)
            mnz_en = [i[1] for i in confs if i[0] == j][0]
            mnz_confs.append((ncidx, mnz_en))

        elif db:
            badConfs.AddConformer(cnf, assignId=True)
    # VISU:
    if db:
        loadCnf2Pml("good", uncl_str3, -1, srv_md=False, r=False)
        loadCnf2Pml("bad", badConfs, -1, srv_md=False, r=False)
        pm("split_states *", "util.cbag good*", "util.cbac bad", "load {}".format(protPdb), "run pl.py", srv_md=False,
           r=False)
    if db2:
        print("PLANE FILTRATION: {} -> {}".format(uncl_str2.GetNumConformers(), uncl_str3.GetNumConformers()))
        say("plane filtering complete")
    return uncl_str3, mnz_confs
def gnr_rlx_flgs(db=False, liuw=False, rlx=True):
    global rlxFlags
    if liuw:
        for chainIdx, chain in enumerate(["A", 'B', 'C', 'D']):
            if chainIdx != 0:
                break
            rwdir('ss/liuw_rlx_{}'.format(chainIdx))
            rlxFlags = {}
            rlxFlags['-extra_res_fa'] = './ri/cg/prms/prms_{}/RRF.params'.format(chainIdx)
            rlxFlags['-nstruct'] = '100'
            # change to 10
            rlxFlags['-relax:default_repeats'] = '10'
            rlxFlags['-out:path:pdb'] = './ri/ss/liuw_rlx_{}/'.format(chainIdx)
            # change to true:
            rlxFlags['-relax:bb_move'] = 'true'
            rlxFlags['-relax:constrain_relax_to_start_coords'] = True
            rlxFlags['-relax:coord_constrain_sidechains'] = True
            rlxFlags['-relax:ramp_constraints'] = 'false'
            rlxFlags['-packing:ex1'] = True
            rlxFlags['-packing:ex2'] = True
            rlxFlags['-packing:use_input_sc'] = True
            rlxFlags['-packing:flip_HNQ'] = True
            rlxFlags['-packing:no_optH'] = 'false'
            rlxFlags['-respect_resfile'] = True
            # rlxFlags['-constraints'] = "ss/cst.cst"
            rlxFlags['-s'] = './ri/ss/ss_batch/lpla_lg_{}.pdb'.format(chainIdx)
            flags = ""
            for k4 in sorted(rlxFlags.iterkeys(), reverse=True):
                if rlxFlags[k4] == True:
                    flags += "{}\n".format(k4)
                elif rlxFlags[k4]:
                    flags += "{} {}\n".format(k4, rlxFlags[k4])
            open("msc/rlx_{}.flags".format(chainIdx), "w").write(flags)
            if db: print("RLX_{} FLAGS:\n{}".format(chainIdx, flags))

    else:
        flags = ""
        for k4 in sorted(rlxFlags.keys(), reverse=True):
            if rlxFlags[k4] == True:
                flags += "{}\n".format(k4)
            elif rlxFlags[k4]:
                flags += "{} {}\n".format(k4, rlxFlags[k4])
        open("msc/rlx.flags", "w").write(flags)
        if db: print("RLX FLAGS:\n{}".format(flags))
def gnr_dsn_flgs(db=False, file_name='msc/ed.flags'):
    # edFlags = rdv('edFlags')
    flags = ""
    for k4 in sorted(edFlags.keys(), reverse=True):
        if edFlags[k4] == True:
            flags += "{}\n".format(k4)
        elif edFlags[k4]:
            flags += "{} {}\n".format(k4, edFlags[k4])
    open(file_name, "w").write(flags)
    if db: print("ED FLAGS:\n{}".format(flags))
def pma(*args):
    pm(*args, r=False, srv_md=False, db=0)
def pmr(*args):
    pm(*args, r=True, srv_md=False, db=0)
def pm(*args, r=True, srv_md=srv_md, db=0):
    import re
    pm_eng = "cmd" if srv_md else "pml"
    if r:
        if srv_md:
            exec("cmd.sync(); {0}.reinitialize(); cmd.sync();".format(pm_eng))
        else:
            exec("{0}.reinitialize()".format(pm_eng))
    for cmnd in args:
        # if pm_eng == "pml": cmnd = re.sub(r'(save | load)(.*)', "\g<1>{}".format(psl('\g<2>')), cmnd)
        if pm_eng == "pml":
            # print('AAAA', cmnd)
            if "/mnt/storage/" in cmnd:
                cmnd = re.sub(r'(save |load )/mnt/storage/rustam/(.*?)(,.*)', "\g<1>/r/\g<2>\g<3>", cmnd)
            else:
                cmnd = re.sub(r'(save |load )/home/domain/data/rustam/(.*?)(,.*)', "\g<1>/r/\g<2>\g<3>", cmnd)
        cmdl = ["{}.sync()".format(pm_eng), "{}.do(\"\"\" {} \"\"\")".format(pm_eng, cmnd), "{}.sync()".format(pm_eng)]
        for cm in cmdl:

            if re.search(r'fetch|save', cmnd):
                # io is very slow via sshfs!!!! maybe on server it is slow too
                sl(1)
                if re.search(r'save', cm):
                    exec("{0}.do('set pse_export_version, 1.721')".format(pm_eng))
                    loadPath = re.search(r'(?<=save ).*', cmnd).group()
                    if not srv_md:
                        rpath = os.path.realpath(loadPath)
                        print("realpath: {}".format(rpath))
                        # if not os.path.exists(os.path.dirname(pls(rpath))) and raw_input("make dir '{}'?".format(os.path.dirname(pls(rpath)))) == "Y":
                        #     rwdir(os.path.dirname(pls(rpath)))

                        cm = cm.replace(loadPath, rpath)
                    sl(1)
                    print(cm)
            if re.search(r'load_traj ', cm):
                loadPath = re.search(r'(?<=load_traj ).*', cmnd).group()
                if not srv_md:
                    cm = cm.replace(loadPath, psl(loadPath))
                sl(1)
            if re.search(r'load ', cm):
                loadPath = re.search(r'(?<=load ).*', cmnd).group()
                if not srv_md:
                    cm = cm.replace(loadPath, psl(loadPath))
                sl(1)
            exec(cm)
            if db:
                note("executing command {}".format(cm))
                # if cm.strip().startswith("cmd.do"):
                #     try:
                #         cm = re.search("(?<=cmd.do\(\"\"\" ).*(?= \"\"\"\))", cm).group()
                #     except Exception:
                #         vd(cm)
                #         pass

                sl(1)
def edv():
    display(HTML("<iframe width=2000 height=2000 src='https://serikov.ru.com'>"))
def pst_rq(headers={}, data={}, url='http://www.serikov.ru.com:3002', proc='cgp', db=False):
    if not headers:  # i.e. scheme-choose mode
        headers = {'ed_id': ed_id, 'proc': proc}
    try:
        with requests.Session() as session:

            response = session.post(url, headers=headers, data=data)
            if db: print(response.text)
            if response: return 'OK'
    except Exception as e:
        print("not sent, skipping %s" % str(e))
        pass
def slg(log_info):
    print(log_info)
    scs = pst_rq(proc='log', data={"log": log_info})
    if not scs:
        print("log info '{}' not sent to server".format(log_info))
def rwdir(*args):
    for dir_name in args:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)
def mkDir(*args):
    for dir_name in args:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
def tupleConfs(lig_mol, mpc=False):
    if mpc:
        return [(0, 0, i) for i in range(lig_mol.GetNumConformers())]
    else:
        return [(i, 0, 0) for i in range(lig_mol.GetNumConformers())]
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]
def columnize(some_list, n=4):
    co2 = chunkify(some_list, n)
    co2 = [*zip(*co2)]
    out_str = ''
    for row in co2:
        row = [str(i) for i in row]
        out_str += "\n{: <30} {: <30} {: <30}".format(*row)

    return (out_str)
def sliceLig(lig_mol, numCnf, rndConfs=False):
    uncl_str = Chem.Mol(lig_mol)
    uncl_str.RemoveAllConformers()
    if rndConfs:
        for mpCl in [random.randint(1, lig_mol.GetNumConformers()) for _ in range(numCnf)]:
            uncl_str.AddConformer(lig_mol.GetConformer(mpCl), assignId=True)
    else:
        for i in range(uncl_str.GetNumConformers() - numCnf):
            uncl_str.RemoveConformer(uncl_str.GetNumConformers() - 1)
    return uncl_str
def prmGen(mpc_flt_str, confs=None, db=True, noMpc=False, skipZero=False, confsInOneFile=True, noConfCHI=True,
           noConfProtonCHI=True):
    co = 1
    confs = confs if confs else tupleConfs(mpc_flt_str, mpc=True)
    # dirs = [re.search(r'(?<=_)\d+(?=\.mol2)', Dir).group() for Dir in glob.glob("cg/mp/mpc_uncl_str_*.mol2")]
    # print(dirs)
    if confsInOneFile:
        if len(confs) == 1:
            sp("cat mp/mpc_uncl_str_%s.mol2 > mpc_uncl_str_batch.mol2" % ",".join([str(i[-1]) for i in confs]),
               cwd="cg")
        else:
            sp("cat mp/mpc_uncl_str_{%s}.mol2 > mpc_uncl_str_batch.mol2" % ",".join([str(i[-1]) for i in confs]),
               cwd="cg")
        sp(
            "python2 /home/domain/data/prog/rosetta_2017/main/source/scripts/python/public/molfile_to_params.py -n {0} -p {0} --conformers-in-one-file --clobber {1}".format(
                ln.upper(), "mpc_uncl_str_batch.mol2"), cwd="cg")
        pmwd = "cg/"
        m2file = "mpc_uncl_str_batch.mol2"
    else:
        # MPI!!!
        rwdir('cg/prms')
        for fltNum in [i[-1] for i in confs]:
            print(fltNum)
            if skipZero and fltNum == 0:
                continue
            pmwd = 'cg/prms/prms_{}/'.format(fltNum)
            print(pmwd)
            m2file = 'mpc_uncl_str_{}.mol2'.format(fltNum)
            rwdir(pmwd)
            # os.mkdir(pmwd)
            shutil.copy('cg/mp/{}'.format(m2file), pmwd)
            dbgStr = "" if db else "2>&1 >/dev/null"
            sp(
                "python2 /home/domain/data/prog/rosetta_2017/main/source/scripts/python/public/molfile_to_params.py -n {0} -p {0} --clobber {1} {2}".format(
                    ln.upper(), m2file, dbgStr), cwd=pmwd)
            if db: dpf('{}{}'.format(pmwd, m2file), '{}{}_0001.pdb'.format(pmwd, ln.upper()),
                       '{}{}.params'.format(pmwd, ln.upper()))
    if noConfProtonCHI: os.system("sed  -ip '/PROTON_CHI/ d' {}{}.params".format(pmwd, ln.upper()))
    if noConfCHI: os.system("sed  -ip '/CHI/ d' {}{}.params".format(pmwd, ln.upper()))
    if db: dpf('{}{}'.format(pmwd, m2file), '{}{}_0001.pdb'.format(pmwd, ln.upper()),
               '{}{}.params'.format(pmwd, ln.upper()))
    if db: dpf("cg/mpc_uncl_str_batch.mol2")
def dirSStrGen(chunk):
    for sStrId in chunk:
        pmwd = 'cg/prms/prms_{}/'.format(sStrId)
        print(pmwd)
        pm("remove organic", "load {0}{1}_0001.pdb, {1}_{2}".format(pmwd, ln.upper(), sStrId),
           "save ss/ss_batch/lpla_lg_{}.pdb".format(sStrId), r=False)
    return

@prntFun
def sStrGen(db=False, confsInOneFile=confsInOneFile, liuw=False, rw=False, afterRlx=False, ssPath="../ri/",
            ssIdxs=[1, 1, 1, 1]):
    cstRes = [83, 184, 140]
    if db: print("ss/lpla_lg2_0001.pdb is a 3a7r structure, relaxed with native ligand")
    if not confsInOneFile and not liuw:
        rwdir('ss/ss_batch')
        pm("load ss/lpla_lg2_0001.pdb")
        # MPI!!! and correct it not to use variable sSTRQUan and BE MORE ROBUST
        pool = multiprocessing.Pool()
        dirs = [re.search(r'(?<=_)\d+(?=/)', Dir).group() for Dir in glob.glob("cg/prms/prms_*/")]
        dirs = chunkify(dirs, 8) if len(dirs) > 8 else chunkify(dirs, 1)
        print(dirs)
        pool.map(dirSStrGen, dirs)
        # REWRITE LINE BELOW!!!
        cmdn = "pymol -cq ss/cst.py -- {} {} {} {} {}".format(ln.upper(), "_".join([str(i) for i in cstRes]), Mg2sStr,
                                                              len(dirs), db)
    elif not liuw:
        #         !!!! not ready code
        pm("load ss/lpla_lg2_0001.pdb", "remove r. LAQ", "load cg/{}.pdb".format(ln.upper()),
           "save {}".format("ss/lpla_lg.pdb"))
        sStrQuan = 1
        print("ASDFASDFASFD")
        cmdn = "pymol -cq ss/cst.py -- {} {} {} {} {}".format(ln.upper(), "_".join([str(i) for i in cstRes]), Mg2sStr,
                                                              1, db)
    elif liuw:
        if rw:
            rwdir('ss/ss_batch/')
            for chainIdx, chain in enumerate(["A", "B", "C", "D"]):
                if chainIdx == 1:
                    break
                ss = "{0}ss/liuw_rlx_{1}/lpla_lg_{1}_000{2}.pdb".format(ssPath, chainIdx, ssIdxs[chainIdx])
                sStrLoadCmd = "load {}, liuw_{}".format(ss,
                                                        chainIdx) if afterRlx else "load ss/liuw_{}.pdb, prot".format(
                    chainIdx)
                # vd(sStrLoadCmd )
                pm(sStrLoadCmd, "alter polymer, chain='A'", "remove org",
                   "load cg/prms/prms_{}/RRF_0001.pdb".format(chainIdx),
                   "save ss/ss_batch/lpla_lg_{}.pdb".format(chainIdx), db=True)
def pat2(ar, pat=(), occ=0, i_s=0, r_s=0):
    from pandas import to_numeric
    df = ar
    for k, v in {"occ": u' Occurance', "i_s": "{}-amp ifE".format(ln.lower()), "r_s": "TotScore"}.items():
        if eval(k):
            if k == "occ":
                df = df.loc[to_numeric(df[v], errors='coerce') > eval(k)]
            else:
                df = df.loc[to_numeric(df[v], errors='coerce') < eval(k)]
    if pat:
        print("pat!")
        if pat[-1] == 1:
            strict = True
        elif pat[-1] == 0:
            strict = False
        else:
            print("Enter correct pattern: 0 = non strict, 1 = strict" + "!" * 20)
            raise ()
        pat = pat[0]
        vd(pat)
        if isinstance(pat[0], dict):
            print("pat: dict")
            for k4, j4 in pat[0].items():
                df = df[df[list([k4])].apply(lambda x: all([j4 in i for i in x]), axis=1)]
        elif isinstance(pat[0], int):
            print("pat: int")
            df = df[df[list(pat)].apply(lambda x: all(["+" in i for i in x]), axis=1)]
            if strict:
                antipat = list(set([i for i in df.keys() if type(i) == int]) - set(pat))
                df = df[df[list(antipat)].apply(lambda x: not any(["+" in i for i in x]), axis=1)]
    print("Now your dataframe in 'df' variable")
    return (df)
def dfull(df):
    pd.set_option("display.max_colwidth", 100)
    df.style.set_properties(**{'text-align': 'left'})
    with option_context('display.max_rows', None, 'display.max_columns', None):
        display(df)
def drop_na_all(df):
    return df.replace('', np.nan).dropna(how='all', axis=1).replace(np.nan, '')
class liuDf():
    def __init__(self, dfa, dfb, dfc, confDf=[]):
        self.da = dfa
        self.db = dfb
        self.dc = dfc
        self.dar = self.da.iloc[:, self.da.columns.get_loc(u' Occurance') - 6:]
        self.dbr = self.db.iloc[:, self.db.columns.get_loc(u' Occurance') - 6:]
        self.dcr = self.dc.iloc[:, self.dc.columns.get_loc(u'Name') - 5:]
        self.cnf = confDf

    def at(self, numConfCols=None):
        confCol(tp="da", numConfCols=numConfCols)

    def bt(self, numConfCols=None):
        confCol(tp="db", numConfCols=numConfCols)

    def ct(self, numConfCols=None):
        confCol(tp="dc", numConfCols=numConfCols)

    def a(self):
        dfull(self.da)

    def b(self):
        dfull(self.db)

    def c(self):
        dfull(self.dc)

    def plt(self):
        confColPlot()

    def ar(self):
        dfull(self.da.iloc[:, self.da.columns.get_loc(u' Occurance') - 6:])

    def br(self):
        dfull(self.db.iloc[:, self.db.columns.get_loc(u' Occurance') - 6:])

    def cr(self):
        dfull(self.dc.iloc[:, self.dc.columns.get_loc(u'Name') - 5:])
def confCol(tp="da", numConfCols=None):
    # CONFORMERS IN TWO_COLUMN tables
    confs1 = sorted([int(re.search(r'(?<=/ro_)\d+(?=/)', edDir2).group()) for edDir2 in edDirs])
    fullElp, elp = [], []
    numConfCols = numConfCols if numConfCols else len(confs1)
    for conf1 in confs1[:numConfCols]:
        k1 = getattr(ld.cnf[conf1], tp)
        if k1 is not None:
            if len(k1) > 1:
                elp.append("<td><h1>{}</h1></td><td>{}</td>".format(conf1, k1.to_html(escape=False)))
    lenElp = len(elp) / 2 + 1 if len(elp) % 2 > 0 else len(elp) / 2
    elp2 = grouper(lenElp, elp, fillvalue="")
    elp3 = zip(*elp2)
    display(HTML("<table><tr>{}</tr></table>".format("</tr><tr>".join(["".join(j) for j in elp3]))))
def confColPlot():
    mutations2 = copy.deepcopy(r2)
    # mutations2 = mutations2[:50]
    pl.rcParams['figure.figsize'] = (len(mutations2) / 10, 6)
    df2 = DataFrame.from_records(
        [{"name": int(mut.name[0]), "i_s": float(mut.i_s), "r_s": float(mut.r_s)} for mut in mutations2])
    for param in ['i_s', 'r_s']:
        display(df2.boxplot(column=param, by='name'))
    return [df2.boxplot(column=param, by='name') for param in ["i_s", "r_s"]]
class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)
def exd(*args, **kwargs):
    db = True if not "db" in kwargs.keys() else kwargs["db"]
    overwrite = True if not "overwrite" in kwargs.keys() else kwargs["overwrite"]
    wd = '/home/domain/data/rustam/dgfl/edRes/'
    exMutsFullFile = "{}exMuts.txt".format(wd)
    if not os.path.exists(wd):
        os.makedirs('edRes')
    exMuts = filter(lambda x: tuple([int(i) for i in x.name]) in args, mutations)
    if len(exMuts) != len(args):
        print
        len(exMuts), len(args)
        raise (ValueError(""))
    print("Pushing {} entered muts to exMuts.txt".format(len(exMuts)))
    if not exMuts:
        warn("no exMuts ")
    if open(exMutsFullFile).read():
        exMutsFull = [obj(j) for j in json.loads(open(exMutsFullFile).read())]
    else:
        exMutsFull = []
    # exMuts = filter(lambda x:  x in args, mutations)
    alreadyInExMuts = filter(lambda x: x.ed_id == ed_id, exMutsFull)
    print(
        "found {} decoys of {} (total: {}) in exMuts.txt file: {}".format(len(alreadyInExMuts), ed_id, len(exMutsFull),
                                                                          str(["_".join(i.name) for i in
                                                                               alreadyInExMuts])))
    for ed_dir_id in list(set([j.ed_id for j in exMutsFull])):
        print("{}\n\t{}".format(ed_dir_id, "+".join(
            ["_".join(k.name) for k in filter(lambda x: ed_dir_id == x.ed_id, exMutsFull)])))
    if overwrite:
        exMutsFull = filter(lambda x: ed_id != x.ed_id, exMutsFull)
    for i in exMuts:
        if db: print("pushing {}".format(i.name))
        setattr(i, "ed_id", ed_id)
        if i.name in filter(lambda x: (x.name, x.ed_id) == (i.name, ed_id), exMutsFull):
            print("mut elem {}, {}, ({}) already in this array. skipping".format(i.name, i.i_s, i.r_s,
                                                                                 str(i.__dict__)[:100]))
        else:
            exMutsFull.append(i)
            print(len(exMutsFull))
    open(exMutsFullFile, "w").write(json.dumps([ob.__dict__ for ob in exMutsFull]))
    print("RESULT:\nfound {} decoys of {} (total: {}) in exMuts.txt file: {}".format(len(alreadyInExMuts), ed_id,
                                                                                     len(exMutsFull), str(
            ["_".join(i.name) for i in alreadyInExMuts])))
    for ed_dir_id in list(set([j.ed_id for j in exMutsFull])):
        print("{}\n\t{}".format(ed_dir_id, "+".join(
            ["_".join(k.name) for k in filter(lambda x: ed_dir_id == x.ed_id, exMutsFull)])))
    if db:
        dpf(exMutsFullFile)
def rar(sch=None, db=False, dSch=0, globalRun=True, onlyLocal=True, filterFunc=None, rld=False):
    global ld, default_seq, ult_mut2, ult_mut3, r2, edDirs, ult, mutations
    ld = None
    print("AAAAAAAAAAAAAA", ld)
    mutations = []
    if globalRun:
        for mIdx, mutEdDirId in enumerate(os.popen(
                """ for j in $(find /home/domain/data/rustam/dgfl/ed/ed_???/rpr/mutations.txt); do dir="${j%rpr/mutations.txt}"ro_0 && if grep -q "BFF" $dir/lpla_lg2__DE_1.pdb; then dir2=${dir#$HOME/dgfl/ed/ed_} && echo ${dir2%/ro_0}; fi 2>/dev/null; done """).read().split(
            "\n")):
            if not mutEdDirId: continue  # if mutEdDirId = ""
            if onlyLocal: mutEdDirId = ed_id
            # if mIdx > 2:
            #     continue
            vd(mutEdDirId)
            for j2 in json.loads(
                    open("/home/domain/data/rustam/dgfl/ed/ed_{}/rpr/mutations.txt".format(mutEdDirId)).read()):
                dirMutations = []
                dirMutObj = obj(j2)
                setattr(dirMutObj, "ed_id", mutEdDirId)
                if not "SR_3_total_score" in dirMutObj.__dict__.keys():
                    warn(
                        "Item from {} had no attribute SR_3_total_score and was excluded [edited: not excluded]. Check it manually.".format(
                            mutEdDirId))
                    dirMutations.append(dirMutObj)
                    mutations.extend(dirMutations)
                    # break
                else:
                    dirMutations.append(dirMutObj)
                    mutations.extend(dirMutations)
            if onlyLocal: break
    else:
        mutations = [obj(j) for j in json.loads(open("mutations.txt").read())]
    ult_mut2, ult_mut3, r2 = [], [], []
    schs = [{'r_s': 1.0, 'tot_pstat_pm': 1.0, 'tot_burunsat_pm': 1.0, "i_s": 1.0},
            {'r_s': 0.5, 'tot_pstat_pm': 0.9, 'tot_burunsat_pm': 0.9, "i_s": 0.5},
            {'r_s': 0.3, 'tot_pstat_pm': 0.9, 'tot_burunsat_pm': 0.9, "i_s": 0.18},
            {'r_s': 0.5, 'tot_pstat_pm': 0.5, 'tot_burunsat_pm': 0.5, "i_s": 0.2},
            ]
    if sch == None:
        display(list(enumerate(schs)))
        sch = schs[int(raw_input("Please enter a scheme: "))]
        interSectScores = sch
    elif isinstance(sch, dict):
        interSectScores = sch
    elif isinstance(sch, int):
        interSectScores = schs[sch]

    preIsDecs = []
    mutations2 = copy.deepcopy(mutations)
    if filterFunc:
        mutations2 = filter(filterFunc, mutations2)
    preIsDecs = []
    for k4, v4 in interSectScores.items():
        preIsDecs.append(
            sorted(mutations2, key=lambda x: getattr(x, k4), reverse=k4 in ['tot_pstat_pm'])[
            :int(v4 * len(mutations2))])
    r2 = set.intersection(*map(set, preIsDecs))
    print("FOUND {} DECS".format(len(r2)))
    assert len(r2) != 0
    """ DATA CLUSTERISATION. MUTATIONS ARE ARRANGED INTO CLUSTERS THEN THEIR MEAN R_S AND I_S WILL BE CALCULATED AND ID NUMBERS WILL BE ASSIGNED"""

    def MutList2Df(list_0, cnfNum=-1, type_="a", dSch=0):
        list_1 = copy.deepcopy(list_0) if cnfNum == -1 else filter(lambda x: int(x.name[0]) == cnfNum,
                                                                   copy.deepcopy(list_0))
        default_seq = sorted([(r, lplaSeq[r]) for r in all_mutations_p(list_1)], key=lambda x: x[0])
        list_ = clustDec(list_1, type_=type_)
        if type_ == "b":  # type_ == b
            if db: print("b")
            for i in list_:
                p = []
                for j in [tuple3[0] for tuple3 in default_seq]:
                    if j in i.data_p:
                        p.append("+")
                    else:
                        p.append(" ")
                i.liu = p
        elif type_ == "a":  # type_ == a
            if db: print("a")
            for i in list_:
                p = []
                for tuple3 in default_seq:
                    if tuple3[0] in i.data_p:
                        p.append("+" + i.data_a[i.data_p.index(tuple3[0])])
                    else:
                        p.append(tuple3[1])
                i.liu = p
        elif type_ == "c":
            if db: print("c")
            for i in list_:
                p = []
                for tuple3 in default_seq:
                    if tuple3[0] in i.data_p:
                        p.append("+" + i.data_a[i.data_p.index(tuple3[0])])
                    else:
                        p.append(tuple3[1])
                i.liu = p

        # dispValuesShortCopy = copy.copy(dispValuesShort)
        dispValuesShortCopy = copy.copy(mutations2[0].__dict__.keys())
        for intrinsicValue in ['i_s', 'r_s', 'ide', 'name', 'qua', 'data_a', 'data_p', 'ed_id']:
            try:
                dispValuesShortCopy.remove(intrinsicValue)
            except Exception:
                pass

        # we should not try to display df with columns, that were not in scorefile.txt:
        emptyList = [" "] if not type_ == "c" else []
        df_pretitle = [
            [" "] * (2 + len(dispValuesShortCopy)) + [" Wild type"] + emptyList + [r1[1] for r1 in default_seq]]
        nameAndQua = [["_".join([x.ed_id] + x.name)] if type_ == "c" else [x.ide, x.qua] for xidx, x in
                      enumerate(list_)]
        dispValuesAttrs = []
        df = [
            ([getattr(x, dispVal) for dispVal in dispValuesShortCopy] + [x.i_s, x.r_s] + nameAndQua[xidx] + list(
                [y.liu for y in list_][xidx]))
            for xidx, x in enumerate(list_)]
        df = sorted(df, key=lambda x: x[len(dispValuesShortCopy)])  # sort by ifE
        df[0:0] = df_pretitle
        # print df[0]
        # print dispValuesShortCopy
        if type_ == "c":
            df2 = DataFrame(data=array(df, dtype=str),
                            columns=dispValuesShortCopy + ["{}-amp ifE".format(ln), "TotScore"] + ["Name"] + [
                                tuple3[0] + 1
                                for tuple3 in
                                default_seq])
        else:
            df2 = DataFrame(data=array(df, dtype=str),
                            columns=dispValuesShortCopy + ["{}-amp ifE".format(ln), "TotScore"] + ["SEQ ID",
                                                                                                   " Occurance"] + [
                                        tuple3[0] + 1 for tuple3 in default_seq])
        # newly added:
        cols = df2.columns.tolist()
        cols.insert(cols.index('TotScore') - 2, cols.pop(cols.index('tot_burunsat_pm')))
        cols.insert(cols.index('TotScore') - 3, cols.pop(cols.index('tot_pstat_pm')))
        if 'SR_3_hbond_sc' in cols:
            cols.insert(cols.index('TotScore') - 4, cols.pop(cols.index('SR_3_hbond_sc')))
        df3 = df2.reindex(columns=cols)
        # df2ull(df23)
        return df3

    # generating "integral" dataframe-obj-a,b,c:
    ld = liuDf(MutList2Df(r2, type_="a", dSch=dSch), MutList2Df(r2, type_="b", dSch=dSch),
               MutList2Df(r2, type_="c", dSch=dSch))

    edDirs = glob.glob('../ro_*/')
    vd(edDirs)
    for conf1 in sorted([int(re.search(r'(?<=/ro_)\d+(?=/)', edDir2).group()) for edDir2 in edDirs]):  # [0,1,2,3.....]
        ld.cnf.append(liuDf(MutList2Df(r2, cnfNum=conf1, type_="a", dSch=dSch),
                            MutList2Df(r2, cnfNum=conf1, type_="b", dSch=dSch),
                            MutList2Df(r2, cnfNum=conf1, type_="c", dSch=dSch)))
    clear_output()
    print("Created: ld.da: {};   ld.db:   {}   ;    ld.dc:    {}".format(*map(lambda x: len(x), [ld.da, ld.db, ld.dc])))
    if rld: return ld
def pdl(cnfNum=-1):
    if cnfNum == -1:
        pm("load vmut.pse".format(cnfNum), srv_md=False)
    else:
        pm("load vmut_{}.pse".format(cnfNum), srv_md=False)
def rms_list(x, y):
    rd = [(i - j) ** 2 for i, j in zip(x, y)]
    return math.sqrt(sum(rd) / len(rd))
def dec_visu(pym_name, repr="spheres"):
    for com in pymol_com.format(pym_name, ln).split("\n"):
        pm(com, r=False)
    liu = "liu_mut_{}".format(pym_name)
    # pm("select {}, {} and pr_{}".format(liu, pymol_expr(liu_mut, sc=True), pym_name), "show sticks, {}".format(liu), "show spheres, {}".format(liu), "color yellow, {}".format(liu), "alter {}, vdw=0.5".format(liu), "rebuild", r=False)
    pm("select {}, {} and pr_{}".format(liu, pymol_expr(liu_mut, sc=True), pym_name), "show sticks, {}".format(liu),
       "show {}, {}".format(repr, liu), "util.cbay {}".format(liu), r=False, db=False)
def dec_rms(dec, dec_xs, max_rms):
    if dec_xs:
        if all([p > max_rms for p in [rms_list(dec, list_) for list_ in dec_xs]]):
            dec_xs.append(dec)
            return True
    else:
        dec_xs.append(dec)
        return True
def dirVmutGen(dirNum, numPmlDec, liuw=False, ssPath="../ri/", ssIdxs=[1, 1, 1, 1]):
    db = True
    vmutRepr = "sticks"
    dec_xs = []
    co2 = 0
    decQuan = {}
    MAX_DEC_RMS = 0.1
    FILTER_RMS_ON, FILTER_FLUR_ON = False, False
    # TO assess the shift we upload the starting structure (tmp):
    vd(lom2)
    # try:
    #     print(lom2)
    # except Exception:
    #     lom2 = False
    tmp2Load = "load ../../ed_U0F/ri/ss/ss_batch/lpla_lg2_0.pdb, tmp2" if lom2 else "load ../ri/ss/lpla_lg2.pdb, tmp2"
    # vd(tmp2Load)
    print(tmp2Load)
    # raise()
    ### HISTORY ###
    pm("load ../ri/ss/lpla_lg2_0001.pdb", "fetch 4tvy", "remove 4tvy & ! (c. A & br. (* w. 15 of org))",
       "super 4tvy, lpla_lg2_0001", "load ../../ed_2B8/ro_10/lpla_lg2_10__DE_5.pdb, ND_double_mutant",
       "fetch 4tvw, liuw")
    for chainIdx, chain in enumerate(["A", "B", "C", "D"]):
        chainIdx += 1
        pm("extract liuw_{}, liuw & c. {}".format(chainIdx, chain), "super liuw_{}, lpla_lg2_0001".format(chainIdx),
           r=False)
        pm("create 4tvw_chains, liuw_{0}, 1, {0}".format(chainIdx), r=False)
    pm("delete liu*", r=False)
    pm("load ../ri/ss/liuw_rlx_0/lpla_lg_0_0001.pdb, ss_liuw_rlx_0_lpla_lg_0_0001", r=False)
    pm("load ../ri/ss/lpla_lg2.pdb, ss_lpla_lg2", r=False)
    pm("select aaaV1_crys, {}".format(pymol_expr(aaaV2)), "select cat, {}".format(pymol_expr(liuCatRes)), "remove e. h",
       "show surface, br. (* w. 7 of org)", "util.cbao aaaV1", "util.cbaw cat",
       "show {}, cat | aaaV1 | org".format(vmutRepr),
       "select mut_crys, i. 20+147+149", "color green, mut_crys",
       'label aaaV1_crys & n. CA, "%s-%s" % (resi, resn)',
       "set label_size, 37", "set label_position, (2,2,2)",
       r=False)
    if liuw:
        Dirs = ["{0}ss/liuw_rlx_{1}/lpla_lg_{1}_000{2}.pdb".format(ssPath, chainIdx, ssIdxs[chainIdx]) for chainIdx in
                range(4)]
        vd(Dirs)
        pm(*["load {0}".format(Dir) for Dir in Dirs], r=False)
    pm("delete ps* & l_* & pr_* & mut* & liu* & mg_*", r=False)

    ### HISTORY ENDED ###
    # pm(tmp2Load, "extract amp, organic", "remove not ( {} )".format(pymol_expr(lig_amp_list[ln], sel="n")),
    #    "hide sticks; show lines; color red", "delete tmp2", r=False)
    pmlDecAnList = sorted(filter(lambda x: [int(x.name[0]), x.ed_id] in dirNum, mutations),
                          key=lambda mut: mut.name[0]) if dirNum != -1 else list(r3)
    for k in sorted(pmlDecAnList[:numPmlDec], key=lambda mut: mut.name[0]):
        co2 += 1
        dir_, dec_ = k.name
        pym_name = str(dir_) + "_" + str(dec_)
        vd(pym_name)
        dec_file = list(braceexpand("../../ed_{2}/ro_{0}/lpla_lg2{{_{0}__,__}}DE_{1}.pdb".format(dir_, dec_, k.ed_id)))
        # dec_file = list(glob.glob("../../ed_{2}/ro_{0}/lpla_lg2*.pdb".format(dir_, dec_, k.ed_id)))
        vd(dec_file)
        dec_file = filter(lambda x: os.path.exists(x), dec_file)[0]
        if db: vd(dec_file)
        if not dec_file: continue
        dec_str = PDBParser().get_structure("dec", dec_file)
        try:
            dec = array([at.coord[0] for at in dec_str[0]["X"]['H_{}'.format(ln.upper()), 1, ' '].child_list])
        except KeyError as e:
            warn("WRONG LIGAND:{}".format(str(e)))
            continue
        if dec_rms(dec, dec_xs, MAX_DEC_RMS) or not FILTER_RMS_ON:
            decQuan[dec_file] = 0
            prev_dec_file = dec_file
            #         print "LOADING {}, {}".format(dec_file, pym_name)
            pm("load {}, {}".format(dec_file, pym_name), "remove e. h", r=False)
            dec_visu(pym_name, repr=vmutRepr)
            pm("select aaaV1, {} & pr_{} & ! liu_mut* ".format(pymol_expr(aaaV2), pym_name),
               "select cat, {} & pr_{}".format(pymol_expr(liuCatRes), pym_name),
               "util.cbao aaaV1",
               "util.cbaw cat",
               "show {}, cat | aaaV1 | org".format(vmutRepr), r=False)
            for k2 in [[lplaSeq[j], k.data_a[k.data_p.index(j)], j + 1] for j in k.data_p]:
                res_sel = pymol_expr([k2[2]], sel="i", sc=True)
                if k2[1] == "A":
                    res_sel = "i. {}".format(k2[2])
                c33 = res_sel + " and pr_{}".format(pym_name)
                mut_name = "mut_{}".format(pym_name)
                pm("select {}, {}".format(mut_name, c33), "show {}, {}".format(vmutRepr, mut_name),
                   "util.cbag {}".format(mut_name),
                   "pseudoatom ps_{0}_{1}, i. {1} and pr_{0}".format(pym_name, k2[2]),
                   "hide labels, {} & i. {}".format(pym_name, k2[2]),
                   'label ps_{0}_{1}, "{2}"'.format(pym_name, k2[2], "".join([str(i) for i in k2])), r=False)
            cmd2 = "create {0}_{1}_{2}-{3}, ps* | l_{4} | pr_{4} | mut* | liu* | mg_{4} ".format(k.ed_id, dir_, dec_,
                                                                                                 "{}-{}".format(abs(
                                                                                                     round(k.i_s, 1)),
                                                                                                     abs(
                                                                                                         round(
                                                                                                             k.r_s,
                                                                                                             1))),
                                                                                                 pym_name)
            vd(cmd2)
            pm(cmd2, "set label_color, white", "set label_size, 35",
               "set label_position, (2,2,3.5)",
               "delete ps* & l_* & pr_* & mut* & liu* & mg_*", r=False)
        decQuan[prev_dec_file] += 1
    pprint.pprint(decQuan)
    pm("set pse_export_version, 1.721", "zoom org", "disable all", r=False)
    if dirNum == -1:
        pm("save vmut.pse", r=False)
    else:
        pm("save vmut_{}.pse".format(dirNum), r=False)
    print("DECOYS WHOS LINKER DIDN'T MATCH THE LIU'S LINKER WERE DELETED")
def get_view():
    view = [round(i, 2) for i in pml.get_view()]
    mcell("pml.set_view({})".format(view), md=False)
def vmutGen(numPmlDirs=0, numPmlDec=0, mp=False, names=[], targList="r2", liuw=False, ssPath="../ri/",
            filterFunc=lambda x: True,
            ssIdxs=[1, 1, 1, 1]):
    global r3
    db = True
    map(os.remove, glob.glob("../rpr/vmut*.pse"))
    numPmlDec = 10000 if not numPmlDec else numPmlDec
    numPmlDirs = 10000 if not numPmlDirs else numPmlDirs
    if names:
        r3 = filter(lambda nam: map(int, nam.name) in names, copy.deepcopy(eval(targList)))
    else:
        r3 = copy.deepcopy(eval(targList))
    if filterFunc:
        r3 = filter(filterFunc, copy.deepcopy(r3))
    vd(r3)
    # raise
    pmlDirList = list(set([(int(i.name[0]), i.ed_id) for i in r3]))
    if len(pmlDirList) > 3:  # number of directories < 3
        pmlDirList = chunkify(pmlDirList, 3)  # chunkify(list(set([(int(i.name[0]), i.ed_id) for i in r2])), 3)

    pmlDirList = pmlDirList[:numPmlDirs]
    if db: vd(pmlDirList)
    pmlDirList = pmlDirList + [-1] if len(
        pmlDirList) > 1 else pmlDirList  # Iterating over  all_confs (-1) and each conformer iteratively:
    if db: vd(pmlDirList)
    if mp:
        pool = Pool(14)
        pool.map(functools.partial(dirVmutGen, numPmlDec=numPmlDec), pmlDirList)
    else:
        dirVmutGen(-1, numPmlDec, liuw=liuw, ssPath=ssPath, ssIdxs=ssIdxs)
def dirMutGen(edDir, numPerlDecs=None, numEdDirs=None):
    vd(interfE, "interfE in dirMutGen")
    mutationsW = []
    db = True
    awkCmd = """
    scDir=\"""" + edDir + """scorefile.txt" && numRows="$(sed -n \$= $scDir)" && awk -v "numRows=$numRows" 'BEGIN {
    amatch2 = "^description$"
    co = 0
    getline
    liuTableString = ""
    for (i=1; i<NF+1; i++) {
    if ( $i !~ amatch2 ) {
    liuTableString = liuTableString" "i
    liuTableDict[i] = $i
    }

    }
    printf "%s",("[")
    }

    {
    n = split(liuTableString, val, " ")
    val[n+1] = $NF
    for (j=n+1; j>0; j--)
    {
    if (j == n+1) {
    printf "{\\"name\\": \\"%s\\", ", val[j]
    }
    else
    if (j == 1) {
    printf "\\"%s\\":%s ",liuTableDict[val[j]],$val[j]
    }
    else {
    printf "\\"%s\\":%s, ",liuTableDict[val[j]],$val[j]
    }
    }
    if (NR == numRows)
    {printf "%s","}] "}
    else {
    {printf "%s","}, "}
    }
    }

    ' $scDir
"""
    # print awkCmd
    try:
        perlDecs = eval(os.popen(awkCmd).read())
    except SyntaxError:
        print("some of your ro_ .. directories exists and empty. It is bad.")
        return [], []
    if db:
        print(edDir, interfE)
        print(len(perlDecs))
    numPerlDecs = numPerlDecs if numPerlDecs else len(perlDecs)
    for dec in perlDecs[:numPerlDecs]:
        # perlDecs: [....., {'name': 'lpla_lg2__DE_15', 'i_s': '-7.94', 'r_s': '-449.43'}]
        dirName = re.search(r'(?<=ro_).*(?=/)', edDir).group()
        #         if dirName not in ["43", "48"]:
        #             continue
        decSeq = [aaMap[j] for j in os.popen("awk 'BEGIN { id = 1} {if ($6 == id) { print $4; id++ }}' %s%s.pdb" % (
            edDir, dec["name"])).read().split("\n") if j]
        mut_p_a = zip(
            *sorted([[k, j[0]] for k, j in dict(enumerate(zip(decSeq, list(lplaSeq)))).items() if j[0] != j[1]],
                    key=lambda x: x[0]))
        if not mut_p_a:
            print("NAT_DEC detected")
            continue
        mutTuple = [(j[1], j[0], int(k + 1)) for k, j in dict(enumerate(zip(decSeq, list(lplaSeq)))).items() if
                    j[0] != j[
                        1]]  # [('E', 'S', 19), ('W', 'A', 36), ('V', 'G', 76), ('H', 'S', 78), ('H', 'G', 148), ('T', 'G', 150)]
        dec["name"] = (dirName, re.search(r'(?<=__DE_)\d*', dec["name"]).group())
        dec["data_a"] = mut_p_a[1]
        dec["data_p"] = mut_p_a[0]
        dec["i_s"] = dec.pop(interfE)
        dec["r_s"] = dec.pop("total_score")
        mutationsW.append(mut(**dec))
    vd(mutationsW)
    # ultW.append(decoy(mut_p_a[0], mutTuple, name=dec["name"], i_s=dec["i_s"], r_s=dec["r_s"]))
    return mutationsW
def mutGen(lom2=False, numEdDirs=0, numPerlDecs=0, db=False):
    global numNatDecs, numRuns, interfE, mutations
    # db = False
    mutations = []
    wd = "{}ed_{}/".format(lmpr, ed_id)
    if lom2:
        sp("rm -r ../ro_*")
        lscp('ro_*', '', toLom2=False)
    edDirs = glob.glob('../ro_*/')
    try:
        interfE = shc(
            """gawk 'BEGIN {a=getline; for (i=1; i<=NF; i++) {if ($i ~ /^SR_1_total_score$|^SR_3_interf_E_1_2$/) {print $i}}}' ../ro_*/scorefile.txt | tail -n1""",
            db=db)
    except IndexError:
        warn("len(edDirs) = 0\n if not lom2: edDirs = glob.glob('../ro_*/')")
        return
    if not interfE:
        raise (ValueError("interfE is empty. means (=''). run db=True in shc and remove the empty dir "))
    if db: vd(interfE, "interfE (i_s_ will be assigned to ")

    numEdDirs = numEdDirs if numEdDirs else len(edDirs)
    pool = multiprocessing.Pool(14)
    mutationsMult = pool.map(functools.partial(dirMutGen, numPerlDecs=numPerlDecs, numEdDirs=numEdDirs),
                             edDirs[:numEdDirs])

    mutations = list(itertools.chain.from_iterable(mutationsMult))
    mutations = filter(lambda x: x != [], mutations)
    print(mutations)

    open("mutations.txt", "w").write(json.dumps([ob.__dict__ for ob in mutations]))
def lom2copyDirs():
    lom2Dirs = ['2B9', 'RDS', '2a6', 'vbv', 'vFg', '8AP', 'll7', 'N36', 'pXa', 'CeY', 'xiY', 'U0F', 'dlV', 'gDk']
    lom2Dirs.remove('gDk')
    os.system('tmux ls >/dev/null 2>&1 || tmux new-session -d ')  # if no server running
    tmux_srv = libtmux.Server()
    tmName = "lom2Copy"
    en_ssn = tmux_srv.find_where({'session_name': tmName})
    if en_ssn:
        en_ssn.kill_session()
    frstItem = lom2Dirs[0]
    en_ssn = tmux_srv.new_session(tmName, window_name=frstItem)
    for core_ in lom2Dirs:
        if core_ != frstItem:
            en_win = en_ssn.new_window(window_name=core_)
        else:
            en_win = en_ssn.find_where({'window_name': frstItem})
        cmnd = "scp -r fbbstudent@lomonosov2.parallel.ru:/home/fbbstudent/_scratch/work/rustam/dgfl/ed/ed_{0}/ro_* /home/domain/data/rustam/dgfl/ed/ed_{0}".format(
            core_)
        en_win.attached_pane.send_keys("cd ../ && {}".format(cmnd))
def inspChanges(confLib, attr, srv_md=srv_md, prompt=False, db=False):
    pm("load unl.acpype/unl_NEW.pdb", 'label all, "%s-%s" % (ID, name)', 'zoom ID 1', srv_md=srv_md)
    cnf = confLib.GetConformer()
    bond_list = []
    attrArgNum = {'bonds': 2, 'angles': 3, 'dihedrals': 4}[attr]
    attrArgNumPlus = {'bonds': 3, 'angles': 4, 'dihedrals': 5}[attr]
    rdAttr = {'bonds': "GetBondLength", 'angles': "GetAngleDeg", 'dihedrals': "GetDihedralDeg"}[attr]
    for idx, bond in enumerate(getattr(itp.header.moleculetype, attr).data):
        bondBefore = str(bond)
        bondIdxs = [int(i) for i in list(bond)[:attrArgNum]]
        rdbond = getattr(Chem.rdMolTransforms, rdAttr)(cnf, *[i - 1 for i in bondIdxs])
        rdbond = rdbond * 0.1 if attr == "bonds" else rdbond
        if attr == "dihedrals":
            if round(rdbond, -1) in [0, 180, -180]:
                rdbond = abs(round(rdbond, -1))
            else:
                vd(rdbond)

        acbond = float(bond[attrArgNumPlus])
        dist = round(abs(abs(rdbond) - abs(acbond)), 1)
        #     vd(bondIdxs)
        #     vd(dist)
        strBondIdxs = "+".join(map(lambda x: str(x), bondIdxs))

        def pmLabel(distThrs=0):
            if dist > distThrs:
                # bondLabel = "{}|{}".format(strBondIdxs, round(acbond,4))
                bondLabel = "{}|{}->{}|{}".format(strBondIdxs, round(acbond, 4),
                                                  round(rdbond, 4), dist)
                pm("pseudoatom bond_{}, ID {}, label='{}'".format(idx, strBondIdxs, bondLabel), r=False, srv_md=srv_md)

        ctrlList = list(bond)[:attrArgNum]
        vd(ctrlList)
        if filter(lambda x: x[1] == "B", itp.header.moleculetype.atoms.data)[0][0] in ctrlList or attr == "dihedrals":
            if attr == "bonds":
                bond[3], bond[4] = rdbond, 400000
                pmLabel(0)
                print("changed bond: {} ->  {}".format(bondBefore, bond))
            elif attr == "angles":
                bond[4], bond[5] = rdbond, 440
                pmLabel(6.0)
                print("changed bond: {} ->  {}".format(bondBefore, bond))
            elif attr == "dihedrals":
                mult = "2" if rdbond in [0.0, 180.0] else "0"
                # if mult == "2" and all([i not in range(16,33) for i in list(bond)[:2]]) and int(bond[7]) == 2:
                #     # bond[5], bond[6], bond[7] = rdbond, '200.0', mult
                #     bond[6] = '200.0'
                #     print("changed bond: {} ->  {}".format(bondBefore, bond))
                #     pmLabel()
                if 1 in list(bond)[:attrArgNum] and not float(bond[6]):  # constraints not defineed
                    bond[5], bond[6], bond[7] = rdbond, '4.0', mult
                    print("changed bond: {} ->  {}".format(bondBefore, bond))
                    pmLabel()
            else:
                print("skipped: {}. Bond would be: {} , {}".format(bond, rdbond, rdbond / (acbond)))
        else:
            print("skipped: {}. Bond would be: {} , {}".format(bond, rdbond, rdbond / (acbond)))
    if db:
        pm("set pse_export_version, 1.721", "save /home/domain/rustam/.tmp/bond.pse", r=False, srv_md=srv_md)
        pmr("load /home/domain/rustam/.tmp/bond.pse")
    if prompt and attr != "dihedrals": raw_input()
def fullPath(files, cwd=os.path.curdir, glb=True):
    if isinstance(files, list):
        list_of_filenames = map(lambda p: glob.glob(os.path.realpath(os.path.join(cwd, p))), files)
        return list(itertools.chain(*list_of_filenames))
    else:
        glbList = glob.glob(os.path.realpath(os.path.join(cwd, files)))
        if len(glbList) > 1 and glb:
            return glbList
        else:
            return glbList[0]
def rmPat(files=None, cwd=os.path.curdir, r=False, db=False, prompt=True):
    """

    :param files:    if isinstance(files, str): files = [files]
    :param cwd:
    :param r:
    :param db:
    :param prompt:
    :return:
    """
    files = fullPath(files, cwd=cwd)
    if not files:
        files = dirSelMult()
        rmp(files, prompt=False)
    else:
        if isinstance(files, str):
            files = [files]
        for dirpath, dirnames, filenames in os.walk(cwd, ):
            if os.path.realpath(cwd) == os.path.realpath(dirpath):
                dir_n_file_names = fullPath(dirnames + filenames, cwd=cwd)
                not_found_files = list(set(files) - set(dir_n_file_names))
                if not_found_files: warn(vd(not_found_files))
                if db: vd(files, "files before")
                if r:
                    files = list(set(dir_n_file_names) - (set(files) - set(not_found_files)))
                else:
                    files = list((set(files) - set(not_found_files)))
                if db: note("", vd(files), vd(not_found_files), vd(dir_n_file_names))
                if files:
                    rmp(files, prompt=prompt)
                else:
                    warn("No files!")
def rmp(files=None, prompt=False):
    if not prompt or rInp(
            "\nDelete following files:\n{0}\n{1}\n{0}\n? [Y/n]\n".format("-" * len(files[0]), "\n".join(files))) == "Y":
        rmFunc = lambda x: os.system("rm -rf {}".format(x))
        rmFunc2 = lambda x: os.system("rm -rf '{}'".format(x))
        [*map(rmFunc, files)]
        [*map(rmFunc2, files)]
    if not prompt:
        print("\nDeleted following files:\n{0}\n{1}\n{0}\n".format("-" * len(files[0]), "\n".join(files)))
def gmxIns(topol, pat, insertion, after=True, endw=True, ins=False):
    topolSplit = topol.split("\n\n")
    found = [*filter(lambda x: x.endswith(pat) if endw else x.startswith(pat), topolSplit)]
    if len(found) > 1:
        warn("found 2 instances: {}.....".format(str(found)[:50]))
    elif not found:
        warn("not found!!!")
        return
    foundIdx = topolSplit.index(found[0])
    foundIdx = foundIdx + 1 if after else foundIdx
    #     print topolSplit[-1]
    if ins:
        topolSplit[foundIdx - 1] = insertion
    else:
        topolSplit[foundIdx:foundIdx] = [insertion]
    return "\n\n".join(topolSplit)
def mdDir(dir_, crop=0, trSl=1, pd=0, nom=0, db=0):
    """
    :param dir: ["2B8_10_46", "../2B8_10_46", "/home/domain/data/rustam/dgfl/md/2B8_10_46" OR WITH /1] OR WITH TRAILING SLASHES
    :print: 2B8_10_46/ or  /home/domain/data/rustam/dgfl/md/2B8_10_46/ if trSl == True
    """
    dir_ = dir_.replace('__', '/')
    if os.path.exists(dir_):  # ../HRX_0_1 or /home/domain/data/rustam/dgfl/md/HRX_0_1 or with /1
        dir_ = os.path.realpath(dir_)
    else:  # HRX_0_1 or HRX_0_1/1
        stripped_dir = dir_.strip("./")
        maybe_dir = mdr + stripped_dir
        if os.path.exists(maybe_dir):
            dir_ = maybe_dir

    if db:
        vd(dir_)
    if mdr not in dir_:
        pass
        # raise(ValueError("if mdr not in dir_"))
    cropped_dir = dir_.replace(mdr, "")
    if crop:
        dir_ = cropped_dir
    if nom:
        dir_ = cropped_dir.strip("./").replace("/", "__")
        return dir_
    if trSl:
        dir_ = dir_ + "/"
    if pd:
        dir_ = "../" + cropped_dir

    # return dir_
    return (dir_)
def getMdDir(crop=False, all=True, trSl=True, pd=False, nom=0, current_directory=''):
    if all:
        directories = filter(lambda x: len(x.split("_")) == 3 and x.split("_")[1].isdigit(), os.listdir(os.path.pardir))
        return [mdDir(j, crop=crop, trSl=trSl, pd=pd, nom=nom) for j in
                directories]
    else:
        current_directory = os.path.realpath(os.curdir) if not current_directory else current_directory
        return mdDir(current_directory, crop=crop, trSl=trSl, pd=pd, nom=nom)
def dirSelMult(widgetDesc="select files/directories", widgetOptions=os.listdir(os.path.curdir), widgetValue=None):
    global button
    # query_input = widgets.Text(description="Query string:")
    widgetValue = tuple([widgetOptions[0]]) if not widgetValue else widgetValue
    button = widgets.Button(description="Submit")
    query_input = widgets.SelectMultiple(
        options=widgetOptions,
        value=widgetValue,
        rows=len(widgetOptions),
        description=widgetDesc,
        disabled=False
    )

    def input_observe(ev):
        value = ev['new']

    query_input.observe(input_observe, 'value')
    box = widgets.VBox(children=[query_input, button])
    display(box)
    get_ipython().magic('block button')
    clear_output()
    return query_input.value
def pmtr(comress_traj=1, mdirs=[], db=0, srv_md=True, pilot=0, sln='not resname SOL', nf=40, fs=0, start=0, stop=-1):
    # pm("", srv_md=srv_md)
    # CHOOSING MAIN DIRECTORY. IN CASE OF Ms - it is Ms.
    main_dir = mdirs[0] if not len(mdirs) else mdirs[0].parent
    print("main_dir:", main_dir)
    for mdir in mdirs:
        pm("", srv_md=srv_md)
        print(mdir)
        display(mdir)
        print(f'\r->{mdir.nom} Compressing traj..')
        if comress_traj:
            success = mdir.compress_traj(pilot=pilot, sln=sln, nf=nf, fs=fs, db=db, start=start, stop=stop)
            if not success:
                warn(f'skipping {mdir}')
                continue
        pm("set defer_builds_mode, 3", f"load {mdir.joinpath('mda.gro')}, {mdir.nom}",
           f"load_traj {mdir.joinpath('mda.xtc')}", db=db, r=False, srv_md=srv_md)
        # cmd_ = ["show spheres, i. 20 | org | r. SOL", "orient all", "select imp, * w. 20 of org",
        #          "color cyan, r. SOL", f"extract prot, polymer",
        #         f"show surface, prot"]
        # note("Vars", vd(cmd_))
        # pm(*cmd_, r=False, srv_md=srv_md, db=db)
        pm(f"save {mdir.joinpath('mda.pse')}, all", r=False)
    # pm(f"save {main_dir.joinpath('mda.pse')}", db=db, r=False, srv_md=srv_md)
def dr(obj):
    print(", ".join([k for k in dir(obj) if not k.startswith("_")]))
def brc(dirs):
    if len(dirs) == 1:
        return dirs[0]
    elif len(dirs) == 0:
        warn("empty!!!!")
    else:
        return "{{{}}}".format(",".join(dirs))
def mdsrv(sch=0, dirs=[], number_md_cores=10, db=False, cancel=False):
    warn("ECHO -E NOT WORKING WITH ! SIGN IN JUPYTER. USE BASH MAGIC. Use db=True to debug")
    dirs = "{}/".format(os.path.basename(os.path.realpath(os.path.curdir))) if not dirs else dirs
    if isinstance(dirs, str):
        dirs = [dirs]
    vd(dirs)
    dirs = [*map(lambda x: os.path.basename(os.path.realpath(x)), dirs)]

    if not isinstance(number_md_cores, list):
        number_md_cores = [number_md_cores // len(dirs) for _ in range(len(dirs))]
    assert (0 not in number_md_cores)
    doBashStrs = [doBashStr.format(num_md_dir_cores) for num_md_dir_cores in number_md_cores]
    os.system('tmux ls >/dev/null 2>&1 || tmux new-session -d ')  # if no server running
    tmux_srv = libtmux.Server()
    for mdDirIdx, mdDir in enumerate(dirs):
        if cancel:
            tmux_srv.kill_session(mdDir)
            continue
        realDir = "/home/domain/data/rustam/dgfl/md/{}".format(mdDir)
        vd(mdDir)
        mkDir("../.tmp/")
        open("../.tmp/do.bash".format(mdDirIdx), "w").write(doBashStrs[mdDirIdx])
        vd(realDir)
        if db:
            mcell("%%bash\n{}".format(doBashStrs[mdDirIdx]), md=False)
            break
        tsp("cd {} && bash ../.tmp/do.bash".format(realDir, mdDirIdx), sn=mdDir)
def mdlom(dirs=[], sch="em", num_lm_nodes=1, lmTime=10, rwDirs=False, cpDirs=False, cpDoBash=False, cancel=False,
          launch=True, reverseCopy=False, reverseCopyAll=False):
    if cancel:
        try:
            global job_id
            scancel(job_id);
        except Exception as e:
            note(str(e))
        return
    dirs = "{}/".format(os.path.basename(os.path.realpath(os.path.curdir))) if not dirs else dirs
    if isinstance(dirs, str):
        dirs = [dirs]
    dirs = [*map(lambda x: os.path.basename(os.path.realpath(x)), dirs)]
    rwdir("/home/domain/rustam/.tmp/doBash/")
    rl = []
    if reverseCopy:
        for Dir in dirs:
            sp(
                "scp -r fbbstudent@lomonosov2.parallel.ru:/home/fbbstudent/_scratch/work/rustam/dgfl/md/{0}{{npt.tpr,md.tpr,md.xtc}} {1}{0}".format(
                    mdDir(Dir, crop=True), mdr))
        return
    if rwDirs:
        lmsp("rm -r {}{}".format(lmdr, brc(dirs)))
    doBashPath = "/home/domain/rustam/.tmp/doBash/do.bash"
    open(doBashPath, "w").write(doBashStr.replace("gmx", "gmx514").format(num_lm_nodes * 14))
    if cpDirs:
        lscp("{}{}".format(mdr, brc(dirs)), "{}".format(lmdr), ed_id=None)
    if cpDoBash:
        lscp(doBashPath, lmdr, ed_id=None)
    for dir_ in dirs:
        dir_ = "{}/".format(dir_)
        rl.append({
            'name': "{}".format(dir_),
            'path': "{}{}".format(lmdr, dir_),
            'command': "bash {}do.bash".format(lmdr),
            'cpu': num_lm_nodes * 14,
            'out': "{}{}out.out".format(lmdr, dir_),
            'err': "{}{}err.err".format(lmdr, dir_),
        })
    if not launch: return
    open("/home/domain/rustam/.tmp/f.run", "w").write(json.dumps(rl, indent=4))
    lscp("/home/domain/rustam/.tmp/f.run", "/home/fbbstudent/_scratch/work/rustam/.run/", ed_id=False)
    lsp("chmod 777 -R {}{}".format(lmdr, brc(dirs)))
    rsp = lmsp("python /home/golovin/_scratch/progs/lom_tools/run_multiple_tasks.py -t {}.run/f.run -p".format(lmr),
               silent=True)
    job_id_str = lmsp(rsp.split("########## Submit command ##########\n")[-1].strip().replace("sbatch",
                                                                                              "cd {2}{0} && sbatch -J rst_md -t {1} ".format(
                                                                                                  dirs[0], lmTime,
                                                                                                  lmdr)))
    job_id = int(job_id_str.split("Submitted batch job")[-1].strip())
    note("Your out files will be in {}".format(dirs[0]))
    ringStarted(job_id, ed_id, clo=False)
def coordPdb2Gro(coord, coord2=None, r=False):
    # vd(coord)
    # vd(coord2)
    def pdb2gro(coord):
        p = list(coord)
        k = p.index(".")
        p.pop(k)
        p.insert(k - 1, ".")
        return "".join(p[:-1])

    if isinstance(coord, str):
        if coord2:
            return pdb2gro(coord) == coord2 or pdb2gro(coord2) == coord1
        else:
            return pdb2gro(coord)
    elif isinstance(coord, list):
        if coord2:
            return all([pdb2gro(x) == y or pdb2gro(y) == x for x, y in zip(coord, coord2)])
        else:
            return map(pdb2gro, coord)
def edRun(confsInOneFile=True, lom2=False, rwDir=True, sStrQuan=0, num_ed_cores=8, lmTime=40):
    if not sStrQuan:
        sStrQuan = len([re.search(r'(?<=_)\d+(?=/)', Dir).group() for Dir in glob.glob("cg/prms/prms_*/")])
    # rstOut = " > out.txt " if rstOut else ""
    # lom2 = True

    # if lom2:
    #     rosettaStr = "/home/golovin/_scratch/progs/rosetta/rosetta_bin_linux_2016.32.58837_bundle/main/source/bin/enzyme_design.linuxgccrelease"
    #
    # else:
    #     rosettaStr = "/home/domain/data/prog/rosetta_2017/main/source/bin/enzyme_design.default.linuxgccrelease"

    # cmnds = [
    #     "{1} @ri/msc/ed.flags -in:file:s ./ri/ss/lpla_lg2.pdb -extra_res_fa ./ri/cg/{2}.params -out:path:pdb ./ro_{0} -out:file:o ./ro_{0}/scorefile.txt".format(
    #         core_, rosettaStr, ln.upper()) for core_ in range(sStrQuan)]
    if not lom2:
        if confsInOneFile:
            os.system('tmux ls >/dev/null 2>&1 || tmux new-session -d ')  # if no server running
            tmux_srv = libtmux.Server()
            ed_i = 'ed_{}'.format(ed_id)
            en_ssn = tmux_srv.find_where({'session_name': ed_i})
            os.system("rm -r ../ro_*; rm ../rpr/*.csv")

            if en_ssn:
                en_ssn.kill_session()
            en_ssn = tmux_srv.new_session(ed_i, window_name='core_0')
            for core in range(num_ed_cores):
                if core != 0:
                    en_win = en_ssn.new_window(window_index=core, window_name='core_{}'.format(core))
                else:
                    en_win = en_ssn.find_where({'window_name': 'core_0'})
                ro_i = '../ro_{}'.format(core)
                os.makedirs(ro_i)
                cmnd = "/home/domain/data/prog/rosetta_2017/main/source/bin/enzyme_design.default.linuxgccrelease @ri/msc/ed.flags -out:path:pdb ./ro_{0} -out:file:o ./ro_{0}/scorefile.txt -in:file:s ./ri/ss/lpla_lg2.pdb -extra_res_fa ./ri/cg/{1}.params".format(
                    core, ln.upper())
                vd(cmnd)
                en_win.attached_pane.send_keys("cd ../ && {}".format(cmnd))
        #             break

        else:
            os.system("rm -r ../ro_*; rm ../rpr/*.csv")
            for core in range(sStrQuan):
                rwdir('../ro_{}'.format(core))
            os.system('tmux ls >/dev/null 2>&1 || tmux new-session -d ')  # if no server running
            tmux_srv = libtmux.Server()
            ed_i = 'ed_{}'.format(ed_id)
            en_ssn = tmux_srv.find_where({'session_name': ed_i})
            if en_ssn:
                en_ssn.kill_session()
            en_ssn = tmux_srv.new_session(ed_i, window_name='core_0')

            cmnds = [
                "{1} @ri/msc/ed.flags -in:file:s ./ri/ss/ss_batch/lpla_lg2_{0}.pdb -extra_res_fa ./ri/cg/prms/prms_{0}/{2}.params -out:path:pdb ./ro_{0} -out:file:o ./ro_{0}/scorefile.txt".format(
                    core_, rosettaStr, ln.upper()) for core_ in range(sStrQuan)]
            vd(cmnds)
            num_exe_ed_cores = (sStrQuan // num_ed_cores + 1)
            cmndQueues = ["cd /home/domain/data/rustam/dgfl/ed/ed_{} && ".format(ed_id) + " && ".join(cmd) for cmd in
                          grouper(num_exe_ed_cores, cmnds, "echo 'Queue filler, just ignore it'")]
            for core in range(len(cmndQueues)):
                if core != 0:
                    en_win = en_ssn.new_window(window_index=core, window_name='core_{}'.format(core))
                else:
                    en_win = en_ssn.find_where({'window_name': 'core_0'})
                en_win.attached_pane.send_keys(cmndQueues[core])
    else:
        wd = "{}ed_{}/".format(lmpr, ed_id)
        if rwDir:
            print("REMOVE MANUaLLY YOUR DIR!!!!")
            #         rwdir(plms(wd))
            print("copying directory ...")
            #         sp(" scp -r /home/domain/data/rustam/dgfl/ed/ed_{0}/ri fbbstudent@lomonosov2.parallel.ru:/home/fbbstudent/_scratch/work/rustam/dgfl/ed/ed_{0}/ ".format(ed_id))
            print("copied!")
        sp(
            """ ssh fbbstudent@lomonosov2.parallel.ru "rm -r /home/fbbstudent/_scratch/work/rustam/dgfl/ed/ed_{0}/ro_*" """.format(
                ed_id))
        for core in range(sStrQuan):
            sp(
                """ ssh fbbstudent@lomonosov2.parallel.ru "mkdir /home/fbbstudent/_scratch/work/rustam/dgfl/ed/ed_{}/ro_{}" """.format(
                    ed_id, core))
        rl = []
        for core in range(sStrQuan):
            doBashStr = """#!/bin/bash
    export OMP_NUM_THREADS=1
    export LD_LIBRARY_PATH=${{LD_LIBRARY_PATH}}:/home/golovin/_scratch/progs/rosetta/rosetta_bin_linux_2016.32.58837_bundle/main/source/build/external/release/linux/3.10/64/x86/gcc/4.8/static/
    {} """
            open("{}do_{}.bash".format(plms(wd), core), "w").write(doBashStr.format(cmnds[core]))
            rl.append({
                'name': "{}_{}".format(ed_id, core),
                'path': '/home/fbbstudent/_scratch/work/rustam/dgfl/ed/ed_{}'.format(ed_id),
                'command': 'do_{}.bash'.format(core),
                'cpu': 1,
                'out': '/home/fbbstudent/_scratch/work/rustam/dgfl/ed/ed_{}/{}.out'.format(ed_id, core),
                'err': '/home/fbbstudent/_scratch/work/rustam/dgfl/ed/ed_{}/{}.err'.format(ed_id, core)
            })
        os.system("chmod 777 -R {}*".format(plms(wd)))
        open("/home/domain/rustam/.tmp/f.run", "w").write(json.dumps(rl, indent=4))
        sp("scp /home/domain/rustam/.tmp/f.run {}.run/".format(lmr))
def mrk():
    mcell("".join(ld.da.to_html().split()))
def rmFile(fileName):
    if os.path.exists(fileName):
        os.remove(fileName)
def kw_list_gen(laq_hydr_cnts, match, unl_file_list, to_visu=False):
    kw_list = []
    visu_list = []
    for tr_it in laq_hydr_cnts:
        kws = {'verbose': True, 'debug': True, 'selection1_type': 'donor'}
        for tii, tr_it_pos in enumerate(tr_it):
            if isinstance(tr_it_pos, int):
                new_name = unl_file_list[match[tr_it_pos]]
                kws["selection{}".format(tii + 1)] = "(resname UNL) and (name {})".format(new_name)
                kws[["donors", "acceptors"][tii]] = [new_name]
            else:
                kws["selection{}".format(tii + 1)] = "(resid {}) and (name {})".format(*tr_it_pos)
                kws[["donors", "acceptors"][tii]] = [tr_it_pos[1]]
                # if to_visu:
                #     visu_list.append([tii + 1, new_name])

        kw_list.append(kws)
    if to_visu:
        return to_visu
    return kw_list
def stv(power=0):
    pm("mview store, object=nat", "mview store, object=mut", "mview store", "power={}".format(power), r=False)
def stv():
    pm("mview store, object=nat, freeze=1, auto=0", "mview store, object=mut, freeze=1, auto=0",
       "mview store, freeze=1, auto=0", r=False)
def mtf(fr=None, step=40):
    pm("frame {}".format(fr), r=False)
def rnt():
    pm("mview reinterpolate", r=False)
def expand_dict(decPathRouting):
    delete_me = []
    for k, v in decPathRouting.items():
        if "*" in k or "{" in k:
            for p3 in list(braceexpand(k)):
                for p2 in glob.glob(p3):
                    decPathRouting[p2] = v
            delete_me.append(k)
    for k in delete_me:
        del decPathRouting[k]
    return decPathRouting
def ltx(x):
    return str(x).replace("_", "\_")
def df2ltt(df, caption=""):
    strs = "\\begin{table}\\centering\\begin{adjustbox}{max width=\\textwidth}"
    strs += "\\begin{tabular}{|" + " | ".join(["c"] * len(df.columns)) + "|}\\hline\n"
    strs += " & ".join(["\\textbf{{{}}}".format(ltx(str(x))) for x in df.columns]) + " \\\\ \\hline\n"
    for i, row in df.iterrows():
        strs += " & ".join([ltx(str(x)) for x in row.values]) + " \\\\ \\hline\n"
    strs += "\\end{tabular}\\end{adjustbox}\\caption{" + caption + "}\\label{tab:1}\\end{table}"
    print(strs)




def msoc(x=None, make_oc_tables=0, at=0, exec=False):
    # from lib.base import Info
    import itertools
    import pandas as pd

    info = Info()
    if make_oc_tables:
        db = info.con1
        cur = db.cursor()
        cur.execute("show tables")
        db.close()
        wrv(list(itertools.chain(*cur.fetchall())), "oc_tables")
        return
    oc_tables = rdv("oc_tables", db=0)
    if at:
        return oc_tables
    if x in oc_tables:
        df = pd.read_sql(f'SELECT * FROM {x}', con=info.con1)
        return df
    elif exec:
        db = info.con1
        cur = db.cursor()
        cur.execute(x)
        db.close()
    else:
        df = pd.read_sql(x, con=info.con1)
        # OR pd.read_sql_table("oc_manufacturer_description", engine)
        return df

def gen_file(dir_or_list, in_prefix='', out_prefix=''):
    if isinstance(dir_or_list, P):
        files = dir_or_list.files
    elif isinstance(dir_or_list, list):
        # in_prefix autodetection:
        # if not in_prefix:
        #     for list_item in dir_or_list:
        #         m = re.search(r'.*(\d+)$', str(list_item))
        #         if m:
        #             in_prefix = m.group(0).split('/')[-1]
        files = dir_or_list
    elif isinstance(dir_or_list, types.GeneratorType):
        files = [*dir_or_list]
    else:
        raise ('Not a file not a P')
    digit_names = [int(i.name.replace(in_prefix, '')) for i in files if i.name.replace(in_prefix, '').isdigit()]
    new_name = max(digit_names) + 1 if digit_names else 0
    new_name = out_prefix + str(new_name)
    if isinstance(dir_or_list, P):
        return dir_or_list.joinpath(new_name)
    else:
        return new_name
class MetaPath(type):
    def __new__(cls, cls_name, attrs):
        cls2 = type.__new__(cls_name, pathlib.PosixPath, attrs)
        return cls2
class L2p:
    def __init__(self, script=cur_ser.d_u.joinpath('test.sh'), name='', t=300, n_nodes=1, pt=0, cwd=cur_ser.d_u):
        name = str(t) if not name else name
        #         files = [*d__tdir.rglob('sbs_*')]
        #         digit_names = [int(i) for i in sbs if i.name.isdigit()]
        #         self.id = max(digit_names) + 1 if digit_names else 0
        #         wrv(sbs + [self.id], 'sbs')
        t = 15 if pt else t
        if isinstance(script, str):  # Just a bash (obviously) command, not a file
            scripts_dir = cur_ser.d_u.joinpath('.run/scripts')
            scripts_dir.mkdir()
            script_file = gen_file(scripts_dir)
            script_file.write_text(script)
            script = script_file
            script.t

        self.script = script
        self.name = name
        self.t = t
        self.n_nodes = n_nodes
        self.partition = 'test' if pt else 'compute'
        self.cwd = cwd
        self.interpreter = cur_ser.e_hce7bp if script.s.endswith('py') else cur_ser.bash_exe

    @property
    def status(self):
        sqdf_ = sqdf()
        if self.job not in sqdf().index:
            return 0
        else:
            sqdf_time = sqdf_.TIME.values[0]
            if (sqdf_time == '0:00'):
                return 'queued'
            else:
                return sqdf_time

    def wr(self, db=0):
        rl = []
        for dir_i in range(self.n_nodes):
            rl.append({
                'name': f'name_{dir_i}',
                'nodes': 1,
                'path': self.cwd.s,
                'command': f'{self.interpreter} {self.script} {dir_i}',
                'out': self.cwd.joinpath(f'{dir_i}.out').s,
                'err': self.cwd.joinpath(f'{dir_i}.err').s,
            })
        self.run_dir = cur_ser.d_u.joinpath('.run/run_files')
        self.run_dir.mkdir()
        self.run_file = gen_file(self.run_dir)
        self.run_file.write_text(json.dumps(rl, indent=4))
        if db:
            self.run_file.t

    def run(self):
        p = run(
            f'sbatch -N {self.n_nodes} -J rst_{self.name} -t {self.t} -p {self.partition} {cur_ser.run_mult_exe} -t {self.run_file}')
        self.job = int(p.stdout.strip().split()[-1])

    def cancel(self):
        run(f'scancel {self.job}')


@for_all_methods(status)
class HD(P):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, path):
        super().__init__(path)
        self.sdir = self.joinpath('str')
        self.mp = self.sdir['mp']
        self.sdir = self.joinpath('str')
        self.spores_dir = self.joinpath('spores')
        self.mp = self.sdir['mp']
        # self.ES_pdb = self.sdir['ES.pdb']
        self.X_pdb = self.sdir['X.pdb']
        self.X_mol2 = self.sdir['X.mol2']
        self.XH_pdb = self.sdir['XH.pdb']
        self.S_pdb = self.sdir['S.pdb']
        self.E_pdb = self.sdir['E.pdb']



    def launch(self):
        t = self
        t._dir_initialize()
        t._lig_prot_initialize()
        t.run_spores(db=0)
        t.run_mopac()
        t.prepare_PL_0__1()
        t['PL_0__1'].launch()

        # t.write_plants_config()
        # t.plants.run_plants(wait=1, db=1)
        # t.plants.dirs[0].ls
        # # t.plants.dirs[0].make_visu()
        # # t.plants.dirs[0].load_visu()
        # t.plants.select_plants_results()

    def prepare_PL_0__1(self):
        kid = self['PL_0__1']
        kid.initialize()
        self.spores_dir['sporesed_prot.mol2'].copy_to(kid['str/prot.mol2'])
        self.mp['lig.mol2'].copy_to(kid['str/lig.mol2'])
        lines = kid['str/lig.mol2'].read_text().split('\n')
        lines[1] = lines[1].split('/')[-1]
        kid['str/lig.mol2'].write_text('\n'.join(lines))


    @property
    def hrx(self):
        if not self.kid:
            return
        return self.kid.kid
        # j1 = 1 if l2 else 0
        # j2 = 1 if l2 else 1
        # return self.plants.dirs[j1].hrxs[j2]

    def run1(self):
        t = self
        t._dir_initialize()
        t._lig_prot_initialize()
        t.run_spores(db=0) # ???????? only for protein. For ligand we use different methods.

        pl = t.kid

        pl.write_config()
        pl.run_plants()

        #unofficial

        t.plants.dirs[0].run_plants(wait=1, db=1)
        # t.plants.dirs[0].make_visu()
        # t.plants.dirs[0].load_visu()
        pl.select_plants_results()

        t.hrx.prepare()
        t.hrx.run_hrx()
        # t.hrx.make_cv_table()
        # t.hrx.make_visu_hrx_1() # bad code
        t.make_min_cluster_centroids()
        t.hrx.run_eds()



    def _dir_initialize(self):
        self.mkdir()
        self.spores_dir.mkdir()
        self.sdir.mkdir()
        self.mp.mkdir()

    # @status
    # def _lig_prot_initialize_old(self, make_ES_pdb=0, make_X_pdb=0):
    #     # --------------------------------------TEST STUFF-------------------------
    #     if make_ES_pdb: # WON'T BE IN PRODUCTION
    #         if ln == "rrf":
    #             pm('fetch 3a7r', f'save {args.ES_pdb}')
    #         if ln == 'L_PI35P':
    #             pm('fetch 1xww', 'save {sdir["L_PI35P.pdb"]}, r. 3PI', db=1)

    #     if make_X_pdb:# WON'T BE IN PRODUCTION
    #         if ln == "rrf":
    #             pm('fetch 3a7r', 'remove not polymer', 'fetch 4tvw', 'remove 4tvw & ! c. A', 'super 4tvw, 3a7r', 'remove (4tvw &! org)',
    #             'select sulfur, (r. 37P and e. S)', 'select nitr, sulfur around 2 and e. N', 'alter nitr, name="O33"',
    #             'alter nitr, elem="O"',
    #             'alter sulfur, name="P"',
    #             'alter sulfur, elem="P"',
    #             'alter org, resn="UNL"',
    #             f'save {sdir.joinpath("RRF.pdb")}, org',
    #             )
    #         if ln == 'L_PI35P':
    #             pm('fetch 1zvr', 'save ')
    #     # -------------------------------------------------------------------------
    #     if args.X_pdb: # Setting X, method #1
    #         assert len(args.X_cat_atoms_pdb_names) == len(args.S_cat_atoms_pdb_names)
    #         P(args.X_pdb).copy_to(self.X_pdb)
    #         pm(f'load {self.X_pdb}', f'h_add', f'save {self.XH_pdb}')
    #         # -------------------------------------FINALLY-------------------------
    #         self.X_cat_atoms_pdb_names = args.X_cat_atoms_pdb_names
    #         # ---------------------------------------------------------------------
    #     elif args.X_smiles: # Setting X, method #2
    #         X = Chem.MolFromSmiles(args.X_smiles)
    #         X = Chem.AddHs(X)
    #         AllChem.EmbedMolecule(X)
    #         wr = Chem.rdmolfiles.PDBWriter(self.XH_pdb.s)
    #         wr.write(X)
    #         u = mda.Universe(self.XH_pdb.s)
    #         # -------------------------------------FINALLY-------------------------
    #         self.X_cat_atoms_pdb_names = [u.residues[0].atoms[i].name for i in args.X_cat_atoms_ranks]
    #         # ---------------------------------------------------------------------
    #     P(args.ES_pdb).copy_to(self.ES_pdb)
    #     u = mda.Universe(self.ES_pdb.s)
    #     ligand_residues = []
    #     for residue in u.residues:
    #         common_atom_names =  set(args.S_cat_atoms_pdb_names) & set([k.name for k in residue.atoms])
    #         if len(common_atom_names) == 2:
    #             ligand_residues.append(residue)
    #     if len(ligand_residues) == 0:
    #         raise ValueError('No ligand found in input structure: {args.ES_pdb}\
    #         matching ligand atom names: {args.S_cat_atoms_pdb_names}')
    #     if len(ligand_residues) > 1:
    #         raise ValueError('More than 1 ligands found in input structure: {args.ES_pdb}\
    #         matching ligand atom names: {args.S_cat_atoms_pdb_names}')
    #     ligand_residue = ligand_residues[0]
    #     ligand_residue.resname = 'UNL'
    #     ligand_residue.atoms.write(self.S_pdb.s)
    #     prot = u.select_atoms(f'protein')
    #     prot.write(self.E_pdb.s)
    #     self.S_cat_atoms_pdb_names = args.S_cat_atoms_pdb_names
    #     # -------------------------------------FINALLY-------------------------
    #     # We have:
    #     # ES_pdb
    #     # E_pdb
    #     # S_pdb, S_cat_atoms_pdb_names
    #     # X_pdb, X_cat_atoms_pdb_names
    #     # XH_pdb
    #     # ---------------------------------------------------------------------



    def _lig_prot_initialize(self, make_ES_pdb=0, make_X_pdb=0):
        # --------------------------------------TEST STUFF-------------------------

        if make_ES_pdb: # WON'T BE IN PRODUCTION
            if ln == 'L_PI35P':
                pm('fetch 1xww', f'save {args.E_pdb}, polymer')

        if make_X_pdb:# WON'T BE IN PRODUCTION
            if ln == 'L_PI35P':
                pm('fetch 1zvr', f'save {d__sdir["L_PI35P.pdb"].s}, r. 3PI')
                pmr(); d__sdir["L_PI35P.pdb"].l
                # manually remove like in ptp.org (only etyl group)
                pma('h_add', f'save {args.XH_pdb}, r. 3PI')
                args.XH_pdb.t

        self.wrv(args.lit_at_pairs, "lit_at_pairs")
        args.XH_pdb.copy_to(self.XH_pdb)
        args.E_pdb.copy_to(self.E_pdb)


        X_u = mda.Universe(self.XH_pdb.s)
        X_u.resname = 'UNL'
        X_u.atoms.write(self.XH_pdb.s)

        # appending XH atom indices to lit_at_pairs
        t = self
        t.wrv(args.lit_at_pairs, 'lit_at_pairs')
        XH_pdb_u = mda.Universe(t.XH_pdb.s)
        lit_at_pairs_new = []
        lit_at_pairs = t.rdv('lit_at_pairs')
        # Last item in cat_.._at_lst will be the atom number
        for cat_E_at_lst, cat_X_at_lst in lit_at_pairs:
            cat_X_at_name = cat_X_at_lst[1]
            cat_X_at_inferred_id = [at.id for at in XH_pdb_u.atoms if at.name == cat_X_at_name][0]
            cat_X_at_lst.append(cat_X_at_inferred_id)
            lit_at_pairs_new.append([cat_E_at_lst, cat_X_at_lst])
        t.wrv(lit_at_pairs_new, 'lit_at_pairs')
        # t.rdv('lit_at_pairs')
        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------
        # if args.X_pdb: # Setting X, method #1
        #     assert len(args.X_cat_atoms_pdb_names) == len(args.S_cat_atoms_pdb_names)
        #     P(args.X_pdb).copy_to(self.X_pdb)
        #     pm(f'load {self.X_pdb}', f'h_add', f'save {self.XH_pdb}')
        #     # -------------------------------------FINALLY-------------------------
        #     self.X_cat_atoms_pdb_names = args.X_cat_atoms_pdb_names
        #     # ---------------------------------------------------------------------
        # elif args.X_smiles: # Setting X, method #2
        #     X = Chem.MolFromSmiles(args.X_smiles)
        #     X = Chem.AddHs(X)
        #     AllChem.EmbedMolecule(X)
        #     wr = Chem.rdmolfiles.PDBWriter(self.XH_pdb.s)
        #     wr.write(X)
        #     u = mda.Universe(self.XH_pdb.s)
        #     # -------------------------------------FINALLY-------------------------
        #     self.X_cat_atoms_pdb_names = [u.residues[0].atoms[i].name for i in args.X_cat_atoms_ranks]
        #     # ---------------------------------------------------------------------
        # P(args.ES_pdb).copy_to(self.ES_pdb)
        # u = mda.Universe(self.ES_pdb.s)
        # ligand_residues = []
        # for residue in u.residues:
        #     common_atom_names =  set(args.S_cat_atoms_pdb_names) & set([k.name for k in residue.atoms])
        #     if len(common_atom_names) == 2:
        #         ligand_residues.append(residue)
        # if len(ligand_residues) == 0:
        #     raise ValueError('No ligand found in input structure: {args.ES_pdb}\
        #     matching ligand atom names: {args.S_cat_atoms_pdb_names}')
        # if len(ligand_residues) > 1:
        #     raise ValueError('More than 1 ligands found in input structure: {args.ES_pdb}\
        #     matching ligand atom names: {args.S_cat_atoms_pdb_names}')
        # ligand_residue = ligand_residues[0]
        # ligand_residue.resname = 'UNL'
        # ligand_residue.atoms.write(self.S_pdb.s)
        # prot = u.select_atoms(f'protein')
        # prot.write(self.E_pdb.s)
        # self.S_cat_atoms_pdb_names = args.S_cat_atoms_pdb_names

    def run_RED(self):
        t.j('red').rwdir()
        # ante_red_input = output_babel_h_deh
        ante_red_input = t.mp['lig.H.mop.pdb']
        # ante_red_input = t.sdir['lig.H.pdb']
        ante_red_input = ante_red_input.copy_to(t['red'])

        t.j('red').run(f'{cur_ser.e__ante_red_exe} {ante_red_input.name}')

        red_exe = cur_ser.e__red_exe.copy_to(t['red'])
        text = red_exe.read_text()
        # text = text.replace('$NP     = "12";', '$NP     = "24";')
        text = text.replace('$CHR_TYP = "RESP-A1";', '$CHR_TYP = "RESP-A2";')
        text = text.replace('MWORDS=32', 'MWORDS=512')
        red_exe.write_text(text)

        p2n = t['red/lig-out.p2n']
        p2n_text = p2n.read_text()
        t[f'red/Mol_red1.p2n'].write_text(p2n_text.replace(
        'REMARK CHARGE-VALUE 0',
        f'REMARK CHARGE-VALUE {charge}').replace(
        'REMARK MULTIPLICITY-VALUE 1',
        f'REMARK MULTIPLICITY-VALUE {multiplicity}'))

        scr_dir = P('/mnt/scr/rustam')
        scr_dir.rwdir()

        t['red'].trun(f'{red_exe}', sn=args.root_name, wn=f'red_{ln}')
        # arak = P('/home/domain/data/rustam/A1/rrf/red/lig-out.pdb') # maybe I wanted to use it later as pdb in plants ..?


    def babel_hydrogenate(self):
        prop = {
            'gmx_path': args.gmx_path,
            'input_format': 'pdb',
            'output_format': 'pdb',
            'ph': pH
        }
        BabelAddHydrogens(input_path=self.sdir['lig.pdb'].s,
                          output_path=self.sdir['lig.H.pdb'].s,
                          properties=prop).launch()

    def remove_hydroxyl_hydrogen(self):
        self.sdir['lig.H.pdb'].make_res('with_hidroxyl')
        u = mda.Universe(self.sdir['lig.H.pdb'].s)
        hydroxyl_oxygen = u.select_atoms(f'name {args.hydroxyl_name}')

        hydroxyl_hydrogen = None
        for bond in hydroxyl_oxygen.bonds:
            for bond_atom in bond.atoms:
                if bond_atom.type == 'H':
                    hydroxyl_hydrogen = bond_atom
        if not hydroxyl_hydrogen:
            raise ('Unable no hydroxyl hydrogen!')
        deh = u.select_atoms(f"not index {hydroxyl_hydrogen.index}")
        deh.write(self.sdir['lig.H.pdb'])

    def babel_opt(self):
        from biobb_chemistry.babelm.babel_minimize import BabelMinimize
        prop = {
            'gmx_path': args.gmx_path,
            'method': 'sd',
            'criteria': '1e-200',
            'force_field': 'GAFF'
        }

        # Create and launch bb
        BabelMinimize(input_path=self.sdir['lig.H.pdb'].s,
                      output_path=self.sdir['lig.H.opt.pdb'].s,
                      properties=prop).launch()

    def run_mopac(self):
        t = self
        t.prepare_mop()
        t.mop_opt()
        t.mop_out_to_pdb()
        t.mop_out_to_mol2() # temporarily (without RED)

    def prepare_mop(self, db=0):
        import pybel
        # pybel_molecule = [*pybel.readfile('pdb', self.sdir['lig.H.pdb'].s)][0]
        pybel_molecule = [*pybel.readfile('pdb', self.XH_pdb.s)][0]
        assert pybel_molecule, 'corrupted / empty pdb file: ' + self.XH_pdb.s
        pybel_molecule.write(
            format='mop', filename=self.mp['lig.mop'].s,
            opt={'k': f'PM6 CHARGE={args.net_charge} PRECISE pKa EF THREADS=14'},
            overwrite=True
        )

        # lines = self.mp['lig.mop'].read_text().split('\n')
        # new_lines = []
        # for i, line in enumerate(lines):
        #     splt = line.split()
        #     if len(splt) == 7:
        #         splt[2], splt[4], splt[6] = '0', '0', '0'
        #         new_lines.append('  '.join(splt))
        #     else:
        #         new_lines.append(line)
        # self.mp['lig.mop'].write_text('\n'.join(new_lines))


    def mop_opt(self):
        my_env = os.environ.copy()
        my_env.update(cur_ser.a_mopac_env_vars)
        self.mp.run(f'{cur_ser.e__mopac_exe} {self.mp["lig.mop"]}', env=my_env)

    def mop_out_to_pdb(self):
        mop_out = [*pybel.readfile('mopout', self.mp['lig.out'].s)][0]
        mop_out.write('pdb', self.mp['lig.H.mop.pdb'].s, overwrite=True)

    def mop_out_to_mol2(self):
        mop_out = [*pybel.readfile('mopout', self.mp.joinpath('lig.out').str)][0]
        mop_out.write('mol2', self.mp.joinpath('lig-unl1.mol2').s, overwrite=True)
        mol2u = mda.Universe(self.mp.joinpath('lig-unl1.mol2').s)
        mol2u.residues[0].resname = 'UNL'
        mol2u.atoms.write(self.mp.joinpath('lig.mol2').str)

    def _get_lig_center_str(self):
        # u = mda.Universe(self.ES_pdb.s)
        # lig = u.select_atoms(f'resname {args.lig_name}')
        # self.lig_center = lig.center_of_mass()
        # self.lig_center_str = '   '.join([*map(str, self.lig_center)])

        E_u = mda.Universe(t.E_pdb.s)
        lit_at_pairs = t.rdv('lit_at_pairs')
        lit_E_cat_at_resids = [str(i[0][0]) for i in lit_at_pairs]
        lit_E_cat_at_resids_str = " ".join(lit_E_cat_at_resids)
        self.lig_center = E_u.select_atoms(f'resid {lit_E_cat_at_resids_str}').center_of_mass()
        self.lig_center_str = '   '.join([*map(str, self.lig_center)])
        return self.lig_center_str


    def _get_cat_atom_id(self):
        u = mda.Universe(self.XH_pdb.s)
        carbon_atom_name = args.X_cat_atoms_pdb_names[0]  # 'C15'
        cat_atom = u.select_atoms(f'name {carbon_atom_name}')
        return int(cat_atom._ix) + 1

    def run_spores(self, db=0):
        self.spores_dir.rwdir()
        # self.sdir.joinpath("lig.pdb").copy_to(self.spores_dir)
        self.E_pdb.copy_to(self.spores_dir)
        # self.spores_dir.run(f'{cur_ser.e__spores_exe} --mode complete lig.pdb sporesed_lig.mol2')
        self.spores_dir.run(f'{cur_ser.e__spores_exe} --mode complete E.pdb sporesed_prot.mol2')

        # appending E atom indices to lit_at_pairs
        t = self
        E_pdb_u = mda.Universe(self.spores_dir['sporesed_prot.mol2'].s)
        lit_at_pairs_new = []
        lit_at_pairs = t.rdv('lit_at_pairs')
        # Last item in cat_.._at_lst will be the atom number
        for cat_E_at_lst, cat_X_at_lst in lit_at_pairs:
            cat_E_resid = cat_E_at_lst[0]
            cat_E_at_name = cat_E_at_lst[1]

            cat_E_res = [res for res in E_pdb_u.residues if res.resid == cat_E_resid][0]
            cat_E_at_inferred_id = [at.id for at in cat_E_res.atoms if at.name == cat_E_at_name][0]
            cat_E_at_lst.append(cat_E_at_inferred_id)
            lit_at_pairs_new.append([cat_E_at_lst, cat_X_at_lst])
        t.wrv(lit_at_pairs_new, 'lit_at_pairs')



class Pld(P):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, path, pl=None, db=0):
        super().__init__(path)
        self.out_dir_name = self.name
        self.pl = pl
        self.pse = self.replace_suffix('pse')

    def make_visu(self):
        self.par.make_visu(spec=self.name)

    def load_visu(self):
        print_(f'Loading {self.pse} ... ', end="")
        pmal(self.pse)
        print_(f'\rLoaded: {self.pse}          !')

    @property
    def mol2s_all(self):
        # fs = [P(i) for i in self.rglob(f'*entry_*_conf_*.mol2')]
        fs = [P(i) for i in self.rglob(f'lig*.mol2')]
        fs.sort()
        return fs

    @property
    def mol2s_protein(self):
        fs = [P(i) for i in self.rglob(f'lig*_conf_*_protein.mol2')]
        fs.sort()
        return fs

    @property
    def mol2s(self):
        fs = [*set(self.mol2s_all) - set(self.mol2s_protein)]
        fs.sort()
        return fs

    @property
    def hrxs(self):
        return [i for i in self.dirs if i.name.startswith('HRX')]

    # def get_ranking(self):
    #     import csv
    #     import pandas as pd
    #     # make dict for multiple mol2s!!!
    #     csv_reader = csv.reader(open(self['ranking.csv'].s, 'r'))
    #     np_array = np.array([*csv_reader])
    #     scores_df = pd.DataFrame(np_array[1:, 1:], columns=np_array[0, 1:], index=np_array[1:, 0])
    #     return(scores_df)

    # def get_ranking_tot_score(self):
    #     scores_df = self.get_ranking()
    #     tot_scores = scores_df['TOTAL_SCORE']
    #     return tot_scores

    # def get_ranking_array(self):
    #     ranking_array = np.array(self.get_ranking[1:, 1], dtype=float)
    #     return ranking_array

    def get_ranking(self):
        import csv
        # make dict for multiple mol2s!!!
        if self['ranking.csv'].exists():
            csv_reader = csv.reader(open(self['ranking.csv'].s, 'r'))
            np_array = np.array([*csv_reader])
            return(np_array)
        else:
            return np.array([])

    def get_ranking_tot_score(self):
        np_array = self.get_ranking()
        if np_array.shape == (11,11):
            tot_scores = np.array(np_array[1:, 1], dtype=float)
            return tot_scores
        else:
            return np.array([])

    # def get_ranking_array(self):
    #     ranking_array = np.array(self.get_ranking[1:, 1], dtype=float)
    #     return ranking_array

    def ar(self):
        print("kjkj")


@for_all_methods(status)
class Pl(P):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, path):
        super().__init__(path)

    @property
    def pld(self):
        self.plds[0]

    @property
    def plds(self):
        return [i for i in self['plds'].dirs if type(i) == Pld]


    def initialize(self):
        self.mkdir()
        self['str'].mkdir()


    def launch(self):
        self.write_config() # writes a lot of configs (2 000) for sigmoids
        self.run_plants()
        # self.pld.make_visu()
        self.analyze_plants_results_PL__1()
        self.exhaust_sigmoid_kids()

    def if_convergence_reached(self):
        # TODO write this function
        # kids_st_4 = self.kids_n(4)
        return False

    def exhaust_sigmoid_kids(self):
        self.wrv([], 'queue')
        # self.rm_kids()
        while True:
            kid = self.prepare_next_kid()
            convergence_reached_p = self.if_convergence_reached()
            if convergence_reached_p:
                print('SSDO succeeded! Find your results in folder ...?')
            elif kid:
                kid.launch2queue()
                self.exhaust_queue()
            else:
                print('SSDO failed: all docking results exhausted, but convergence not reached!')
            break
    def analyze_plants_results_PL__1(self):
        total_scores = []
        # it = t.kid.rdv('out_dir_names') # TODO
        # temporary code:
        it = self['plds'].rglob('D1-*')
        # it = self.plds
        for iti_i, iti in enumerate(it):
            try:
                iti = Pld(iti)
                total_scores_i = np.array(iti.get_ranking_tot_score(), dtype=object)
                total_scores_i
                if len(total_scores_i):
                    arr = np.array(np.insert(total_scores_i, 0, iti.name))
                    total_scores.append(arr)
            except Exception as e:
                print(str(e))
                pass
        total_scores = np.vstack(total_scores)

        ar2 = []
        for i1 in total_scores:
            for i3, i2 in enumerate(i1[1:]):
                ar2.append([i1[0], int(i3), float(i2)])
        tot_scores_sorted = np.array(ar2, dtype=object)
        tot_scores_sorted = tot_scores_sorted[tot_scores_sorted[:, 2].argsort()]
        self.wrv(total_scores, 'total_scores')
        self.wrv(tot_scores_sorted, 'tot_scores_sorted')

        total_scores_all = np.array(tot_scores_sorted[:, 2], dtype=float)
        self.plt_plot(total_scores_all, 'total_scores_all', "Номер результата докинга", "Скор программы PLANTS", "Результаты PLANTS, cluster_structures=10, ~2 000 запусков")
        self.plt_savefig()



    def prepare_next_kid(self):
        # TODO do it as it must be done

        # Creating kid:
        idxs = [kid.idx for kid in self.kids]
        dir_idx = max(idxs) + 1 if idxs else 0
        kid = self[f'HR_{dir_idx}__2']
        kid.initialize()

        # Picking and writing coordinates in total_scores array
        import random as rnd
        sigmoid_segment_xs = [0, 500]
        rnd_sigmoid_x = rnd.randint(*sigmoid_segment_xs)
        tot_scores_sorted = self.rdv('tot_scores_sorted')
        rnd_sigmoid_Y = tot_scores_sorted[rnd_sigmoid_x]
        kid.wrv(rnd_sigmoid_Y, 'rnd_sigmoid_Y')
        # TODO make through set() - ...
        # TODO clusterstructure to avoid duplicates
        self.rdv('tot_scores_sorted')

        # BAD CODE - only one mol2hrx
        mol2hrx = self['plds'][rnd_sigmoid_Y[0]].mol2s[rnd_sigmoid_Y[1]]
        idx = mol2hrx.name_prefix.split('_')[-1].lstrip('0')  # '01'
        par_name = mol2hrx.parent.name  # '4_constraints_minus_100_10'
        mol2hrx.copy_to(kid['lig.mol2'])
        self.root.E_pdb.copy_to(kid)

        return(kid)


    def generate_literature_cat_at_constraints(self):
        lit_at_pairs = t.rdv('lit_at_pairs')
        prot_lig_constr_str = '\n'
        for lit_E_at_lst, lit_X_at_lst in lit_at_pairs:
            prot_lig_constr_str += f'protein_ligand_distance_constraint 0.0 3.5 -500.0 {lit_E_at_lst[-1]} {lit_X_at_lst[-1]}\n'
        print('generated constraints: \n' + prot_lig_constr_str)
        return prot_lig_constr_str

    def write_config(self, **kwargs):
        # TODO relaunch with 1 constraint (P) - avoid desinformation in ssdo.org file
        # TODO write batch configs for sigmoid
        t = self
        if ln == 'L_PI35P':
            literature_cat_at_constraints = self.generate_literature_cat_at_constraints()
            self.write_config_2('pld_results_c_A', literature_cat_at_constraints)
    def write_config_2(self, out_dir_name='results', addition='', flex_prot_side_chain_numbers=[], n_struct=10, db=0, auto_name=True, flush_out_dir_names=True):
        # Batch development:
        # out_dir_names = self.rdv('out_dir_names', if_not_exists=[])
        # Single run

        flex_list = [f'flexible_protein_side_chain_number {i}' for i in flex_prot_side_chain_numbers]
        flex_sc = '\n'.join(flex_list)
        flex_sc = ''
        if flush_out_dir_names:
            out_dir_names = []
        else:
            out_dir_names = self.rdv('out_dir_names', if_not_exists=[])
        if not auto_name:
            out_dir_name += f'_n_{n_struct}_flex_{len(flex_list)}'
        lig_center_str = self.par._get_lig_center_str()
        # fixed_scaffold = self.par._get_cat_atom_id()
        # plants_conf_file = self.joinpath(f"plantsconfig_{out_dir_name}")
        plants_conf_file = self[f'plants_configs/plantsconfig_{out_dir_name}']
        # Removed lines:
        # ligand_file {self['str/lig.mol2']} fixed_scaffold_{fixed_scaffold}
        plants_conf_file.write_text(textwrap.dedent(f"""
            protein_file  {self['str/prot.mol2']}
            ligand_file {self['str/lig.mol2']}
            # flexible side-chains
            {flex_sc}
            # output
            output_dir {out_dir_name}
            # write single mol2 files (e.g. for RMSD calculation)
            write_multi_mol2 0
            # binding site definition
            bindingsite_center  {lig_center_str}
            bindingsite_radius 30
            search_speed {args.plants_speed}
            # cluster algorithm
            cluster_structures {n_struct}
            cluster_rmsd 2.0""" + addition))

        out_dir_names.append(out_dir_name)
        if db:
            plants_conf_file.t
        self.wrv(out_dir_names, 'out_dir_names')
        # vd(out_dir_names)

    def run_plants(self, db=1, wait=0, b=0, e=100000000):
        from multiprocessing import Pool
        self.cd
        out_dir_names = self.rdv('out_dir_names')
        print(f'out_dir_names ({len(out_dir_names)}) :', out_dir_names)
        ps = {}
        for out_dir_name in out_dir_names[b:e]:
            print(f'Running plants for out_dir_name: {out_dir_name}:')
            self.joinpath(out_dir_name).rm()
            cmd_ = f"{cur_ser.e__plants_exe} --mode screen {self.joinpath(f'plants_configs/plantsconfig_{out_dir_name}')}"

            if db:
                self.run(cmd_)
            else:
                proc = subprocess.Popen(
                    cmd_,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=isinstance(cmd_, str),
                    cwd=self['plds'],
                )
                ps[out_dir_name] = proc
        if not db:
            while any([i.poll() == None for i in ps.values()]) and wait:
                printr([[i, j.poll()] for i, j in ps.items()])
                sl(3)

    def run_plants_lom2(self, *args, db=0, **kwargs):
        code = textwrap.dedent("    from lib import *\n" + inspect.getsource(Pl.run))
        lom2_script = self.joinpath('run_lom2.py')
        lom2_script.write_text(code)
        if db: print_(lom2_script.read_text())
        kwargs['cwd'] = self
        sb = L2p(lom2_script, *args, **kwargs)
        sb.wr()
        sb.run()
        return sb

    def make_visu(self, spec=None):
        if isinstance(spec, str):
            specs = [spec]
        elif isinstance(spec, list):
            specs = spec
        else:
            specs = self.rdv('out_dir_names')
        specs = [Pld(self['plds'][i], pl=self) for i in specs][:1]
        for out_dir in specs:
            t0 = time.time()
            print(out_dir)
            while not out_dir.mol2s[:1]:
                sl(1)
                dt = datetime.timedelta(seconds=time.time() - t0)
                print_(f'\rWaiting for {dt}', end='')
            print(f'Detected {len(out_dir.mol2s)} mol2s')
            # pm(f'load {self.par.ES_pdb}')
            # pm(f'load {self.par.E_pdb}')
            pm(f'load {self["str/prot.mol2"]}')
            for res in out_dir.mol2s:
                pm(f'load {res}', r=False)
            # pm(r=False)
            # # VISU CONTACTS:
            cmds_pml = []
            for i, (cat_E_at_lst, cat_X_at_lst) in enumerate(t.rdv('lit_at_pairs')):
                cmds_pml += [
                    f'select pair_{i}_prot, id {cat_E_at_lst[-1]} & polymer',
                    f'select pair_{i}_lig,  id {cat_X_at_lst[-1]} & org',
                    f'distance dist_{i}, pair_{i}_prot, pair_{i}_lig',
                    f'select pair_{i}, pair_{i}_prot | pair_{i}_lig',
                    f'label pair_{i}, "%s-%s-%s" % (name, resn, resi)' ]
            cmds_pml += ['hide cart', 'show lines']
            pm(*cmds_pml, r=False)
            # # VISU CONTACTS:
            pm(f'save {out_dir.pse.s}', r=False)
            # decVisu(pmName=out_dir.pse.s, repr="sticks")

    def select_plants_results(self):
        mol2hrxs = t.PL.dirs[0].select_mol2hrxs()
        t.PL.dirs[0].make_hrxs(mol2hrxs)

    def exhaust_queue(self):
        while True:
            queue = self.rdv('queue', if_not_exists=[])
            # filtering queue
            d = defaultdict(list)
            filtered_queue = []
            for item in queue:
                d['|{}|__|{}|'.format(item['dir_str'], item['fun_str'])].append(item)
            for key, value in d.items():
                value.sort(key=lambda i: i['time_registered'], reverse=True)
                most_recent_entry = value[0]
                filtered_queue.append(most_recent_entry)
                # filtered_queue.sort(reverse=True, key=lambda i: i['stage'])

            from functools import cmp_to_key
            filtered_queue = sorted(filtered_queue,key=cmp_to_key(ssdo_nodes_coords_compare))

            if not filtered_queue:
                print('queue finished, proceeding with next sigmoid_kid')
                return
            item = filtered_queue.pop()
            self.wrv(filtered_queue, 'queue')
            self.wrv(item, 'queue_popped_item')
            stage, id_, dir_str, fun_str = item['stage'], item['id_'], item['dir_str'], item['fun_str']
            id_real = os.stat(dir_str).st_ino
            if id_real != id_:
                print(f'\
                Outdated queue record : \
                id_real ({id_real}) != id_ ({id_}) \
                stage, id_, dir_str, fun_str = {item}\
                ')
                self.wrv(None, 'queue_popped_item')
                continue
            dir_ = P(dir_str)
            getattr(dir_, fun_str)()
            break

def mutate_protein(modelname, respos, restyp, chain):
    mutate_model_py = cur_ser.d__lb37['mutate_model.py']
    self.run(f'{cur_ser.e_hce7bp} {mutate_model_py} {modelname} {respos} {restyp} {chain}')

class Mutate_model:
    def __init__(self, modelname, respos, restyp, chain):
        # %%writefile /home/domain/data/rustam/tools/mutate_model.py
        # modelname, respos, restyp = 'prot_mutated', 20, 'ALA'
        import sys
        import os

        from modeller import group_restraints
        from modeller.optimizers import molecular_dynamics, conjugate_gradients
        from modeller.automodel import autosched

        #
        #  mutate_model.py
        #
        #     Usage:   python mutate_model.py modelname respos resname chain > logfile
        #
        #     Example: python mutate_model.py 1t29 1699 LEU A > 1t29.log
        #
        #
        #  Creates a single in silico point mutation to sidechain type and at residue position
        #  input by the user, in the structure whose file is modelname.pdb
        #  The conformation of the mutant sidechain is optimized by conjugate gradient and
        #  refined using some MD.
        #
        #  Note: if the model has no chain identifier, specify "" for the chain argument.
        #

        def optimize(atmsel, sched):
            # conjugate gradient
            for step in sched:
                step.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)
            # md
            refine(atmsel)
            cg = conjugate_gradients()
            cg.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)

        # molecular dynamics
        def refine(atmsel):
            # at T=1000, max_atom_shift for 4fs is cca 0.15 A.
            md = molecular_dynamics(cap_atom_shift=0.39, md_time_step=4.0,
                                    md_return='FINAL')
            init_vel = True
            for (its, equil, temps) in ((200, 20, (150.0, 250.0, 400.0, 700.0, 1000.0)),
                                        (200, 600,
                                         (1000.0, 800.0, 600.0, 500.0, 400.0, 300.0))):
                for temp in temps:
                    md.optimize(atmsel, init_velocities=init_vel, temperature=temp,
                                max_iterations=its, equilibrate=equil)
                    init_vel = False

        # use homologs and dihedral library for dihedral angle restraints
        def make_restraints(mdl1, aln):
            rsr = mdl1.restraints
            rsr.clear()
            s = selection(mdl1)
            for typ in ('stereo', 'phi-psi_binormal'):
                rsr.make(s, restraint_type=typ, aln=aln, spline_on_site=True)
            for typ in ('omega', 'chi1', 'chi2', 'chi3', 'chi4'):
                rsr.make(s, restraint_type=typ + '_dihedral', spline_range=4.0,
                         spline_dx=0.3, spline_min_points=5, aln=aln,
                         spline_on_site=True)

        # first argument
        # modelname, respos, restyp, chain, = sys.argv[1:]

        # log.verbose()

        # Set a different value for rand_seed to get a different final model
        env = environ(rand_seed=-49837)

        env.io.hetatm = True
        # soft sphere potential
        env.edat.dynamic_sphere = False
        # lennard-jones potential (more accurate)
        env.edat.dynamic_lennard = True
        env.edat.contact_shell = 4.0
        env.edat.update_dynamic = 0.39

        # Read customized topology file with phosphoserines (or standard one)
        env.libs.topology.read(file='$(LIB)/top_heav.lib')

        # Read customized CHARMM parameter library with phosphoserines (or standard one)
        env.libs.parameters.read(file='$(LIB)/par.lib')

        # Read the original PDB file and copy its sequence to the alignment array:
        mdl1 = model(env, file=modelname)
        ali = alignment(env)
        ali.append_model(mdl1, atom_files=modelname, align_codes=modelname)

        # set up the mutate residue selection segment
        s = selection(mdl1.chains[chain].residues[respos])

        # perform the mutate residue operation
        s.mutate(residue_type=restyp)
        # get two copies of the sequence.  A modeller trick to get things set up
        ali.append_model(mdl1, align_codes=modelname)

        # Generate molecular topology for mutant
        mdl1.clear_topology()
        mdl1.generate_topology(ali[-1])

        # Transfer all the coordinates you can from the template native structure
        # to the mutant (this works even if the order of atoms in the native PDB
        # file is not standard):
        # here we are generating the model by reading the template coordinates
        mdl1.transfer_xyz(ali)

        # Build the remaining unknown coordinates
        mdl1.build(initialize_xyz=False, build_method='INTERNAL_COORDINATES')

        # yes model2 is the same file as model1.  It's a modeller trick.
        mdl2 = model(env, file=modelname)

        # required to do a transfer_res_numb
        # ali.append_model(mdl2, atom_files=modelname, align_codes=modelname)
        # transfers from "model 2" to "model 1"
        mdl1.res_num_from(mdl2, ali)

        # It is usually necessary to write the mutated sequence out and read it in
        # before proceeding, because not all sequence related information about MODEL
        # is changed by this command (e.g., internal coordinates, charges, and atom
        # types and radii are not updated).

        mdl1.write(file=modelname + restyp + respos + '.tmp')
        mdl1.read(file=modelname + restyp + respos + '.tmp')

        # set up restraints before computing energy
        # we do this a second time because the model has been written out and read in,
        # clearing the previously set restraints
        make_restraints(mdl1, ali)

        # a non-bonded pair has to have at least as many selected atoms
        mdl1.env.edat.nonbonded_sel_atoms = 1

        sched = autosched.loop.make_for_model(mdl1)

        # only optimize the selected residue (in first pass, just atoms in selected
        # residue, in second pass, include nonbonded neighboring atoms)
        # set up the mutate residue selection segment
        s = selection(mdl1.chains[chain].residues[respos])

        mdl1.restraints.unpick_all()
        mdl1.restraints.pick(s)

        s.energy()

        s.randomize_xyz(deviation=4.0)

        mdl1.env.edat.nonbonded_sel_atoms = 2
        optimize(s, sched)

        # feels environment (energy computed on pairs that have at least one member
        # in the selected)
        mdl1.env.edat.nonbonded_sel_atoms = 1
        optimize(s, sched)

        s.energy()
        # give a proper name
        print('writing: ' + modelname + '.pdb', modelname, respos, restyp, chain)
        mdl1.write(file=modelname + '.pdb')

        # delete the temporary file
        os.remove(modelname + restyp + respos + '.tmp')

def plt_plot(arr, name="", xlabel="", ylabel="", suptitle="", label="", scratch=1, p_dir=None):
    # arr = arr if len(arr) == 2 else arr[0]
    name = str(uuid.uuid4()) if not name else name
    if p_dir:
        name = f'{p_dir.name}_{name}'
    suptitle = name if not suptitle else suptitle
    plt.plot(arr, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(suptitle)
    plt.p_dir = p_dir
    if scratch:
        plt.i_f_n = d_ais[name + '.png']
    else:
        plt.i_f_n = d_ai[name + '.png']
    plt.show()
    return(plt.i_f_n)

def plt_savefig(name="", p_dir=None):
    name = plt.i_f_n if not name else name
    fig = plt.gcf()
    fig.savefig(name, dpi=300)
    d_ais[name].s

def parse_hist_like_data_from_plumed(file_path):
    file_path_split = file_path.read_text().split('\n')
    file_path_split_corred = [i for i in file_path_split if not i.startswith('#')]
    list_ = []
    list_all = []
    for line in file_path_split_corred:
        line_split = line.split();
        if len(line_split) == 0:
            list_all.append(list_)
            list_ = []
        else:
            arr = np.array([float(i) for i in line.split()])
            list_.append(arr)
    arr_all = np.array([np.array(j) for j in list_all])
    return(arr_all)

def ssdo_nodes_coords_compare(node1, node2): # True -> first has higher priority
    node1_coords = node1['coords']
    node2_coords = node2['coords']
    if len(node1_coords) != len(node2_coords):
        return len(node1_coords) > len(node2_coords) #????
    else:
        for coord_idx in range(len(node1_coords)):
            if node1_coords[coord_idx] > node2_coords[coord_idx]:
                return True # ????
        return True # ??

# QUICK AND DIRTY code:
if l2 and not 'get_ipython' in globals().keys():
    print("Importing lib at lom2: OK")
# t = P(cur_ser.d_a[ln])
t = P(cur_ser.d_a[ln])
# self = t.kid.kid.kid.kid
# self = t.kid.kid.kid.kid
# self = t.kid.kid
# self = t.kid.plds[0]
# self = t.kid
self  = t.kid.kid
# self  = t.kid.kid.kid
# self = P('/home/domain/data/rustam/A1/L_PI35P/PL_0__1/HR_14__2/')
# self = P('/home/domain/data/rustam/A1/L_PI35P/PL_0__1/HR_15__2/ED_52__3/HR_0__4')
# self = P('/home/domain/data/rustam/A1/L_PI35P/PL_0__1/HR_15__2/')
# self2 = P('/home/domain/data/rustam/A1/L_PI35P/PL_0__1/HR_14__2/')
# self = Pld('/home/domain/data/rustam/A1/L_PI35P/PL_0__1/D1-0-809_n_10_flex_0')
# self = glv
# self = t.kid.kid.chd1
# self = t.kid.kid
if t.exists():
    get_ipython().ex('t.cd')
    t.cd
for thing in dir(cur_ser):
    if thing.startswith("_"):
        continue
# /home/domain/data/share/ext/rustam/ssdo/
