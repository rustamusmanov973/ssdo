# print('base.py')
import os
l2 = os.popen('cat /etc/machine-id 2>/dev/null').read().strip() == '11a2e8e0953e4023ad8806b6d61669a7'
if l2 and not 'get_ipython' in globals().keys():
    # print(f'aAgent #{ag_num}  Importing lib...')
    print('not ipython')
    print('Importing lib at lom2...')
import sys

# sys.path = [i for i in sys.path if i != '/home/golovin/progs/lib/python2.7/site-packages']
sys.path = [i for i in sys.path if i not in [ '/home/golovin/progs/lib/python2.7/site-packages', '/home/golovin/progs/lib/python2.7/site-packages/argparse-1.4.0-py2.7.egg' ] ]

from multiprocessing import cpu_count
n_cpu_avail = cpu_count()
if l2 and not 'get_ipython' in globals().keys():
    print(f'Process on {n_cpu_avail} cpus.')

def print_fun(func):
    def decorated_function(*args, **kwargs):
        import time, datetime
        t0 = time.time()
        print(f'{func.__name__} started ...')
        result = func(*args, **kwargs)
        t1 = time.time()
        delta_t = datetime.timedelta(seconds=t1 - t0)
        print(f'{func.__name__} ended ...: {delta_t}')
        return result

    return decorated_function


def ta(session='ge'):
    os.system(f'tmux attach-session -t {session}')

import shutil
from pathlib import PosixPath

if 'get_ipython' in globals().keys():
    # get_ipython().run_cell_magic('HTML', '',
                                 # '<style>\ndiv.prompt {display:none}\n.container {width: 98%}\n</style>')


    def vd(var, var_desc=None, db=False):
        global task_complete, trace_vd_ind
        try:
            if task_complete:
                trace_vd_ind, task_complete = 0, False
            else:
                trace_vd_ind += 1
        except NameError:
            trace_vd_ind, task_complete = 0, False
        if db:
            note("INIT", "trace_vd_ind = {}".format(trace_vd_ind), "task_complete = {}".format(task_complete))
        trace_vd = traceback.format_stack()[-2]  # File "<ipython-input-41-376be27568c2>", line 4, in <module> \n vd(a)
        vd_expr = trace_vd.split("\n")[1]  # vd_expr =  print(vd(x)) or warn(vd(x)) or vd(x)
        # IT is impossible to do the 'vd' task correctly, so we decide just to assume that if "warn/note/print" words are in your line, THEN ALL vd must return values and not print(them to console)
        # bare_vd = vd_expr.strip().startswith("vd") or vd_expr.strip().startswith("if ")
        bare_vd = not re.search(r'(\bwarn\b|\bnote\b|\bprint\b)', vd_expr.strip())
        if db: note("TRACE PARSING", "trace_vd = {}".format(trace_vd), "vd_expr = {}".format(vd_expr),
                    "bare_vd = {}".format(bare_vd))
        try:
            task_len = len(re.findall(r'(?<=vd\().*?(?=\))', trace_vd))
            vd_args_str = re.findall(r'(?<=vd\().*?(?=\))', trace_vd)[
                trace_vd_ind]  # str_, "my str_ variable", "print(accurate please")
            vd_args = [j.strip('"') for j in vd_args_str.split(",")]  # ['str_', 'my str_ variable', 'print(accurate' ])
            var_name = vd_args[0]
            if db: note("TRY construction", "task_len = {}".format(task_len), "vd_args_str = {}".format(vd_args_str),
                        "vd_args = {}".format(vd_args), "var_name = ".format(var_name))
            if trace_vd_ind == task_len - 1:
                task_complete = True
        except Exception as e:
            task_complete = True
            warn("Unable to parse your traceback: %s" % trace_vd)
            warn("error occured: %s" % str(e))
            return
        callers_local_vars = inspect.currentframe().f_back.f_locals
        callers_global_vars = inspect.currentframe().f_back.f_globals
        if db: note("LOCALS AND GLOGALS OF THE PRECEDING FRAME", "callers_local_vars = {}".format(callers_local_vars),
                    "callers_global_vars = {}".format(callers_global_vars))
        if not var_desc: var_desc = var_name
        # long variables will be printed with \n JUST for convinence (to copy it)

        # search of the variable:
        try:
            lbreak = " " if len(str(callers_local_vars[var_name])) < 8 else "\n"
            var_dump = "{}:{}{}".format(var_desc, lbreak, callers_local_vars[var_name])
        except KeyError:
            try:
                lbreak = " " if len(str(callers_global_vars[var_name])) < 8 else "\n"
                var_dump = "{}:{}{}".format(var_desc, lbreak, callers_global_vars[var_name])
            except KeyError as e:
                warn("unable to find value of the variable {} in both locals and globals: {}".format(var_name, str(e)))
        if bare_vd:
            note(var_dump)
        else:
            return var_dump

    def note(*note_):
        note_ = [str(j).replace("\n", "<br>") for j in note_]
        if len(note_) == 1:
            display(HTML("""<p style='color: grey'> {} </p>""".format(note_[0])))
        else:
            html_string = """<div style="width: fit-content; padding: 7px; border: 2px solid green; border-radius: 5px; "><p><b>{}:</b></p>""".format(
                note_[0])
            for j in note_[1:]:
                html_string += """<p style='color: grey'> {} </p>""".format(j)
            html_string += "</div>"
            display(HTML(html_string))

    def warn(str_):
        str_ = str(str_)
        str_ = str_.replace("\n", "<br>")
        display(HTML("""<p style='color: darkred'> {} </p>""".format(str_)))

    def mcell(text, md=False):
        from IPython.display import display_javascript
        if md:
            get_ipython().set_next_input("")
            text = text.replace('\n', '\\n').replace("\"", "\\\"").replace("'", "\\'")
            text2 = """var t_cell = IPython.notebook.get_selected_cell();
            var t_index = IPython.notebook.get_cells().indexOf(t_cell);
            t_index = parseInt(t_index) + 1;
            t_cell = IPython.notebook.get_cells()[t_index]
            t_cell.set_text('{0}');
            IPython.notebook.to_markdown(t_index);
            IPython.notebook.get_cell(t_index).render();""".format(text)
            display_javascript(text2, raw=True)
        else:
            get_ipython().set_next_input(text)


def print_(x, *args, **kwargs):
    import datetime
    print(f'{datetime.datetime.now()}: {x}', *args, **kwargs)

def printr(x, *args, **kwargs):
    import datetime
    kwargs['end'] = ''
    print(f'\r{datetime.datetime.now()}: {x}', *args, **kwargs)


# @print_fun
def path_cls_inferrer(path):
    if cur_ser.a_dgfl:
        pp = PosixPath(path)
        if path.name.startswith('ed') or path.name.startswith('ED'):
            return E

        if pp.parent.name == 'A1' and path.name.startswith('L_'):
            return HD

        if pp.name.startswith('PL'):
            return Pl

        if (path.name.startswith('pld') or path.name.startswith('D1-')) and pp.is_dir():
            return Pld

        if path.name.startswith('HR'):
            return MM

        if pp.parent.name.startswith('HR') and path.name.isdigit():
            return M

    return P

def get_size(start_path='.'):
    # print('get_size function')
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def for_all_methods(decorator):
    def decorate(cls):
        import os
        for attr in cls.__dict__: # there's propably a better way to do this
            if callable(getattr(cls, attr)) and not attr.startswith('__'):
                if os.popen('hostname').read().strip() == 'ripper':
                    continue
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

class P(PosixPath):
    def __new__(cls, path, *args, **kwargs):
        if path == '.':
            path = os.path.abspath(os.curdir)
        path = PosixPath(path)

        check_existance = getattr(cls, 'check_existance', True)
        if check_existance and path.exists(): # path.exists() - very time consuming thing!!!! from (do not delete) lib.machines import * - 0.4 instead of 0.1 s
            path = path.resolve()
        if getattr(cls, 'infer_cls', True):
            if cls == P:
                proper_cls = path_cls_inferrer(path)
            else:
                proper_cls = cls
        else:
            proper_cls = cls
        # print(f"P class resolution [{proper_cls}]: {path}||")
        cls_new =  PosixPath.__new__(proper_cls, path, *args, **kwargs)
        cls_new.__init__(path)
        return cls_new

    def __init__(self, path):
        # print("P __init__")
        super(PosixPath, self).__init__()

        # new_self = super(PosixPath, self).resolve()

        # self = new_self

        self.prefix = os.path.splitext(str(self))[0]
        self.name_prefix = os.path.splitext(self.name)[0]
        self.double_name_prefix = self.parent.name + '__' + self.name_prefix
        self.splitext = os.path.splitext(str(self))
        self.name_splitext = os.path.splitext(self.name)
        # if self.name
        self.pd_sl_path = f'../{self.parent.name}/{self.name}/' if self.name.isdigit() else f'../{self.name}/'
        self.double_name = os.path.join(*self.parts[-2:])


    @property
    def PLs(self):
        return [i for i in self.dirs if i.name.startswith('PL') and i.name[:-1].endswith('__')]

    @property
    def _PLs(self):
        return [i for i in self.dirs if i.name.startswith('PL')]

    @property
    def PL(self):
        if not self.PLs:
            return
        return [i for i in self.PLs if i.name.split('__')[0][-1] == '0'][0]

    @property
    def _PL(self):
        if not self._PLs:
            return
        return [i for i in self._PLs if i.name[-1] == '0'][0]

    @property
    def is_node(self):
        self.name[:-1].endswith('__')

    def nodify(self):
        self.move_to(self.append2prefix('_'))

    def denodify(self):
        self.move_to(self.parent[self.name + '_'])

    @property
    def root(self):
        cur_directory = self
        for _ in range(self.stage):
            cur_directory = cur_directory.parent
        return P(cur_directory)

    @property
    def coords(self):
        cur_directory = self
        coords_list = []
        for _ in range(self.stage):
            coords_list.insert(0, cur_directory.idx)
            cur_directory = cur_directory.parent
        return coords_list

    @property
    def idx(self):
        assert self.is_dir()
        if self.name[:-1].endswith('__'):
            return int(self.name.split('__')[0].split('_')[-1])

    @property
    def stage(self):
        assert self.is_dir()
        if self.name[:-1].endswith('__'):
            return int(self.name[-1])
        else:
            cur_directory = self
            depth = 0
            while True:
                if depth > 5:
                    return
                par_directory = cur_directory.par
                if par_directory.s == "/":
                    return
                if par_directory.name == args.root_name:
                    return depth
                depth += 1

    def all_kids_dict(self, recursion=False, depth=1, org_file_list=[]):
        org_file_list_saved = self.rdv('org_file_list', if_not_exists=[])
        if not recursion:
            import datetime
            org_file_list = ['* LIGAND STATUS TREE']
        status_history = self.rdv('status_history', if_not_exists='')
        status_history = "\n".join(status_history)
        status = self.rdv('status', if_not_exists='')
        org_file_list += [f'{"*" * depth} | *{self.name}* |__{status}']
        org_file_list += [f'{"*" * (depth + 1)} Status history:\n\
        #+STARTUP: fold\
        ']
        org_file_list += [f'{status_history}']
        if self.kids:
            [child.all_kids_dict(recursion=True, depth=depth+1, org_file_list=org_file_list) for child in self.kids]
        if not recursion:
            if org_file_list_saved != org_file_list:
                self.wrv(org_file_list, 'org_file_list')
                d__tdir['ORG_TREE.org'].write_text("\n".join(org_file_list))
                print(f'Updating org_file_list. Date: {datetime.datetime.now()}')
            else:
                print(f'Not Updating org_file_list. Date: {datetime.datetime.now()}')

    def kids_fw(self, stage=None):
        if stage == None:
            stage = self.stage
        if stage == 4:
            return []
        kidss = self.kids + [kid.kids_fw(stage=stage+1) for kid in self.kids]
        kidss = [i for i in kidss if i]
        return kidss

    def kids_n(self, stages=None):
        all_kids = self.kids_fw()
        stages = [stages] if isinstance(stages, int) else stages
        stages = range(self.stage + 1, 5) if not stages else stages
        kids_n = [kid for kid in all_kids if kid.stage in stages]
        return kids_n

    @property
    def kids(self):
        # if self.__class__ == Pl: # for speed
        #     return [*self.rglob('HR_*__2')]
        return [i for i in self.dirs if i.name[:-1].endswith('__')]

    @property
    def kid(self):
        if not len(self.kids):
             return
        return sorted(self.kids, key = lambda i: int(i.name.split('__')[0].split('_')[-1]))[-1]

    @property
    def _id(self):
        return os.stat(self.s).st_ino
    def sle(self):
        sl(3)
        print(datetime.datetime.now())

    def launch2queue(self):
        import datetime
        qu = t.kid.rdv('queue', if_not_exists=[])
        launch_dict = {
            'stage': int(self.stage),
            'coords': self.coords,
            'id_': os.stat(self.s).st_ino,
            'dir_str': self.s,
            'fun_str': 'launch',
            'time_registered': datetime.datetime.now(),
        }
        qu.append(launch_dict)
        t.kid.wrv(qu, 'queue')

    def rm_kids(self):
        for kid in self.kids:
            kid.rm()

    def launch_kids(self):
        for kid in self.kids:
            kid.launch2queue

    def __str__(self):
        super_repr = super().__str__()
        super_repr = super_repr.replace('/mnt/storage/rustam/', '/home/domain/data/rustam/')
        return super_repr

    def __repr__(self):
        super_repr = super().__repr__()
        super_repr = super_repr.replace('/mnt/storage/rustam/', '/home/domain/data/rustam/')
        return super_repr

    def __getitem__(self, item):
        return self.joinpath(item)

    def __add__(self, other):
        return self[P(other)].s

    def __getattr__(self, attr):
        if hasattr(str, attr):
            return getattr(str(self), attr)
        else:
            raise AttributeError(f"Attribute {attr} not found in attributes")

    def write_text(self, *args, **kwargs):
        super().write_text(*args, **kwargs)
        self.chmod(0o777)

    @property
    def par(self):
        return P(self.parent)

    def joinpath(self, *args):
        # print(self, args)
        old_path = super().joinpath(*args)
        return P(old_path)

    def j(self, *args):
        return self.joinpath(self, *args)

    def all_methods(self, one_cell=False):
        k = [i.strip() for i in inspect.getsource(self.__class__).split('\n') if i.strip().startswith('def')]
        all_methods = [(i.split('(self')[0] + '()').replace('def ', 'self.') for i in k]
        if one_cell:
            mcell('\n'.join(all_methods))
        else:
            for j in all_methods[::-1]:
                print(j)
                get_ipython().ex(f'mcell({j!r})')

    # def rm_pat(self, subpaths_to_delete):
    #     for spath in subpaths_to_delete:
    #         spath =
    #     if 'cwd' not in kwargs.keys():
    #         kwargs['cwd'] = self.str
    #     rmPat(*args, **kwargs)

    def is_good(self):
        if '.' in self.name_prefix:
            return False
        if '#' in self.name.prefix:
            return False
        return True

    def g(self, rglob):
        return([*self.rglob(rglob)])

    def f(self, pref):
        pref_glob = self.rglob(pref)
        next_value = "NOT FOUND"
        try:
            next_value = next(pref_glob)
        except Exception:
            pass
        return next_value


    def d(self, pref):
        pref_glob = [i for i in self.dirs if i.name.startswith(pref)]
        next_value = "NOT FOUND"
        if pref_glob:
            print("RETURNED:", pref_glob[0], sep="\n--->")
            if len(pref_glob[1:]):
                print(*["Found matches to '{pref}':"] + pref_glob[1:], sep="\n--->")
            next_value = pref_glob[0]
        return next_value

    @property
    def url(self):
        if l2:
            return self.s.replace('/mnt/scratch/users/fbbstudent/', 'http://localhost:51517/edit/_scratch/')
        else:
            return self.s.replace('/home/domain/', 'http://localhost:51515/edit/')

    @property
    def t(self):
        print_(self, end='\n\n')
        print_(self.read_text())

    @property
    def cd(self):
        os.chdir(self.s)

    @property
    def ju(self):
        from IPython.display import Javascript
        display(Javascript(f'window.open({self.url!r});'))
        pass

    @property
    def n(self):
        nv_ = nv.show_file(self.s)
        display(nv_)

    @property
    def l(self):
        # print(f"globals: {len(globals().keys())}")
        # for i, j in globals().items():
        #     print(i, j)
        # print("-------------------locals")
        # for i, j in locals().items():
        #     print(i, j)
        # pmal("pepe")

        pmal(self.s)

    @property
    def rl(self):
        pmr()
        pmal(self.s)

    @property
    def mdirs(self):
        dirs = []
        for dir_ in self.dirs:
            if isinstance(dir_, M) or isinstance(dir_, MM):
                dirs.append(dir_)
        return dirs

    @property
    def agdirs(self):
        dirs = []
        for dir_ in self.dirs:
            if isinstance(dir_, M) or isinstance(dir_, MM):
                dirs.append(dir_)
        return dirs

    def replace_suffix(self, ext):
        return P(self.prefix + '.' + ext)

    def replace_prefix(self, new_name):
        return P(self.parent.joinpath(new_name + self.suffix))

    def append2prefix(self, prefix_complement):
        return P(self.parent.joinpath(self.prefix + '_' + prefix_complement + self.suffix))

    def prepend2prefix(self, prefix_complement):
        return P(self.parent.joinpath(prefix_complement + '_' + self.name_prefix + self.suffix))

    def make_res(self, appendage):
        self.copy_to(self.append2prefix(appendage))

    def iterdir(self):
        dirs = super().iterdir()
        for dir_ in dirs:
            yield P(dir_)
        # return [P(dir_) for dir_ in dirs]

    def run(self, *args, **kwargs):
        old_cwd = os.getcwd()
        os.chdir(self.s)
        from invoke import run
        from textwrap import dedent
        if len(args):
            print(dedent(f'Execing command:\n\
            {args[0]}\
            '))
        run(*args, **kwargs)
        os.chdir(old_cwd)

    def trun(self, *args, **kwargs):
        kwargs['cwd'] = self
        tsp(*args, **kwargs)

    @property
    def str(self):
        return str(self)

    @property
    def s(self):
        return str(self)

    @property
    def size(self):
        if self.is_file():
            size = self.stat().st_size()
        if self.is_dir():
            size = get_size(str(self))
        else:
            raise NotImplementedError(f'{self}: not file and not dir')
        return size

    @property
    def hsize(self):
        import humanize
        if self.is_file():
            return humanize.naturalsize(self.stat().st_size, gnu=True)
        if self.is_dir():
            size = sum(os.path.getsize(f) for f in os.listdir('.') if os.path.isfile(f))
        return humanize.naturalsize(self.size, gnu=True)

    @property
    def alt(self):


        return P(self.str.replace(cur_ser.d_d.str, cur_ser.alt.ddir.str))

    @property
    def dirs(self):
        return [i for i in self.iterdir() if i.is_dir()]

    @property
    def files(self):
        return [i for i in self.iterdir() if i.is_file()]

    @property
    def ls(self):
        return list(self.iterdir())

    @property
    def r(self):
        # IF two files not workint!!!
        # file_addition, dirs_addition = columnize(self.files, 4), columnize(self.dirs, 4)

        if self.files:
            print('Files: ')
            for file in self.files:
                print(file)
            # display(DataFrame([f.hsize, f for f in self.files]))
        if self.dirs:
            print('Dirs: ')
            for dir_ in self.dirs:
                print(dir_)
            # print(f'\tFiles:\n{self.files}\n\tDirs:\n{self.dirs}')

    # def itr(self):
    #     tree = ipytree.Tree(stripes=True)
    #     display(tree)
    #     dir_node = ipytree.Node('aaa')
    #     tree.add_node(dir_node)
    #     return tree
    #     # for root, dirs, files in os.walk(self):
    #     #     dir_node = ipytree.Node(os.path.basename(root))
    #     #     tree.add_node(dir_node)
    #     #     level = root.replace(str(self), '').count(os.sep)
    #     #     indent = ' ' * 4 * (level)
    #     #     print('{}{}/'.format(indent, os.path.basename(root)))
    #     #     subindent = ' ' * 4 * (level + 1)
    #     #     for f in files:
    #     #         print('{}{}'.format(subindent, f))

    def tree(self, dpth=3, cut=10):
        print(f'+ {self.name}')
        eq = 0
        depth2 = None
        for path in sorted(self.rglob('*')):
            depth = len(path.relative_to(self).parts)
            if depth2 != depth:
                eq = 0
            else:
                eq += 1
            if dpth and depth > dpth:
                continue
            spacer = '    ' * depth
            if eq > cut:
                continue
            print(f'{spacer}+ {path.hsize:<7}| {path.name}')
            depth2 = depth

    @property
    def tr(self):
        self.tree(3)

    def relpath(self, second_path=os.path.curdir, db=1):
        if db: vd(second_path)
        # print(os.path.realpath(os.path.curdir), 'aaa', os.path.relpath(P('../RR3_10_3/dec.pdb').resolve()))
        return os.path.relpath(self.name, second_path)

    def mkdir(self, db=0):
        if db: print(f'Making dir: {self}')
        if not self.exists():
            super().mkdir()

    def rm(self):
        if self.par != d__tdir:
            double_name = self.s.replace("/",">>")
            [i.rm() for i in d__tdir.ls if i.name.startswith(f'|{double_name}')]
            # if cur_ser.a_name == "srv":
            #     [i.rm() for i in d__tdir.ls if i.name.startswith(f'|{double_name}')]
            # else:
            #     print("Do not know your t_dir! Fixme please")
        if self.exists():
            if self.is_file():
                self.unlink()
            if self.is_dir():
                shutil.rmtree(self)

    def rwdir(self):
        self.rm()
        self.mkdir()

    def rwfile(self):
        self.rm()
        self.touch()

    def copy_from(self, source):
        source = P(source).resolve()
        if source.is_file():
            shutil.copy(source, self)
        if source.is_dir():
            shutil.copytree(str(source), str(self))
        else:
            raise NotImplementedError('not a file and not a dir')

    def rsync_to(self, *args, **kwargs):
        import sysrsync
        kwargs.update({
            'verbose': True,
            'destination': str(args[0]),
            'source': str(self)
        })
        if 'options' not in kwargs.keys():
            kwargs['options'] = []
        kwargs['options'] = list(set(kwargs['options'] + ['-a', '-v']))
        sysrsync.run(**kwargs)

    def rsync_from(self, *args, **kwargs):
        import sysrsync
        kwargs.update({
            'verbose': True,
            'source': str(args[0]),
            'destination': str(self)
        })
        if 'options' not in kwargs.keys():
            kwargs['options'] = []
        kwargs['options'] = list(set(kwargs['options'] + ['-a', '-v']))
        sysrsync.run(**kwargs)

    def rsync_to_alt(self, **kwargs):
        print(f'mdir: {mdir}, udir: {udir}, ddir: {ddir}, edir: {edir}')
        print(f'{cur_ser}')
        print(f'{cur_ser.alt.ddir}')
        print(f'{cur_ser.d_d}')
        print(f'{srv.udir}')
        print(f'{srv.ddir}')

        print(f'rsync_to_alt with kwargs: {kwargs}!!!!')

        # change the code or move md and ed from data/rustam to data/rustam/dgfl
        # self.str.replace(cur_ser.d_d.str, cur_ser.alt.ddir.str),

        kwargs.update({
            'verbose': True,
            'source': self.str,
            'destination': self.str.replace(cur_ser.d_u.str,
                                            cur_ser.alt.udir.str),
            'destination_ssh': cur_ser.alt.a_host_alias
        })
        if 'options' not in kwargs.keys():
            kwargs['options'] = []
        kwargs['options'] = list(set(kwargs['options'] + ['-a', '-v']))
        print(f'rsync_to_alt with kwargs: {kwargs}!!!! UPDATED')
        sysrsync.run(**kwargs)

    def rsync_from_alt(self, **kwargs):
        import sysrsync
        kwargs.update({
            'verbose': True,
            'destination': self.str,
            'source': self.str.replace(cur_ser.d_d.str, cur_ser.alt.ddir.str),
            'source_ssh': cur_ser.alt.a_host_alias
        })
        if 'options' not in kwargs.keys():
            kwargs['options'] = []
        kwargs['options'] = list(set(kwargs['options'] + ['-a', '-v']))
        sysrsync.run(**kwargs)

    def move_to(self, dest):
        double_name = self.s.replace("/",">>")
        [i.rm() for i in d__tdir.ls if i.name.startswith(f'|{double_name}')]
        shutil.move(self, dest)

    def copy_to(self, dest):
        if self == dest:
            print('bad code! you need to check by inodes or by diff')
            return
        dest = P(dest)
        if self.is_file():
            shutil.copy(self, dest)
            if dest.is_dir():
                return dest.j(self.name)
            else:
                return dest
        elif self.is_dir():
            import subprocess
            from invoke import run
            run(f'cp -r {self.str} {dest.str}')
            return dest
        else:
            raise NotImplementedError('not a file and not a dir')

    def duplicate(self, suffix='00'):
        new_dir = P(self.prefix + '00')
        new_dir.initialize()
        new_dir.ed.initialize()
        print('eu')
        self.joinpath('BFF4agent_1.itp').copy_to(new_dir)
        self.ed.cg.joinpath('lig_from_traj.pdb').copy_to(new_dir.ed.cg)

        new_prot_from_traj = new_dir.ed.ss.joinpath('prot_from_traj.pdb')
        self.ed.ro.joinpath('lpla_lg2_0__DE_1.pdb').copy_to(new_prot_from_traj)
        pm(f'load {new_prot_from_traj}', 'remove not polymer',
           f'save {new_prot_from_traj}')

        return new_dir

    def rmv(self, *args, **kwargs):

        kwargs['md_dir'] = self
        return rmv(*args, **kwargs)


    def rdv(self, *args, **kwargs):

        kwargs['md_dir'] = self
        return rdv(*args, **kwargs)

    def apv(self, *args, **kwargs):

        kwargs['md_dir'] = self
        return apv(*args, **kwargs)

    def wrv(self, *args, **kwargs):

        kwargs['md_dir'] = self
        wrv(*args, **kwargs)

    def plt_plot(self, *args, **kwargs):
        kwargs['p_dir'] = self
        plt_plot(*args, **kwargs)

    def plt_savefig(self, *args, **kwargs):
        kwargs['p_dir'] = self
        plt_savefig(*args, **kwargs)



def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def tsp(cmnd, sn=19, wn="wnd", cc=True, kill_window=True, ipk=False, silent=False, log=True, wait=False, cwd=os.path.curdir):
    import libtmux
    os.system('tmux ls >/dev/null 2>&1 || tmux new-session -d ')  # if no server running
    tmux_srv = libtmux.Server()

    sn, wn = str(sn), str(wn)
    cmnd = "cd {} && {}".format(os.path.realpath(os.path.join(os.path.curdir, cwd)), cmnd)
    print("Trying to find existing session: {}. Please wait...".format(sn))
    en_ssn = tmux_srv.find_where({'session_name': sn})
    en_ssn = tmux_srv.find_where({'session_name': sn})
    if not en_ssn:
        en_ssn = tmux_srv.new_session(sn, window_name=wn)
        en_win = en_ssn.windows[0]
    else:
        en_ssn = tmux_srv.find_where({'session_name': sn})
        en_win = en_ssn.find_where({'window_name': wn})
        if not en_win:
            en_win = en_ssn.new_window(wn)
    if cc:
        en_win.attached_pane.send_keys('C-c', enter=False, suppress_history=False)
    elif kill_window:
        en_win.kill_window()
        en_win = en_ssn.new_window(wn)
    en_win.attached_pane.send_keys(cmnd)

def dr(obj, full=False):
    for i in [v for v in dir(object) if not callable(getattr(object, v))]:
        if i.startswith('_'):
            continue
        print('\n%s:' % i)
        exec('print(object.%s\n\n)' % i)

    #     # print(", ".join([k for k in dir(obj) if not k.startswith("_")]))
    # for attr in dir(obj):
    #     if not full and attr.startswith("_"):
    #         continue
    #     print(attr, getattr(obj, attr))

def send2tmux(cmnd, session_name='general', window_name="w", cc=False,
              kill_session=False, kill_window=False, cwd=None, ntf=None, say=False, bash=True):
    # TOO SLOW! WE NEED BASH MAYBE: make with bash !!!!!!
    # os.system(f'tmux send-keys -t general:rc "{cmnd}" C-m')

    if cwd:
        cmnd = f'cd {cwd}; {cmnd}'
    # if ntf:
    #     cmnd = f'{cmnd}; _ntf("MAYBE SUCCEDED", """{cmnd}""")'
    if say:
        # cmnd = f'{cmnd} && say "ok" || (say "bad" && _ntf(\'aha\'))'
        cmnd = f'{cmnd} 2>&1 | ntf'

    if bash:
        os.environ['PATH'] += os.pathsep + '/usr/local/bin'
        os.system(f'tmux send-keys -t general:rc "{cmnd}" C-m')
    else:
        import libtmux
        os.system('tmux ls >/dev/null 2>&1 || tmux new-session -d ')  # if no server running
        tmux_srv = libtmux.Server()
        en_ssn = tmux_srv.find_where({'session_name': session_name})
        session_killed = en_ssn and kill_session
        no_ssn = not en_ssn
        if session_killed:
            en_ssn.kill_session()

        if no_ssn or session_killed:
            en_ssn = tmux_srv.new_session(session_name, window_name=window_name)

        en_win = en_ssn.find_where({'window_name': window_name})
        window_killed = en_win and kill_window
        no_win = not en_win
        if window_killed:
            en_win.kill_window()
        if no_win or window_killed:
            en_win = en_ssn.new_window(window_name)
        pane = en_win.attached_pane
        if cc:
            pane.send_keys('C-c', enter=False, suppress_history=False)



        pane.send_keys(cmnd)

def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    import itertools
    return itertools.zip_longest(fillvalue=fillvalue, *args)

def _rp(args):
    return os.path.abspath(args[0])


def _ligr():
    from invoke import run
    run('sshfs -o allow_other srv:/home/domain/data/rustam/ ~/r')

def _tl():
    from invoke import run
    run('tmux ls')

def _ta(args):
    # from invoke import run
    os.system(f'tmux attach-session -t {args[0]}')

def _s():
    os.system('ssh srv')


# ALIASES list
if 'xonsh' in sys.modules:
    import re
    for leading_slashed in [i for i in dir() if re.match(r'_[^_]+', i)]:
        alias = leading_slashed[1:]
        # print(f'Adding to aliases: {eval(leading_slashed)} -> aliases["{alias}"]')
        aliases[alias] = eval(leading_slashed)

# wrv and rdv in dgfl are bound to P. Here are simple wrv and rdv for variables.



def dr(obj):
    print(", ".join([k for k in dir(obj) if not k.startswith("_")]))

def dri(obj):
    for k in dir(obj):
        try:
            if k.startswith("_"):
                continue
            v = getattr(obj, k)
            print(f"--->{k}")
            print(f"{v}")
        except Exception:
            print(f"--->{k}")
            print("Error occured")

def rec_dr(obj, num):
   file_name = f"/tmp/tmp_{num}"
