#!/Users/rst/miniconda/envs/py37/bin/python3
# IMPORTANT MODULES
# print('local.py')

import os
import re
import subprocess
import sys


if __name__ == "__main__":  # if ipython executes it
    # from IPython.core.display import display, HTML
    # from pandas import option_context

    # get_ipython().run_cell_magic('HTML', '', '<style>\ndiv.prompt {display:none}\n.container {width: 98%}\n</style>')


    # RUN IT ONLY WITH IPYTHON
    # lq", opts)
    # if opts.w:
    #     print("LQ")

    # CLASSES
    # FUNCTIONS
    # def dfull(df):
    #     df.style.set_properties(subset=['text'], **{'width': '30000px'})
    #     with pandas.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 500):
    #         display(df)
    def dfull(df):
        with option_context('display.max_rows', None, 'display.max_columns', None):
            display(df)

def _bundle_id():
    import applescript


def hwd():
    os.system("say 'hello world'")

def _bk(paths: list):
    import datetime
    for path in paths:
        from lib.base import P
        P.infer_cls = True
        path = P(path)
        path.parent['.res'].mkdir()
        appendage = str(datetime.datetime.now()).replace(' ', '_')
        bk_path = path.parent[f'.res/{path.name}_{appendage}']
        assert not bk_path.exists()
        print(f'copying {path} to: {bk_path}')
        path.copy_to(bk_path)


def _pe():
    from richxerox import copy, paste
    from pyautogui import hotkey
    text = paste()
    text = text.encode('unicode_escape').decode()
    copy(text)
    hotkey('command', 'v')


def _eml():
    os.system("say emailing zip")


def ntf(title='Title', text=""):
    if text.strip() == 'Done!':
        os.system('say ok')
        return
    os.system("""
              osascript -e 'display notification "{}" with title "{}"'
              """.format(text, title))
    print(text)


def _ntf(*args, stdin=None):
    from richxerox import copy
    title = 'NTF'
    if stdin:
        text = stdin.read().strip()
    else:
        if args:
            if len(args) == 1:
                text = args[0]
            elif len(args) == 2:
                title = args[0]
                text = args[1]
        title = args[0]
    ntf(title, text)


def _gk():
    from lib.base import send2tmux
    send2tmux('goku', 'general', 'rc', say=True)


def _cb():
    import pyautogui
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.typewrite('c """')
    pyautogui.hotkey('ctrl', 'e')
    pyautogui.typewrite('"""')
    pyautogui.press('enter')




def _ara():
    os.system('say apple')


def _p():  # pdf selection
    ''' Special paste . Autodetects what is in your clipboard and chooses the right
    way to fix it.
    e. g.:
    - Ipython >>> ....
    - stackoverflow $ ....
    - bash -> xonsh incompatible code:
        - some_var="$(...)" - > some_var = $(....)
    '''
    from richxerox import copy, paste
    import pyautogui
    # pyautogui.hotkey('command', 'c')
    text_sel = paste()

    # text_sel = re.sub(r'Scale bars, 500 nm\.', '', text_sel)
    # text_sel = re.sub(r"Scale bars, 5'0 nm\.", '', text_sel)
    # text_sel = re.sub(r'\([Pp]robe [Ss]et #\d+, Table S\d+\)', '', text_sel)
    # text_sel = re.sub(r'\([Pp]robe [Ss]et#\d+, Table S\d+\)', '', text_sel)
    text_sel = re.sub(r'See also Figure S\d+.', '', text_sel)
    text_sel = re.sub(r'(\w)(-\n)', '\g<1>', text_sel)
    text_sel = re.sub(r'(\w)(- )', '\g<1>', text_sel)
    text_sel = re.sub(r' (\(.[^\(]*?, \d{4}(?:[a-z]|)\))', '', text_sel)
    text_sel = text_sel.strip('\n ').replace('\n', '<br>')
    style = """
    p {
    line-height: 0.1
    }
"""
    # br {
    #    display: block;
    #    margin: 50px 0 !important;
    # }
    html_sel = f'<style>{style}</style><p>{text_sel}</p>'
    html_sel = re.sub(r'(<br>\([A-Z]-[A-Z]\)(?: |\.))', '<b>\g<1></b> ', html_sel)
    html_sel = re.sub(r'(\([A-Z] and [A-Z]\)(?: |\.))', '<b>\g<1></b> ', html_sel)
    # html_sel = re.sub(r'((?:<br>|)\([A-Z]\)(?: |\.))', '<b>\g<1></b> ', html_sel)
    html_sel = re.sub(r'(\([A-Z]\))', '<b>\g<1></b> ', html_sel)

    copy(html=html_sel)
    pyautogui.hotkey('command', 'v')


def _dds():  # dedent selection
    from richxerox import paste, copy
    from textwrap import dedent
    sel = paste()
    new_lines = []
    for line in sel.split('\n'):
        line = re.sub(r'\s*>>>\s*', '', line)
        line = re.sub(r'\w+@\w+ ~ \$ ', '', line)
        if line.startswith('..'):
            new_line = re.search(r'\.+(.*)', line).group(1)
        else:
            new_line = line
        new_lines += [new_line]
    new_sel = '\n'.join(new_lines)
    new_sel = dedent(new_sel)
    copy(new_sel)
    import pyautogui
    pyautogui.hotkey('command', 'v')


def _p():
    print('execing : _p')
    _dds()


def _ava3():
    import pyautogui
    pyautogui.press('v')
    # pyautogui.hotkey('command', 'v')


def _c(args, stdin=None):
    from richxerox import copy
    if stdin:
        to_copy = stdin.read().strip()
    else:
        to_copy = ' '.join(args)
    to_copy = rf'{to_copy}'
    copy(to_copy)
    # print(f'{x} copied!')


def _idt(args):
    from richxerox import copy, paste
    from pyautogui import hotkey
    if args:
        prefix = args[0]
    else:
        prefix = '    '
    hotkey('command', 'c')
    text = paste()
    new_text_list = []
    for line in text.splitlines():
        new_text_list.append(prefix + line)
    new_text = '\n'.join(new_text_list)
    copy(new_text)
    hotkey('command', 'v')


def _sudoc(cmd):
    from lib.machines import cur_ser
    from invoke import run
    run(f'sudo {cur_ser.udir["lib/msc/sudoc"]} --cmd {" ".join(cmd)}')
    # subprocess.call(["sudo", cur_ser.udir['lib/msc/sudoc'].s, "--cmd", "\n".join(cmd)])


def _agi(args):
    print(args)
    for arg in args:
        _sudoc(['apt-get', 'install', arg])


def cbpy2():
    from richxerox import paste, copy
    import re
    cb_text = paste(format='text')
    lines = cb_text.split('\n')
    new_lines = []
    for line in lines:
        line = re.sub(r'(#.*$)', '', line).strip()  # we strip comments
        if line:
            new_line = re.sub(r'print (.*)', r'print(\g<1>)', line)
            new_lines.append(new_line)
    new_text = '\n'.join(new_lines)
    copy(new_text)
    _ntf('COPIED', 'to clipboard')


def live_process_output(proc, return_=1):
    outputs = []
    while True:
        output = proc.stdout.readline()
        # print('NTF', output, proc.poll())
        if output in [b'', ''] and proc.poll() is not None:
            break
        if output:
            output = output.decode()
            outputs.append(output)
            print(output, end='')
    # return True
    o = ''.join(outputs)
    if return_:
        return o


def next_file_name(filePath):
    base_name = os.path.basename(filePath)
    fidx, fext = os.path.splitext(base_name)
    fidx = int(fidx)
    new_file_name = filePath
    import glob
    while glob.glob(new_file_name):
        fidx += 1
        new_file_name = f"{os.path.dirname(filePath)}/{fidx}.*"
        print(f"new_file_name: {new_file_name}")
    new_file_name = f"{os.path.dirname(filePath)}/{fidx}{fext}"
    return (new_file_name)


def get_cur_wnn():
    command = "xprop -root _NET_ACTIVE_WINDOW | sed 's/.* //'"
    frontmost = subprocess.check_output(["/bin/bash", "-c", command]).decode("utf-8").strip()
    fixed_id = frontmost[:2] + "0" + frontmost[2:]
    subprocess.check_output(["/bin/bash", "-c", command]).decode("utf-8").splitlines()
    command = "wmctrl -lp"
    from pandas import DataFrame
    df = DataFrame(map(lambda x: x[:42].split() + [x[42:]],
                       subprocess.check_output(["/bin/bash", "-c", command]).decode("utf-8").splitlines()))
    wnn = df[df[0] == fixed_id][4].values[0]
    return wnn


def _rc2():
    from lib.base import send2tmux
    from lib.local import ntf
    cmnd = 'cd ~/; p37; driver = start_firefox_selenium_driver()'
    _ntf('Restaring rc...')
    send2tmux(cmnd, 'general', 'rc2', cc=False, kill_window=True)


def en_read_firefox(query):
    from textwrap import dedent
    from lib.base import send2tmux

    query = query.replace(' ', '%20').replace('-', '%20')
    url = f'https://ru.forvo.com/search/{query}/'
    cmnd = dedent(f"""
                driver.get('{url}')
                driver.find_element_by_css_selector('.play').click()
                """)
    print(cmnd)
    send2tmux(cmnd, 'general', 'rc2', cc=False, kill_window=False)


def _en(query):
    db = 0
    import urllib3
    from bs4 import BeautifulSoup
    import shutil

    if not query:
        from richxerox import paste
        import pyautogui
        pyautogui.hotkey('command', 'c')
        query = paste()
    else:
        query = ' '.join(query)
    en_read_firefox(query)
    query = query.replace(' ', '-')
    http = urllib3.PoolManager()
    url = f'https://www.collinsdictionary.com/dictionary/english/{query}'
    print('url:', url)
    response = http.request('GET', url)
    try:
        soup = BeautifulSoup(response.data, features="html.parser")
    except Exception as e:
        print(f'Exception: {e} \n url: {url} \n response: {response}')

    # pron
    gj = None
    for j in soup.find_all('span'):
        if j.has_attr('class'):
            if all([i in j.attrs['class'] for i in ['pron']]):
                gj = j
                break
    if gj:
        pron = gj.get_text().strip()
        _ntf(query, pron)

    # mp3
    mp3 = None
    for j in soup.find_all('a'):
        if j.has_attr('data-src-mp3'):
            mp3 = j.attrs['data-src-mp3']
            break
    if not mp3:
        en_read_firefox(query)
    else:
        filename = '/tmp/da_audio.mp3'
        c = urllib3.PoolManager()
        with c.request('GET', mp3, preload_content=False) as resp, open(filename, 'wb') as out_file:
            shutil.copyfileobj(resp, out_file)
        resp.release_conn()

        subprocess.Popen(['afplay', filename])


# def _pron(args):


def _da(args):
    import os
    from lib.base import P
    if not args:
        from richxerox import paste
        import pyautogui
        pyautogui.hotkey('command', 'c')
        query = paste()
    else:
        query = ' '.join(args)
    query = query.replace(' ', '+')
    # tmp_file = P('/tmp/dict_aggr_query_1.html')
    # template_file = P('/Users/rst/.config/example.html')
    # template_text = template_file.read_text()
    # template_text = template_text.replace('apple juice', query)
    # tmp_file.write_text(template_text)
    # os.system(f'open {tmp_file}')
    os.system(f'open "https://alesia.store/da.php?query={query}"')


def sl(tim=1.3):
    import time
    time.sleep(tim)


def say(smth):
    os.system(f"espeak '{smth}'")


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def _ligr():
    from lib.machines import cur_ser
    from invoke import run
    import inspect
    fun_name = inspect.stack()[0][3]
    run("sshfs srv:/home/domain/data/rustam/ ~/r -o reconnect,"
        "auto_cache,defer_permissions,noappledouble,"
        "Compression=no -ovolname=srv2")


def _liga():
    from lib.machines import cur_ser
    from invoke import run
    import inspect
    fun_name = inspect.stack()[0][3]
    run(
        "sshfs als:/var/www/ ~/a -o reconnect,auto_cache,defer_permissions,noappledouble,Compression=no -ovolname=als")

def _uma():
    from invoke import run
    run('sudo umount -f /Users/rst/a')

def sls(rc, id_):
    return (rc.seq[int(rc.features[id_].location.start): int(rc.features[id_].location.end)])


def sp(command, silent=False, cwd="", py36=False, com=None, exp=0):
    if cwd:
        cwd = os.path.abspath(cwd)
    if py36:
        command = "/Users/rst/anaconda3/etc/profile.d/conda.sh; conda activate py36; " + command
    if exp:
        command = """ export PATH=$PATH":/Users/rst/bin/:/Users/rst/lib/:/Users/rst/l   ib/rc/:/Users/rst/lib/grave_compose/";""" + command
    my_env = os.environ.copy()
    print("Executing command:\n{}".format(command))
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.abspath(cwd),
                         env=my_env, executable='/bin/bash')
    ser = ""
    so, ser = p.communicate()
    so, ser = map(lambda x: x.decode('utf-8') if type(x) == bytes else x, [so, ser])
    if not silent:
        if ser: ser = "\nSTDERR:\n{}".format(ser)
        print("STDOUT:\n{}{}".format(so, ser))
    return so, ser


def clb(text):
    xsel_proc = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
    xsel_proc.communicate(text.encode())


def clp():
    import pyautogui
    cl(str(list(pyautogui.position())))


def lcl(str_):
    cl(str_.replace("_", "\_"))


def get_diff(a1, b1):
    import difflib
    def show_diff(seqm):
        """Unify operations between two compared strings
    seqm is a difflib.SequenceMatcher instance whose a & b are strings"""
        output = []
        for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
            if opcode == 'equal':
                output.append(seqm.a[a0:a1])
            elif opcode == 'insert':
                output.append("<ins>" + seqm.b[b0:b1] + "</ins>")
            elif opcode == 'delete':
                output.append("<del>" + seqm.a[a0:a1] + "</del>")
            elif opcode == 'replace':
                output.append("(" + seqm.a[a0:a1] + "->" + seqm.b[a0:a1] + ")")
                # print("what to do with 'replace' opcode?")
            else:
                print("unexpected opcode")
        return ''.join(output)

    sm = difflib.SequenceMatcher(None, a1, b1)
    print(show_diff(sm))
    display(HTML(show_diff(sm)))


def russianDictToDict(dict_, php_format=False):
    import json
    p = json.dumps(dict_, sort_keys=True, indent=4)
    pr = re.findall(r"\"\\.*?\"", p)
    # print p
    u_notation = "u" if not php_format else ""
    for pr2 in pr:
        p = p.replace(pr2, u"{}{}".format(u_notation, eval(u"u'{}'".format(pr2))))
    return p


def listFromStr(list_):
    return u"[u'{}']".format("', u'".join([j for j in list_]))


# megaCatDict[u'Каталог'] = catalog
def utfToRus(str_, php_format=False, quote_mark='"'):
    p = str(str_)
    if quote_mark == '"':
        pr = re.findall(r"\"\\.*?\"", p)
    elif quote_mark == "'":
        pr = re.findall(r"'\\u.*?'", p)
    u_notation = "u" if not php_format else ""
    for pr2 in pr:
        if quote_mark == '"':
            p = p.replace(pr2, u"{}{}".format(u_notation, eval(u"u'{}'".format(pr2))))
        elif quote_mark == "'":
            p = p.replace(pr2, u"{}{}".format(u_notation, eval(u"u\"{}\"".format(pr2))))
    return p

def wrv(variable, var_name, db=1, md_dir=None):
    import pickle
    pickle_path = P("/tmp")[var_name]
    if db: print(f'wrv: {pickle_path} {str(variable)[:20]}')
    pickle.dump(variable, pickle_path.open("wb"))

def rdv(var_name, db=0, if_not_exists='IOError'):
    #  ./.cache/auto-save/site/#!home!ec!.emacs.d!layers!+spacemacs!spacemacs-defaults!funcs.el#
    import pickle
    pickle_path = P("/tmp")[var_name]
    if not P(pickle_path).exists():
        return if_not_exists
    if db:
        print(f'rdv: {pickle_path}')
    return pickle.load(open(pickle_path, "rb"))

def msoc(x=None, make_oc_tables=0, at=0, exec=False):
    from lib.base import Info
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


def get_rus_id():
    df2 = msoc('oc_language')
    rus_id = int(df2.loc[df2['name'] == 'Russian', 'language_id'])
    return rus_id


def df2str(v, tb=0):
    if len(v) < 2:
        st = "DataFrame(" + str({p: "" for p in v.columns}) + ")"
    else:
        st = "DataFrame(" + str({p: v.loc[0, p] for p in v.columns}) + ")"
    if tb:
        st = f"""ndfs["{tb}"] = """ + st
    cl(st)
    print(f"Copied: \n {st}")
    return st


def shtr(t):
    for c in t.get_children():
        print(c.name, len(c.get_children()))
        print(c.name, len(c.get_descendants()))
    display(t.render("%%inline", tree_style=ts))


def phpar(x, name="ar"):
    str_ = f"${name} = (" + str(x) + ");"
    cl(str_)
    print(f"Copied: \n{str_}")
    return str_


def ssp(cmd_, host='alesia.store', user='rustam', secret='Fnfkfpf9', port=22, silent=False):
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, password=secret, port=port)
    stdin, stdout, stderr = client.exec_command(cmd_)
    data = type('obj', (object,), {'out': stdout.read().decode('utf-8'), 'err': stderr.read().decode('utf-8')})
    if not silent:
        print(f"Err:\n {data.err}\nOut: \n{data.out}")
    client.close()
    return (data)


def start_firefox_selenium_driver():
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options
    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options, executable_path=r'/usr/local/bin/geckodriver')
    print("Headless Firefox Initialized")
    return driver


def cl(text):
    import richxerox
    richxerox.copy(text)


# def finder_copy_selected_items_as_paths():
#     import applescript
#     cmd_ = '''
#         tell application "Finder"
#             set finderSelList to selection as alias list
#         end tell
#         if finderSelList ≠ {} then
#             repeat with i in finderSelList
#                 set contents of i to POSIX path of (contents of i)
#             end repeat
#
#             set AppleScript's text item delimiters to linefeed
#             finderSelList as text
#         end if
#         log finderSelList
#         set the clipboard to finderSelList as text
#     '''
#     applescript.run(cmd_)
#
# def compress_and_email():
#     from richxerox import copy, paste
#     finder_copy_selected_items_as_paths()
#     selected_filenames = paste()
#
#     pass


def _nf():
    sp("/usr/bin/automator '/Users/rst/Library/Services/New Empty Text File.workflow'")


# ALIASES list
if 'xonsh' in sys.modules:

    for leading_slashed in [i for i in dir() if re.match(r'_[^_]+', i)]:
        alias = leading_slashed[1:]
        # print(f'Adding to aliases: {eval(leading_slashed)} -> aliases["{alias}"]')
        aliases[alias] = eval(leading_slashed)

    tmux_commands = {
        'jpf_gpu': 'ssh -N -L 51515:localhost:51515 gpu',
        'jpf_lom2': 'ssh -N -L 51515:localhost:51515 lom2',
        'ppf_gpu': 'ssh -N -R 9170:localhost:9170 gpu',
        'ppf_als': 'ssh -vvv -N -R9000:localhost:9000 als',
        'j_l': 'source activate py37; jupyter notebook --no-browser'
    }
    from lib.base import send2tmux

    for name, cmd_ in tmux_commands.items():
        aliases[name] = lambda x: send2tmux(cmd_, 'general', name)

# UBUNTU ONLY
# def kl(x):
#     import klembord
#     klembord.init()
#     klembord.set_text(x)

# def cl(text, v=False):
#     import pyperclip
#     pyperclip.copy(str(text))
#     if v:
#         print(f"Copied:\n{str(text)} ")

def sys_activate_window_by_pid(pid):
    print(f"--> new app, : {len(windowList)}")
    for window in windowList:
        print("\t", window["kCGWindowOwnerName"], window["kCGWindowOwnerPID"])
        if type(window["kCGWindowOwnerPID"]) == pid:
            print("emacs_main_pid-window FOUND. activating..")
            NSLog('%@', window)
            break

# def sys_get_active_window():

def sys_activate_window_by_pid_emacs():
    pid = int(open("/tmp/emacs_main_pid").read())
    print(f"emacs_main_pid read: {pid} ")
    sys_activate_window_by_pid(pid)
    for i in windowList:
        if i["kCGWindowOwnerName"] == "emacs":
            display(i)

def sys_get_frontmost_window():
    from AppKit import NSApplication, NSApp, NSWorkspace
    from Foundation import NSObject, NSLog
    from PyObjCTools import AppHelper
    from Quartz import kCGWindowListOptionOnScreenOnly, kCGNullWindowID, CGWindowListCopyWindowInfo
    workspace = NSWorkspace.sharedWorkspace()
    activeApps = workspace.runningApplications()
    fma = NSWorkspace.sharedWorkspace().frontmostApplication
    for app in activeApps:
        print(app)
        if app == fma:
            print("horray", app)
        break
        options = kCGWindowListOptionOnScreenOnly
        windowList = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
        # print(f"sys_get_window_list: len of windowList", len(windowList))
        # return windowList

def sys_get_window_list():
    from AppKit import NSApplication, NSApp, NSWorkspace
    from Foundation import NSObject, NSLog
    from PyObjCTools import AppHelper
    from Quartz import kCGWindowListOptionOnScreenOnly, kCGNullWindowID, CGWindowListCopyWindowInfo
    workspace = NSWorkspace.sharedWorkspace()
    activeApps = workspace.runningApplications()
    for app in activeApps:
        print(app)
        break
        options = kCGWindowListOptionOnScreenOnly
        windowList = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
        print(f"sys_get_window_list: len of windowList", len(windowList))
        return windowList

def sys_get_newly_opened_window(callback=None, time_span=2.0):
    import time

    windowList1 = sys_get_window_list()
    time.sleep(time_span)
    windowList2 = sys_get_window_list()
    new_window_list = list(set(windowList2) ^ set(windowList1))
    if new_window_list:
        return new_window_list[0]
    else:
        print("No new windows detected")
"py37s1 \"emacs_gepy_pid = sys_get_newly_opened_window()\"; emc gepy"

# HELM_COMPLETION_START
def descr(x, num=1):
    print("running descr ...")
    open(f"/tmp/tmp{num}", "w").write("\n".join(dir(x)))

def oct_fastorder_client_info():
    from richxerox import copy, paste
    from pyautogui import hotkey, press
    import time
    hotkey("command", "tab")
    for data_piece in ["Rustam", "rustamusmanov973@gmail.com", "9856345704", "Moscow", "Gorod"]:
        copy(data_piece)
        hotkey("command", "v")
        if data_piece == "Moscow":
            time.sleep(.1)
            press("down")
            time.sleep(.1)
            press("enter")
            time.sleep(.1)
        press("tab")
        time.sleep(.1)

def chrome_search_selected(et=0):
    from richxerox import copy, paste
    from pyautogui import hotkey
    addition = ""
    if et:
        addition += " etymology"
    hotkey('command', 'c')
    sl(.3)
    clb2 = paste()
    query = clb2.lower() + addition
    query = query.replace(" ", "+")
    import os
    os.system(f"open 'https://www.google.com/search?q={query}'")

def da_search_selected():
    from richxerox import copy, paste
    from pyautogui import hotkey
    hotkey('command', 'c')
    sl(.3)
    clb2 = paste()
    query = clb2.lower().replace(" ", "+")
    import os
    os.system(f"open 'https://alesia.store/da.php?query={query}'")




def url2org(link="https://www.bitdegree.org/tutorials/c-vs-c-plus-plus/", idx=0, wd="/tmp/rgs_007/"):
    import multiprocessing
    from emacs import Emacs
    import shutil
    import subprocess
    import uuid
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse, urljoin
    import shutil
    from lib.base import P
    import re

    uu = uuid.uuid4()
    wd = f"/tmp/rgs_{uu}/"
    link_parsed = urlparse(link)
    print(f"url2org : {link}\n")
    os.mkdir(wd)
    html_file_name = os.path.join(wd, f"index_{idx}_{uu}.html")
    org_file_name = os.path.join(wd, f"site_{idx}_{uu}.org")
    org_file_name_refined = os.path.join(wd, f"site_refined_{idx}_{uu}.org")
    headers_mobile = {'User-Agent' : 'Mozilla/5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13B137 Safari/601.1'}
    response = requests.get(link, headers=headers_mobile)
    soup = BeautifulSoup(response.content, 'html.parser')
    if True or "www.gnu.org" in link:
        paragraphs_parent = soup
    else:
        paragraphs = soup.findAll("p")
        central_paragraph = paragraphs[len(paragraphs)//2]
        paragraphs_parent = central_paragraph.parent

    imgs = paragraphs_parent.findAll("img")
    imgs2download = []
    for img in imgs:
        if "src" not in img.attrs.keys():
            continue
        img_src = img.attrs["src"]
        if P(img_src).suffix not in [".png", ".jpg", ".gif", "jpeg"]: # maybe data-url - bad
            img.decompose()
            continue
        img_src_parsed = urlparse(img_src)
        if not img_src_parsed.scheme: # https link vs src="/a.png" link
            img_base = img_src # /a.png
            if not img_base.startswith("/"):
                img_base = f"/{img_base}"
            img_url = urljoin(f"{link_parsed.scheme}://{link_parsed.hostname}/", img_src)
            print(f"img_url: {img_url}")
        else:
            img_url = img_src
            img_base = "/" + P(img_url).name
        new_img_path = wd + img_base[1:]
        # import pdb; pdb.set_trace()
        img.attrs["src"] = new_img_path
        imgs2download.append((img_base, new_img_path, img_url))
    # Running pandoc
    open(html_file_name, "w").write(str(paragraphs_parent))
    p_pandoc = subprocess.Popen(f"pandoc --wrap=none -o {org_file_name}  -f html -t org {html_file_name}", shell = True)
    # Launching img downloads..
    for img_base, new_img_path, img_url in imgs2download:
        cmd_ = f"wget -O {new_img_path} '{img_url}'"
        print(f"Execing:\n{cmd_}")
        p = subprocess.Popen(cmd_, shell=True)
    p_pandoc.wait()
    foo = open(org_file_name).read()
    foo = "#+STARTUP: showall\n" + foo
    lines = foo.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith(f"[[{wd}"):
            new_lines.append("#+ATTR_ORG: :width 300")
        new_lines.append(line)
    open(org_file_name, "w").write("\n".join(new_lines))
    # loading file to emacs
    emacs = Emacs.client()
    emacs.eval(f'(find-file "{org_file_name}")')
    shutil.copy(org_file_name, org_file_name_refined)


def aca_q(query="apple", count=50, ecount=50, attributes="Id,DN,FP,CitCon", expr="Composite(J.JN=='nature')"):
    import http.client, urllib.request, urllib.parse, urllib.error, base64

    headers = {
        # Request headers
        'Ocp-Apim-Subscription-Key': 'e6ac04708eff483f8dd7c2a636ef554a',
    }

    params = urllib.parse.urlencode({
        "query": query,
    "complete": 0,
    "normalize": 1,
    "attributes": attributes,
    "offset": 0,
    "timeout": 2000,
    "count": count,
    "entityCount": ecount
    })

    try:
        conn = http.client.HTTPSConnection('api.labs.cognitive.microsoft.com')
        # conn.request("GET", f"/academic/v1.0/calchistogram?expr={expr}&{params}", "", headers)
        # conn.request("GET", f"/academic/v1.0/interpret?expr={expr}&{params}", "", headers)
        # qu = f"/academic/v1.0/interpret?query=%27enzyme%20design%20rosetta3%27&{params}"
        qu = f"/academic/v1.0/interpret?query={params}"
        print(f'Execing queuery: {qu}')
        conn.request("GET", qu, "", headers)
        response = conn.getresponse()
        data = response.read()
        to_eval = data.decode().replace("false", 'False')
        resp = eval(to_eval)
        conn.close()
        return resp
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

def aca_qw(*args, **kwargs):
    resp = aca_q(args, kwargs)
    articles = {}
    for interpretation in resp['interpretations']:
        for interp_rule in interpretation['rules']:
            for entity in interp_rule['output']['entities']:
                articles[entity['Id']] = entity

    len(articles.items())
    for k, v in articles.items():
        print(k, v['DN'])
