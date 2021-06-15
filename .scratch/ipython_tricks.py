# OUTDATED FILE 
print('ipython_tricks.py')
# BAD FILE, bad code. To warn you may use stderr

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


class Av2:
    pass
