# print('constants.py')
ln = 'L_PI35P'
if 'args' not in globals().keys():
    args = type('obj', (object,), {
        'lig_name': ln.lower(),
        'plants2hrx_thresh': 2.5
    })

args.quick = 0
rlx = True
Mg2sStr = False
lom2 = False
confsInOneFile = False
dbg = False
numDecs = 10
if args.quick:
    numDecs = 1
num_ed_cores = 12
lmTime = 50
rstOut = True
gr_fltr_vdow = 0.3
srv_md = True
mpcOptimize = True
cstRes = [121, 83]
# CG_PRMS
num_cg_cores = 8
controlRunType = ""
NumConf = 50
NumLinkAt = 3

lg_mnz_tries = 3
lg_mnz_etol = 1e-25
lg_mnz_ftol = 1e-25
lg_mnz_mxIts = 2000
lg_mnz_cst = 150.0
lg_e_thrs = 152
btn_clst_dst_thrs = 0.2
mpc_gnorm = 20
mpc_relscf = 10
mpc_scfcrt = "1.D-2"
noConfProtonCHI = True
noConfCHI = True
# CG_PRMS_END
# ED_PRMS_END
edFlags = {}
rlxFlags = {}
# RLX_FLAGS_START
rlxFlags['-extra_res_fa'] = './ri/cg/LAQ.params'
rlxFlags['-nstruct'] = '10'
rlxFlags['-relax:default_repeats'] = '2'
rlxFlags['-out:path:pdb'] = './ri/ss'
rlxFlags['-out:path:score'] = './ri/ss/rlx_score.sc'
rlxFlags['-relax:bb_move'] = 'true'
rlxFlags['-relax:constrain_relax_to_start_coords'] = True
rlxFlags['-relax:coord_constrain_sidechains'] = True
rlxFlags['-relax:ramp_constraints'] = 'false'
rlxFlags['-packing:ex1'] = True
rlxFlags['-packing:ex2'] = True
rlxFlags['-packing:use_input_sc'] = True
rlxFlags['-packing:flip_HNQ'] = True
rlxFlags['-packing:no_optH'] = 'false'
rlxFlags['-s'] = './ri/ss/lpla_lg2.pdb'
# RLX_FLAGS_END
# ED_FLAGS_START
edFlags['-resfile'] = './ri/msc/LIG.resfile'
edFlags['-packing:use_input_sc'] = True
edFlags['-packing:no_optH'] = 'false'
edFlags['-packing:flip_HNQ'] = True
edFlags['-packing:ex2'] = True
edFlags['-packing:ex1'] = True
edFlags['-out:level'] = '300'
edFlags['-nstruct'] = str(int(numDecs))
# edFlags['-enzdes:trans_magnitude'] = '0.9'
# edFlags['-enzdes:trans_magnitude'] = '0.01'
# edFlags['-enzdes:rot_magnitude'] = '?????'
# edFlags['-enzdes:rot_magnitude'] = ''
edFlags['-nstruct'] = str(int(numDecs))
# edFlags['-enzdes:favor_native_res'] = '7'
edFlags['-enzdes:favor_native_res'] = '1'
edFlags['-enzdes:detect_design_interface'] = True
# edFlags['-enzdes:design_min_cycles'] = '4'
edFlags['-enzdes:design_min_cycles'] = '1'
# edFlags['-enzdes:cut1'] = '6.0'
# edFlags['-enzdes:cut2'] = '8.0'
# edFlags['-enzdes:cut3'] = '10.0'
# edFlags['-enzdes:cut4'] = '12.0'
edFlags['-enzdes:cut1'] = '4.0'
edFlags['-enzdes:cut2'] = '6.0'
edFlags['-enzdes:cut3'] = '8.0'
edFlags['-enzdes:cut4'] = '10.0'
# edFlags['-enzdes:cut1'] = '1.0'
# edFlags['-enzdes:cut2'] = '2.0'
# edFlags['-enzdes:cut3'] = '2.0'
# edFlags['-enzdes:cut4'] = '2.0'
edFlags['-enzdes:cst_predock'] = True
edFlags['-enzdes:cst_design'] = True
edFlags['-enzdes:chi_min'] = False
edFlags['-enzdes:bb_min_allowed_dev'] = '1.0'
# edFlags['-enzdes:bb_min_allowed_dev'] = '0.0'
edFlags['-enzdes:bb_min'] = True
# edFlags['-cstfile'] = './ri/ss/cst.cst'

# ED_FLAGS_END
aaMap = {'DBB': 'T', 'GLN': 'Q', 'CAS': 'C', 'CME': 'C', 'CCS': 'C', 'ACR': 'X', 'GLZ': 'G', 'ILE': 'I', 'TIH': 'A',
         'UNK': 'X', 'MSE': 'M', 'PCA': 'E', 'OCS': 'C', 'CYS': 'C', 'HIS': 'H', 'SER': 'S', 'LYS': 'K', 'DAL': 'A',
         'PRO': 'P', 'LLP': 'K', 'GLU': 'E', 'SEC': 'U', 'NLE': 'L', 'ASN': 'N', 'ABA': 'A', 'CSO': 'C', 'ALY': 'K',
         'VAL': 'V', 'TPO': 'T', 'CSD': 'C', 'MEN': 'N', 'THR': 'T', 'DHA': 'S', 'ASP': 'D', 'CSX': 'C', 'YCM': 'C',
         'NH2': 'X', 'TRP': 'W', 'MLY': 'K', 'PTR': 'Y', 'GLY': 'G', 'SEP': 'S', 'PHE': 'F', 'ALA': 'A', 'MET': 'M',
         'ACE': 'X', 'PYL': 'O', 'LEU': 'L', 'ARG': 'R', 'HSE': 'S', 'DBU': 'T', 'TYR': 'Y'}
lplaSeq = "STLRLLISDSYDPWFNLAVEECIFRQMPATQRVLFLWRNADTVVIGRAQNPWKECNTRRMEEDNVRLARRSSGGGAVFHDLGNTCFTFMAGKPEYDKTISTSIVLNALNALGVSAEASGRNDLVVKTVEGDRKVSGSAYRETKDRGFHHGTLLLNADLSRLANYLNPDKKKLAAKGITSVRSRVTNLTELLPGITHEQVCEAITEAFFAHYGERVEAEIISPNKTPDLPNFAETFARQSSWEWNFGQAPAFSHLLDERFTWGGVELHFDVEKGHITRAQVFTDSLNPAPLEALAGRLQGCLYRADMLQQECEALLVDFPEQEKELRELSAWMAGAVR"
pmlCpkRgb = {'C': [200, 200, 200], 'H': [255, 255, 255], 'O': [240, 0, 0], 'N': [143, 143, 255], 'P': [255, 165, 0],
             'S': [255, 200, 50]}
pmlCpk = {'C': [0.7843137254901961, 0.7843137254901961, 0.7843137254901961], 'H': [1.0, 1.0, 1.0],
          'O': [0.9411764705882353, 0.0, 0.0], 'N': [0.5607843137254902, 0.5607843137254902, 1.0],
          'P': [1.0, 0.6470588235294118, 0.0], 'S': [1.0, 0.7843137254901961, 0.19607843137254902]}

doBashStr = "".join(["""gmx grompp -f em.mdp -c solv_ions.gro -p topol.top -o em.tpr -maxwarn 1
gmx mdrun -v -deffnm em -o em.trr  -c em.gro -nt {0} 
echo -e '0 & ! a H*\\nq' |  gmx make_ndx -f unl.acpype/unl_GMX.gro -o index_lig.ndx
echo 3 | gmx genrestr -f unl.acpype/unl_GMX.gro -n index_lig.ndx -o posre_lig.itp -fc 1000 1000 1000
# TOPOL  REWRITED
echo -e "4 | 2\\nq" | gmx make_ndx -f em.gro -o index.ndx """, """
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -n index.ndx -o nvt.tpr -maxwarn 1
gmx mdrun -deffnm nvt  -nt {0} """, """
gmx grompp -f npt.mdp -c nvt.gro -t nvt.cpt -r nvt.gro -p topol.top -n index.ndx -o npt.tpr -maxwarn 1
gmx mdrun -deffnm npt  -nt {0} """, """
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -n index.ndx -o md.tpr -maxwarn 1
gmx mdrun -deffnm md -nt {0} """])

liuwPlns = [([-21.494165420532227, -44.19559097290039, 5.1895432472229],
             [-23.64804458618164, -37.82337188720703, 10.791091918945312],
             [-29.385433197021484, -41.542476654052734, 14.23637580871582], 0),
            ([-20.61522674560547, -44.02548599243164, 7.253364086151123],
             [-15.660621643066406, -45.562339782714844, 21.71814727783203],
             [-22.041624069213867, -33.09431076049805, 21.008342742919922], 0), ]

confsInOneFile = False
liuCatRes = [75, 76, 77, 78, 79, 83, 121, 133, 135, 136, 137, 151, 153, 161, 165, 179, 180, 181, 182, 184]
catAmpRes = [133, 180, 182, 78, 83, 151]
lig_amp_list = {}
lig_amp_list["rrf"] = ['C15', 'C14', 'P1', 'N2', 'N3', 'O3', 'N4', 'O4', 'N5', 'N6', 'O6', 'O7', 'O8', 'O9', 'O10',
                       'O11', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27']
lig_amp_list["bff"] = ['P1', 'N3', 'O3', 'N4', 'O4', 'N5', 'O5', 'N6', 'O6', 'N7', 'O7', 'O8', 'O9', 'C16', 'C17',
                       'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25']
lig_amp_list["laq"] = ['P1', 'N3', 'O3', 'N4', 'O4', 'N5', 'O5', 'N6', 'O6', 'N7', 'O7', 'O8', 'O9', 'C16', 'C17',
                       'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25']
liu_mut = [20, 149, 147, 37]
sphCstMolIdxs = [6, 16, 17, 18, 10, 0]
sphCstAts = ["O8", "C22", "C23", "C25", "N6", "O7"]
sphCstRad = [2, 2, 2, 2, 2, 2]
torsAtms = [22, 20, 19, 18, 17, 16, 15, 45]
torsAtmsLists = [torsAtms[i:i + 4] for i in range(len(torsAtms) - 3)]
mopac_key = '11070687a54814547'

aaMap = {'DBB': 'T', 'GLN': 'Q', 'CAS': 'C', 'CME': 'C', 'CCS': 'C', 'ACR': 'X', 'GLZ': 'G', 'ILE': 'I', 'TIH': 'A',
         'UNK': 'X', 'MSE': 'M', 'PCA': 'E', 'OCS': 'C', 'CYS': 'C', 'HIS': 'H', 'SER': 'S', 'LYS': 'K', 'DAL': 'A',
         'PRO': 'P', 'LLP': 'K', 'GLU': 'E', 'SEC': 'U', 'NLE': 'L', 'ASN': 'N', 'ABA': 'A', 'CSO': 'C', 'ALY': 'K',
         'VAL': 'V', 'TPO': 'T', 'CSD': 'C', 'MEN': 'N', 'THR': 'T', 'DHA': 'S', 'ASP': 'D', 'CSX': 'C', 'YCM': 'C',
         'NH2': 'X', 'TRP': 'W', 'MLY': 'K', 'PTR': 'Y', 'GLY': 'G', 'SEP': 'S', 'PHE': 'F', 'ALA': 'A', 'MET': 'M',
         'ACE': 'X', 'PYL': 'O', 'LEU': 'L', 'ARG': 'R', 'HSE': 'S', 'DBU': 'T', 'TYR': 'Y'}
lplaSeq = "STLRLLISDSYDPWFNLAVEECIFRQMPATQRVLFLWRNADTVVIGRAQNPWKECNTRRMEEDNVRLARRSSGGGAVFHDLGNTCFTFMAGKPEYDKTISTSIVLNALNALGVSAEASGRNDLVVKTVEGDRKVSGSAYRETKDRGFHHGTLLLNADLSRLANYLNPDKKKLAAKGITSVRSRVTNLTELLPGITHEQVCEAITEAFFAHYGERVEAEIISPNKTPDLPNFAETFARQSSWEWNFGQAPAFSHLLDERFTWGGVELHFDVEKGHITRAQVFTDSLNPAPLEALAGRLQGCLYRADMLQQECEALLVDFPEQEKELRELSAWMAGAVR"
aaRevMap = {'A': 'ALA', 'C': 'CYS', 'E': 'GLU', 'D': 'ASP', 'G': 'GLY', 'F': 'PHE', 'I': 'ILE', 'H': 'HIS', 'K': 'LYS',
            'M': 'MET', 'L': 'LEU', 'O': 'HYP', 'N': 'ASN', 'Q': 'GLN', 'P': 'PRO', 'S': 'SER', 'R': 'ARG', 'U': 'GLP',
            'T': 'THR', 'W': 'TRP', 'V': 'VAL', 'Y': 'TYR'}
aaMap2 = {'ILE': 'I', 'GLN': 'Q', 'HYP': 'O', 'GLY': 'G', 'GLP': 'U', 'GLU': 'E', 'CYS': 'C', 'ASP': 'D', 'SER': 'S',
          'LYS': 'K', 'PRO': 'P', 'ASN': 'N', 'VAL': 'V', 'THR': 'T', 'HIS': 'H', 'TRP': 'W', 'PHE': 'F', 'ALA': 'A',
          'MET': 'M', 'LEU': 'L', 'ARG': 'R', 'TYR': 'Y'}
pmlCpkRgb = {'C': [200, 200, 200], 'H': [255, 255, 255], 'O': [240, 0, 0], 'N': [143, 143, 255], 'P': [255, 165, 0],
             'S': [255, 200, 50]}
pmlCpk = {'C': [0.7843137254901961, 0.7843137254901961, 0.7843137254901961], 'H': [1.0, 1.0, 1.0],
          'O': [0.9411764705882353, 0.0, 0.0], 'N': [0.5607843137254902, 0.5607843137254902, 1.0],
          'P': [1.0, 0.6470588235294118, 0.0], 'S': [1.0, 0.7843137254901961, 0.19607843137254902]}

hrx_muts = {17: "A", 20: "A", 35: "A", 37: "A", 87: "A", 147: "A", 149: "G", 85: "A"}
aaaV1 = [17, 20, 37, 44, 68, 70, 71, 72, 85, 87, 138, 140, 147, 149]
aaaV2 = aaaV1 + [35, 16]

cst_lgd_laq_and_lpla_lg_list = [['151', 'THR', 'CA', 'CB', 'OG1', 13, 18, 19],
                                ['83', 'ASN', 'CB', 'CG', 'OD1', 19, 18, 13],
                                ['78', 'PHE', 'C', 'CA', 'N', 19, 18, 13],
                                ['182', 'SER', 'CA', 'C', 'O', 11, 10, 12],
                                ['180', 'VAL', 'C', 'CA', 'N', 2, 0, 3],
                                ['75', 'GLY', 'C', 'CA', 'N', 3, 0, 1],
                                ['121', 'ASN', 'CB', 'CG', 'ND2', 1, 0, 3],
                                ['133', 'LYS', 'CD', 'CE', 'NZ', 24, 23, 3]]

# imp1 = *map(lambda x: int(x[0]), cst_lgd_laq_and_lpla_lg_list), + (70)

natCntsList_4tvwH_A = [
    [151, "HG1", "N1A"],
    [83, "OD1", ("H10", "H11")],
    [78, "O", ("H10", "H11")],
    [78, "H", "N7A"],
    [182, "O", "H7"],
    [182, "H", "O2'"],
    [180, "H", "O2S"],
    [133, ("HZ1", "HZ2", "HZ3"), "O46"],
]
natCntDists = [([151, 'HG1', 'N1A'], 2.09), ([83, 'OD1', 'H10'], 2.5), ([78, 'O', 'H11'], 2.12),
               ([78, 'H', 'N7A'], 2.12), ([182, 'O', 'H7'], 2.70), ([182, 'H', "O2'"], 2.74), ([180, 'H', 'O2S'], 1.99),
               ([133, 'HZ1', 'O46'], 2.28)]
laq_hydr_cnts = [
    [(151, "OG1"), 13],
    [19, (83, "OD1")],
    [19, (78, "O")],
    [(78, "N"), 20],
    [11, (182, "O")],
    [(182, "N"), 11],
    [(180, "N"), 2],
    [(133, "NZ"), 24],
]
laq_hydr_cnts = [
    [(151, "OG1"), 13],
    [19, (83, "OD1")],
    [19, (78, "O")],
    [(78, "N"), 20],
    [11, (182, "O")],
    [(182, "N"), 11],
    [(180, "N"), 2],
    [(133, "NZ"), 24],
]
laq_hydr_cnts = [
    [(151, "OG1"), 2],
    [8, (83, "OD1")],
    [8, (78, "O")],
    [(78, "N"), 11],
    [12, (182, "O")],
    [(182, "N"), 12],
    [(180, "N"), 15],
    [(133, "NZ"), 20],
]

bff_hydr_cnts = [
    [6, (16, "O")],
    [6, (87, "OG1")],
    [(79, "NE*"), 17, ],
]
rrf_hydr_cnts = [
    [(79, "NE*"), 11, ],  # [6, ("NH*")]!!!!! ANIONS!!!!!!!
]
oxy_kws = ([{'acceptors': ['O'],
             'donors': ['O'],
             'selection1': '(resname UNL) and (name O)',
             'selection1_type': 'donor',
             'selection2': '((resid 16) and (name O))',
             'verbose': True},
            {'acceptors': ['OG1'],
             'donors': ['O'],
             'selection1': '(resname UNL) and (name O)',
             'selection1_type': 'donor',
             'selection2': '((resid 87) and (name OG1))',
             'verbose': True},
            {'acceptors': ['O9'],
             'donors': ['NE2'],
             'selection1': '((resid 79) and (name NE2))',
             'selection1_type': 'donor',
             'selection2': '(resname UNL) and (name O9)',
             'verbose': True}],)
liuwCntList = [
    [151, "HG1", "N4"],
    [83, "OD1", ("H25", "H24")],
    [78, "O", ("H25", "H24")],
    [78, "H", "N2"],
    [182, "O", "H21"],
    [182, "H", "O9"],
    [180, "H", "O10"],
    [133, ("HZ1", "HZ2", "HZ3"), "O3"],
]
cntRoutingDict = {
    "../LIU_1_1/": liuwCntList,
    "../BFF_1_8/": [
        [151, "HG1", "N5"],
        [83, "OD1", ("H29", "H28")],
        [78, "O", ("H29", "H28")],
        [78, "H", "N3"],
        [182, "O", "H25"],
        [182, "H", "O7"],
        [180, "H", "O3"],
        [133, ("HZ1", "HZ2", "HZ3"), "O1"],
    ],
    "../CRY_1_1/": liuwCntList,
    "": [
        [151, 'HG1', 'N5'],
        [83, 'OD1', ('H26', 'H27')],
        [78, 'O', ('H26', 'H27')],
        [78, 'H', 'N3'],
        [182, 'O', 'H23'],
        [182, 'H', 'O7'],
        [180, 'H', 'O3'],
        [133, ('HZ1', 'HZ2', 'HZ3'), 'O1']
    ]
}
pml_clrs = ['green', 'blue', 'magenta', 'yellow', 'brown', 'salmon', 'actinium', 'aluminum', 'americium', 'antimony',
            'aquamarine', 'argon', 'arsenic', 'astatine', 'barium',
            'berkelium', 'beryllium', 'bismuth', 'black', 'bluewhite', 'bohrium', 'boron', 'brightorange',
            'bromine', 'cadmium', 'calcium', 'californium', 'carbon', 'cerium', 'cesium', 'chartreuse',
            'chlorine', 'chocolate', 'chromium', 'cobalt', 'copper', 'curium', 'cyan', 'darksalmon', 'dash',
            'deepblue', 'deepolive', 'deeppurple', 'deepsalmon', 'deepsalmon', 'deepteal', 'density', 'deuterium',
            'dirtyviolet', 'dubnium', 'dysprosium', 'einsteinium', 'erbium', 'europium', 'fermium', 'firebrick',
            'fluorine', 'forest', 'francium', 'gadolinium', 'gallium', 'germanium', 'gold', 'gray',
            'greencyan', 'grey', 'hafnium', 'hassium', 'helium', 'holmium', 'hotpink', 'hydrogen', 'indium',
            'iodine', 'iridium', 'iron', 'krypton', 'lanthanum', 'lawrencium', 'lead', 'lightblue', 'lightmagenta',
            'lightorange', 'lightpink', 'lightteal', 'lime', 'limegreen', 'limon', 'lithium', 'lonepair',
            'lutetium', 'magnesium', 'manganese', 'marine', 'meitnerium', 'mendelevium', 'mercury',
            'molybdenum', 'neodymium', 'neon', 'neptunium', 'nickel', 'niobium', 'nitrogen', 'nobelium', 'olive',
            'orange', 'osmium', 'oxygen', 'palecyan', 'palegreen', 'paleyellow', 'palladium', 'phosphorus', 'pink',
            'platinum', 'plutonium', 'polonium', 'potassium', 'praseodymium', 'promethium', 'protactinium',
            'pseudoatom', 'purple', 'purpleblue', 'radium', 'radon', 'raspberry', 'red', 'rhenium', 'rhodium',
            'rubidium', 'ruby', 'ruthenium', 'rutherfordium', 'samarium', 'sand', 'scandium',
            'seaborgium', 'selenium', 'silicon', 'silver', 'skyblue', 'slate', 'smudge', 'sodium', 'splitpea',
            'strontium', 'sulfur', 'tantalum', 'teal', 'technetium', 'tellurium', 'terbium', 'thallium', 'thorium',
            'thulium', 'tin', 'titanium', 'tungsten', 'tv_blue', 'tv_green', 'tv_orange', 'tv_red', 'tv_yellow',
            'uranium', 'vanadium', 'violet', 'violetpurple', 'warmpink', 'wheat', 'white', 'xenon',
            'yelloworange', 'ytterbium', 'yttrium', 'zinc', 'zirconium']

keyboardMap = {'semicolon': 'Cyrillic_zhe', 'period': 'Cyrillic_yu', 'comma': 'Cyrillic_be',
               'bracketright': 'Cyrillic_hardsign', 'bracketleft': 'Cyrillic_ha', 'x': 'Cyrillic_che',
               'grave': 'Cyrillic_io', 'a': 'Cyrillic_ef', 'c': 'Cyrillic_es', 'b': 'Cyrillic_i', 'e': 'Cyrillic_u',
               'd': 'Cyrillic_ve', 'g': 'Cyrillic_pe', 'f': 'Cyrillic_a', 'i': 'Cyrillic_sha', 'h': 'Cyrillic_er',
               'k': 'Cyrillic_el', 'j': 'Cyrillic_o', 'm': 'Cyrillic_softsign', 'l': 'Cyrillic_de',
               'o': 'Cyrillic_shcha', 'n': 'Cyrillic_te', 'q': 'Cyrillic_shorti', 'p': 'Cyrillic_ze',
               's': 'Cyrillic_yeru', 'r': 'Cyrillic_ka', 'u': 'Cyrillic_ghe', 't': 'Cyrillic_ie', 'w': 'Cyrillic_tse',
               'v': 'Cyrillic_em', 'y': 'Cyrillic_en', 'apostrophe': 'Cyrillic_e', 'z': 'Cyrillic_ya'}
pymol_com = """extract l_{0}, organic and {0}
extract mg_{0}, e. mg and {0}
extract pr_{0}, polymer and {0} and br. (* w. 18 of r. {1})
delete {0}
set surface_carve_cutoff, 4.5
set surface_carve_selection, all
set surface_carve_normal_cutoff, -0.1
show surface, pr_{0} within 8 of l_{0}
set two_sided_lighting
set transparency, 0.5
show sticks, l_{0}
set surface_color, white
hide lines, pr_{0}
show cartoon, br. (* w. 20 or r. RRF) and pr_{0}
set cartoon_transparency, 0.7
set cartoon_fancy_helices, 1
set cartoon_fancy_sheets, 1"""
specPlns = [([-24.969999313354492, -36.43299865722656, 16.691999435424805],
             [-22.108999252319336, -41.03799819946289, 14.267999649047852],
             [-25.606000900268555, -41.67399978637695, 9.034000396728516], 1),
            ([-27.003000259399414, -37.02399826049805, 13.685999870300293],
             [-23.201000213623047, -45.595001220703125, 11.057999610900879],
             [-18.945999145507812, -42.45000076293945, 7.76200008392334], 0)]
bff_acid_smi = '[B-]1(F)(F)C2=CC(O)=CC=C2C=C3[N+]1=C(C)N(CCCCC(=O)O)C3(=O)'
bff_flur_smi = '[B-]1(F)(F)C2=CC(O)=CC=C2C=C3[N+]1=C(C)NC3(=O)'
bff_lig_smi = '[B-]1(F)(F)C2=CC(O)=CC=C2C=C3[N+]1=C(C)N(CCCCC(=O)O[P@@](=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1cnc2c1ncnc2N)[O-])C3(=O)'
rrf_acid_smi = 'C1=CC2=C(C=C1O)OC3=CC(=O)C(=CC3=N2)CCCCC(=O)O'
rrf_flur_smi = 'C1=CC2=C(C=C1O)OC3=CC(=O)C(=CC3=N2)'
rrf_lig_smi = 'C1=CC2=C(C=C1O)OC3=CC(=O)C(=CC3=N2)CCCCC(=O)O[P@@](=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1cnc2c1ncnc2N)[O-]'
laq_acid_smi = 'C1CSSC1CCCCC(=O)O'
laq_flur_smi = 'C1CSS[C@@H]1'
laq_lig_smi = 'C1CSS[C@@H]1CCCCC(=O)O[P@@](=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1cnc2c1ncnc2N)[O-]'
cou_acid_smi = 'C1=C(O)C=C2C(=C1)C=C(C(=O)NCCCCC(=O)O)C(=O)O2'

for j in range(1, 7):
    cntRoutingDict["../CRY_1_{}/".format(j)] = liuwCntList



args.cv_table_stride = 1
# assert len(args.lig_name) == 3
lnu = args.lig_name
charge = 0
chain = 'A'
pH = 7.0








if ln == 'bff':
    args.cat_aa = liuCatRes + [133, 70]
    args.cat_atoms = ['C15', 'O2']
    args.hydroxyl_name = 'O1'
    # args.complex_pdb = cur_ser.sdir['dec_path_RR3_3_deprot.pdb']

    charge = -2
    multiplicity = 1
if ln == 'rrf':
    # args.cat_aa = liuCatRes + [133, 70]
    args.cat_aa_literature = [133]
    args.no_mut_aa_literature = liuCatRes + [70]
    # args.hydroxyl_name = 'O15'
    # args.cat_atoms_npt_gro = ['C16', 'O3']
    args.S_cat_atoms_pdb_names = ['C10', 'O10']
    # X can be fed in two ways:
    # pdb file with notations
    # args.X_pdb = None
    args.X_pdb = '/home/domain/data/rustam/dgfl/str/RRF.pdb'
    args.X_cat_atoms_pdb_names = ['C44', 'O46']
    # or smiles with ranks:
    args.X_smiles = "C1=CC2=C(C=C1O)OC3=CC(=O)C(=CC3=N2)CCCCC(=O)O[P@@](=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1cnc2c1ncnc2N)[O-]"
    args.X_cat_atoms_ranks = [20, 21] # BAD CODE: check order carbon - oxygen

    # args.complex_pdb = '/mnt/storage/rustam/dgfl/str/dec_path_RR3_3_deprot.pdb'
    args.ES_pdb = '/home/domain/data/rustam/dgfl/str/3a7r.pdb'

    charge = -2
    multiplicity = 1
    args.hrx_atom_selection_thresh = 5
    args.num_threads_em_nvt_npt = 1
if ln == 'coc':
    args.lig_smiles = 'CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC'
    # args.cat_aa = ?
    args.cat_atoms = ['C2', 'O3']
    args.lig_pdb = '/home/domain/data/rustam/dgfl/str/COC.pdb'
    # args.complex_pdb = cur_ser.sdir['dec_path_RR3_3_deprot.pdb']
    charge = 0
    multiplicity = 1
if ln == 'L_PI35P':
    # TODO write function to infer ids
    # args.no_mut_aa_literature = []
    args.lit_at_pairs = [
        [[17,  'SG',  'C'],  [3632, 'P3']],
        # [[129, 'OD1', 'D'],  [3632, 'O31']],
        # [[18,  'NH1', 'R'],  [3632, 'O32']],
        # [[18,  'NH2', 'R'],  [3632, 'O33']],
    ]
    args.cat_aa = [*set([i[0][0] for i in args.lit_at_pairs])]
    args.lig_smiles = ''
    args.cat_atoms = []
    args.num_threads_em_nvt_npt = 12
    args.net_charge = -5
    # args.conv_len_thres = 30000
    args.conv_len_thres = 20
    args.B_len_thres = 100000
    args.run_hrx_gpu_id = 2
    try:
        args.XH_pdb = d__sdir["L_PI35P_E.pdb"]
        args.E_pdb = d__sdir["1xww_prot.pdb"]
    except Exception:
        pass


args.cat_dist_restr_selection_thresh = 4
args.sc_heating_atom_selection_thresh = 5
args.root_name = 'A1'
# args.stage2kids = [
#     [10, 'PL'],
#     [1, 'HR'],
#     [10, 'ED'],
#     [1, 'HR'],
#     [10, 'PL'],
# ]

args.plants_speed = "speed1"
# args.aco_ants = 20
if args.quick:
    args.plants_speed = "speed4"
args.stage2kid_names = ['PL', 'HR', 'ED', 'HR']
args.stage2kids = [
    [1, 'PL'],
    [1, 'HR'],
    # [120, 'ED'],
    [85, 'ED'],
    [1, 'HR'],
    [1, 'PL'],
]
# args.gmx_path = 'module load gromacs/2020.3-gcc-gpu; gmx'
# args.gmx_path = 'gmx'
# args.gmx_str_mod = 'module load gmx/2019.6-threadedmpi-cuda-single-pm2.6-gpu; /home/domain/data/prog/gromacs_2018.6_mpi_single_gpu_pm2.5.1/bin/gmx_mpi'
# args.gmx_path = 'gmx'
args.gmx_path = 'gmx_mpi_d'



