ddfs = []
for kid in self.kids:
    if kid.decoys:
        print(kid)
        try:
            ddfs.append([kid, kid.make_visu_df()])
        except Exception as e:
            pass
    # kid.prepare_kids()
    # kid.launch_kids()
print(ddfs)
for dff in ddfs:
    print(dff[0].s, dff[1])
import itertools
sets = set(itertools.chain([set(i.columns) for i in visus]))
kar = [set(i.columns) for i in visus]
flat_list = [item for sublist in kar for item in sublist]
import pandas as pd
colsd = ['I16', 'R18', 'W49', 'N50', 'D129']
colsd = list(set(flat_list))
colsd
colsd.sort(key=lambda x: int(x[1:]))
colsd

for j in colsd:
    srв

df = pd.DataFrame(columns=colsd)
for dff in ddfs:
    df_ = dff[1]
    itms = [*df_.iteritems()]
    df.loc[dff[0].name + "_1"] = [df_.loc["lpla_lg2_0__DE_1", v] if v in df_.columns else "" for i, v in enumerate(colsd)]
    df.loc[dff[0].name + "_2"] = [df_.loc["lpla_lg2_0__DE_1", v] if v in df_.columns else "" for i, v in enumerate(colsd)]

df
for i in range(df.shape[0]):
    # for idx in range(df.shape[1]-1):
    strs.append(" | ".join([str(df.iloc[i, j]) for j in range(df.shape[1])]))

strs = []
for index, row in df.iterrows():
    print(index, row)
    strs.append(" | ".join([row[j] for j in df.columns[1:]]))
    print(row['c1'], row['c2'])

strs

df.columns

# for i in df.iterrows():

#     print(i)
#     print(i[])





# for k, v in df.items():
#     print(k)


# import pandas.table
# import matplotlib.pyplot as plt
# import pandas as pd
# from pandas.table.plotting import table # EDIT: see deprecation warnings below

# ax = plt.subplot(111, frame_on=False) # no visible frame
# ax.xaxis.set_visible(False)  # hide the x axis
# ax.yaxis.set_visible(False)  # hide the y axis

# table(ax, df)  # where df is your data frame

# plt.savefig(d_eris['aaa.png'])
# d_eris['aaa.png']


#     "I16" in df_.columns

# print(df)

# dir(df_)

#     # all_columns = all_columns or  set(list(dff[1].columns))




# all_decs = []
# for kid in self.kids:
#     if kid != self.kid:
#         if kid.decoys:
#             all_decs.append(kid.decoys)

# idx = 0
# for al_dec in all_decs:
#     for deco in al_dec:
#         idx += 1
#         deco_s_new = f"lpla_lg2_0__DE_{idx}.pdb"
#         print(deco_s_new)
#         deco.copy_to(self.kid.ro[deco_s_new])



# visu_df = self.kid.make_visu_df()



self.make_and_analyze_cv_table()



















self




n_dirs, dir_nam = args.stage2kids[self.stage]
n_dirs, dir_nam = 100, 'ED'
for dir_idx in range(200, 200+n_dirs):
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

        self





if type(self.kid) == E:
    for edd in self.kids:
        protein.write(edd.ss['prot_from_traj.pdb'])
        ligand.write(edd.cg['lig_from_traj.pdb'])

        self
