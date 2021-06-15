
def fes_thresh(xi, thres):
    return thres if xi > thres else xi
fes_thresh_v = np.vectorize(fes_thresh)
fig = plt.gcf()
fig.clear()
X3 = np.repeat(X_99[np.newaxis, :], X_99.shape[0], axis=0)
Y3 = np.repeat(Y_99[:, np.newaxis], Y_99.shape[0], axis=1)
Z3 = Z_99
Z3 = fes_thresh_v(Z3, 60)

label = "CV1-CV223"
supertitle = f"Поверхность свободной энергии $F(x)$, $F(x) = -k_B T \ln H(x)$"
xlabel = 'CV1: RMSD относительно стартовой структуры'
ylabel = 'CV2: Координация, лиганд-фермент'
fig.suptitle(supertitle)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
# help(plt.clim)
# imshowobj = plt.contourf(X3, Y3, Z3, 400, cmap="viridis")
# imshowobj = plt.contour(X3, Y3, Z3, 400, cmap="viridis")
imshowobj = plt.contourf(Z3, 1000, cmap="viridis")
# imshowobj = plt.imshow(data)
plt.sci(imshowobj)
plt.colorbar()
plt.show()
# plt.draw()
fig.savefig(d_eris['CV-test3.png'].s, dpi=300)
