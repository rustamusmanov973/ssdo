conda create -n py37 python=3.7.3
conda activate py37
conda config --append channels conda-forge
conda config --append channels defaults

conda config --append channels  rdkit
conda config --append channels openbabel
conda config --append channels salilab
conda config --append channels psi4
conda config --append channels gerrymandr
conda config --append channels anaconda
conda config --append channels schrodinger
conda config --append channels omnia
conda config --append channels acellera

conda install -c conda-forge acpype
conda install -c bioconda biobb_md
conda install -y acpype biopython braceexpand cached-property docutils fabric fpdf griddataformats gromacswrapper gsd h5py humanize invoke ipdb ipysheet libtmux line-profiler m2r mdanalysis mmtf-python mock msgpack notebook-xterm numkit parso plotly pylibdmtx pymysql python-emacs retrying sqlalchemy sysrsync toml acemd acemd-examples acemd3 ambermini anaconda-client ansible ansible-lint anyconfig apbs appdirs arrow asn1crypto attrs babel backcall bcrypt bhmm binaryornot biopandas blas bleach blosc boto bzip2 c-ares ca-certificates cairo cerberus certifi cffi chain chardet click click-completion clyent colorama conda-package-handling cookiecutter cryptography curl cycler cython dbus debtcollector decorator deepdiff defusedxml dftd3 dill dkh dogpile.cache entrypoints et_xmlfile expat extras fftw3f fixtures flake8 fontconfig freemol freetype future gau2grid gawk gdma glew glib gsl gst-plugins-base gstreamer hdf4 hdf5 hdf5-1820 htmd-pdb2pqr httplib2 icu idna importlib_metadata inflect intel-openmp ipykernel ipython ipython_genutils ipywidgets iso8601 jaraco.itertools jdcal jedi jinja2 jinja2-time jmespath joblib jpeg json5 jsonpatch jsonpickle jsonpointer jsonschema jupyter_client jupyter_console jupyter_core jupyterlab jupyterlab_server keystoneauth1 kiwisolver krb5 libarchive libblas libboost libcblas libcurl libedit libev libffi libgcc libgcc-ng libgfortran libgfortran-ng libglu libholoplaycore libint liblapack libllvm8 libnghttp2 libopenblas libpng libsodium libssh2 libstdcxx-ng libtiff libuuid libxc libxcb libxml2 llvmlite lz4-c lzo markupsafe match matplotlib mccabe mengine mistune mkl mkl-service mkl_fft mkl_random molecule moleculekit monotonic more-itertools mpeg_encode msmtools mtz2ccp4_px multiprocess munch natsort nbconvert nbformat ncurses netaddr netifaces networkx nglview nlopt nodejs notebook numba numexpr numpy numpy-base olefile openbabel openmm openpyxl openssl os-client-config os-service-types oslo.i18n oslo.serialization oslo.utils packaging pandas pandoc pandocfilters paramiko parmed pathlib2 pathos pathspec pbr pcmsolver pcre pdb2pqr pexpect pickleshare pillow pint pip pixman pluggy plumed plumed1 plumed2 pmw pox poyo ppft prettytable prometheus_client prompt_toolkit propka protocolinterface psfgen psutil ptyprocess py py-boost py-plumed pycodestyle pycosat pycparser pycrypto pydantic pyflakes pygments pynacl pyopenssl pyparsing pyqt pyrsistent pysocks pytest python python-dateutil python-gilt python-ironicclient python-libarchive-c python_abi pytz pyyaml pyzmq qcelemental qt rdkit readline requests requestsexceptions rigimol ruamel.yaml ruamel.yaml.clib ruamel_yaml scikit-learn scipy send2trash setproctitle setuptools sh shade shellingham simint simplejson simpletraj sip six sqlite stevedore tabulate terminado testinfra testpath testtools tk tornado tqdm traitlets tree-format urllib3 vina wcwidth webencodings wheel whichcraft widgetsnbextension wrapt xdrfile xz yaml yamllint zeromq zipp zlib zstd ripgrep

pip install ipython-blocking pybel gromacs sysrsync

# modeller conda package is very bad. it will cause errors, that are however easy to amend
