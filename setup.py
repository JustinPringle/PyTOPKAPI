import os
import subprocess

# BEFORE importing distutils, remove MANIFEST. distutils doesn't
# properly update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

from distutils.core import setup

MAJOR               = 0
MINOR               = 5
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

dev_version_py = 'pytopkapi/__dev_version.py'

def generate_version_py(filename):
    try:
        if os.path.exists(".git"):
            # should be a Git clone, use revision info from Git
            s = subprocess.Popen(["git", "rev-parse", "HEAD"],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            out = s.communicate()[0]
            GIT_REVISION = out.strip().decode('ascii')
        elif os.path.exists(dev_version_py):
            # should be a source distribution, use existing dev
            # version file
            from pytopkapi.__dev_version import git_revision as GIT_REVISION
        else:
            GIT_REVISION = "Unknown"
    except:
        GIT_REVISION = "Unknown"

    FULL_VERSION = VERSION
    if not ISRELEASED:
        # FULL_VERSION += '.dev-'
        FULL_VERSION += GIT_REVISION[:7]

    cnt = """\
# This file was autogenerated
version = '%s'
git_revision = '%s'
"""
    cnt = cnt % (FULL_VERSION, GIT_REVISION)

    f = open(filename, "w")
    try:
        f.write(cnt)
    finally:
        f.close()

    return FULL_VERSION, GIT_REVISION

if __name__ == '__main__':
    full_version, git_rev = generate_version_py(dev_version_py)

    setup(name='PyTOPKAPI',
          version=full_version,
          description='TOPKAPI hydrological model in Python',
          long_description="""\
PyTOPKAPI - a Python implementation of the TOPKAPI Hydrological model
=====================================================================

PyTOPKAPI is a BSD licensed Python library implementing the TOPKAPI
Hydrological model (Liu and Todini, 2002). The model is a
physically-based and fully distributed hydrological model, which has
already been successfully applied in several countries around the
world (Liu and Todini, 2002; Bartholomes and Todini, 2005; Liu et al.,
2005; Martina et al., 2006; Vischel et al., 2008, Sinclair and Pegram,
2010).

""",
          license='BSD',
          author='Scott Sinclair & Theo Vischel',
          author_email='theo.vischel@hmg.inpg.fr; sinclaird@ukzn.ac.za',
          url='http://sahg.github.io/PyTOPKAPI',
          download_url='http://github.com/sahg/PyTOPKAPI/downloads',
          packages=['pytopkapi',
                    'pytopkapi.parameter_utils',
                    'pytopkapi.results_analysis'
                    ],
          scripts=['scripts/run-grass-script', 'scripts/process-catchment'],
          classifiers=['Development Status :: 4 - Beta',
                       'License :: OSI Approved :: BSD License',
                       'Environment :: Console',
                       'Operating System :: OS Independent',
                       'Intended Audience :: Science/Research',
                       'Programming Language :: Python',
                       'Topic :: Scientific/Engineering',
                       ],
          )
