# -*- mode: python -*-

block_cipher = None

a = Analysis(['scripts/nlsam_denoising'],
             pathex=['./'],
             datas=None,
             hiddenimports=['scipy.special._ufuncs_cxx',
                            'scipy.linalg.cython_blas',
                            'scipy.linalg.cython_lapack',
                            'scipy.integrate',
                            'cython_gsl',
                            'nlsam.utils',
                            'nlsam.stabilizer',
                            'dipy.core.ndindex',
                            'multiprocessing',
                            'spams',
                            'scipy.special',
                            'scipy.integrate.quadrature',
                            'scipy.integrate.odepack',
                            'scipy.integrate._odepack',
                            'scipy.integrate.quadpack',
                            'scipy.integrate._quadpack',
                            'scipy.integrate._ode',
                            'scipy.integrate.vode',
                            'scipy.integrate._dop',
                            'scipy.integrate.lsoda'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='nlsam_denoising',
          debug=False,
          strip=False,
          upx=True,
          console=True )