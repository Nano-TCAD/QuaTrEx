# QuaTrEx
Quantum Transport at Extreme Scale

# Installation Instructions for Daint
1. Go on scratch folder

2. git clone https://github.com/Nano-TCAD/QuaTrEx.git

3. git checkout printegral_comparison

4. python -m venv --system-site-packages quatrex_env 

5. source quatrex_env/bin/activate

6. pip install -i https://pypi.anaconda.org/intel/simple numpy==1.26.4 (dieser Schritt is wichtig weil wir die das MKL backend von numpy brauchen)

7. cd QuaTrEx

8. pip install -e . 
