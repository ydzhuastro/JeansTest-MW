# README
### Notice

Please contact the authors of [Huang et al. (2020)](https://iopscience.iop.org/article/10.3847/1538-4365/ab994f) to get the fits file that contains the kinematics data of the Red Clump sample.

### Installation

1. Install Intel (R) OneAPI and icc compiler.
2. Install GSL: `./configure CC=icc; make -j 4; make install`.
3. Build and run: `make && ./directsum`.

### Execution

1. Use Huang2020/read_fits_to_GC.py to read the fits file and store the 3D kinematics data with the GC coordinates.
2. Use bin_statistics.py to bin the 3D kinematics data.
3. Compile PotentialSolver/directsum.c with `make`. 
4. Run `directsum` to get the outputs, including potential and rotation curves given by the models: Newtonian, Newtonian+DM, QUMOND, and MOG.
5. JeansTestPy/poi-jeanstest2-dsW22.py and JeansTestPy/poi-jeanstest2-dsM17.py can be used to execute the Jeans-equations tests.

If you have any questions, please contact the developers.
