# README
1. please contact the authors of Huang et al. (2020) to get the fits file that contains the kinematics data of the Red Clump sample.
2. use Huang2020/read_fits_to_GC.py to read the fits file and store the 3D kinematics data with the GC coordinates.
3. use bin_statistics.py to bin the 3D kinematics data.
4. compile PotentialSolver/directsum.c with `make`. 
5. run `directsum` to get the outputs, including potential and rotation curves given by the models: Newtonian, Newtonian+DM, QUMOND, and MOG.
6. JeansTestPy/poi-jeanstest2-dsW22.py and JeansTestPy/poi-jeanstest2-dsM17.py can be used to execute the Jeans-equations tests.

If you have any questions, please contact the developers.
