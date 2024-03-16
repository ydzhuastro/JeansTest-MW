# README
### Notice

Please contact the authors of [Huang et al. (2020)](https://iopscience.iop.org/article/10.3847/1538-4365/ab994f), or visit https://zenodo.org/record/3875974 to get the fits file that contains the kinematics data of the Red Clump sample. 

### Installation

1. Install Intel® OneAPI and icc compiler. (Check installation of [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) and [Intel$^®$ oneAPI HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit-download.html). ) After install tion, run `source /opt/intel/oneapi/setvars.sh` to set the environment variables.
   
   **Note:** if you install the latest verions of icc compiler (e.g., version 2024.0.1), you will find that in the oneAPI suite, Intel has been transitioning to new compiler tools, namely `icx` for `C` and `ifx` for `Fortran`, aligning with the LLVM compiler infrastructure. 
2. Install GSL: `./configure CC=icc; make -j 4; make install` (or `CC=icx` for recent version of Intel C compiler).
3. Build and run: `make && ./directsum`.

### Execution

1. Use Huang2020/read_fits_to_GC.py to read the fits file and store the 3D kinematics data with the GC coordinates.
2. Use bin_statistics.py to bin the 3D kinematics data.
3. Compile PotentialSolver/directsum.c with `make`. 
   
   **Note:** as mentioned in the installation section, if you use the latest version of icc compiler, you should replace `icc` with `icx` in the Makefile. And also specify your own path of the GSL library in the Makefile.
4. Run `directsum` to get the outputs, including potential and rotation curves given by the models: Newtonian, Newtonian+DM, QUMOND, and MOG.
5. JeansTestPy/poi-jeanstest2-dsW22.py and JeansTestPy/poi-jeanstest2-dsM17.py can be used to execute the Jeans-equations tests.

If you have any questions, please contact the developers.