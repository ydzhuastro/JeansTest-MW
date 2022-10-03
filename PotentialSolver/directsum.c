#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

double kpc = 3.085678E19;
double MSun = 1.989E30;
double Myr = 3.1536E13;
double G = 4.493531540994E-12;
double KGpM3tGeVpCM3 = 5.6095888E20;
double toGeV = 3.797641792E-08;
double MpSStKPCpMYRMYR = 32230171.002936795; // 1.0/(kpc/Myr/Myr);
double KMpStKPCpMYR = 0.0010220120181042868; // 1000.0/(kpc/Myr);
double KPCpMYRtKMpS = 978.4620750887875;     // kpc/Myr/1000.0;
double a0 = 0.0038676205203524152;           // 1.2E-10*MpSStKPCpMYRMYR;

char *MODEL = "W21"; //  name of the model

//  model parameters
double r0 = 0.075;
double rcut = 2.1;
double rho0b = 94.64e9;
double Sigma0thin = 1057.5e6;
double Rdthin = 2.39;
double zdthin = 0.3;
double Sigma0thick = 167.76e6;
double Rdthick = 3.20;
double zdthick = 0.9;
double Sigma0gas1 = 53.1e6;
double Rdgas1 = 7.0;
double zdgas1 = 0.085;
double Rm1 = 4.0;
double Sigma0gas2 = 2179.5e6;
double Rdgas2 = 1.5;
double zdgas2 = 0.045;
double Rm2 = 12.0;
double rh = 11.75;
double rho0h = 1.55e7;
double Gamma = 0.95;
double alpha = 1.19;
double beta = -2.95;

//  baryonic
double Baryonic(double R, double Z)
{
    if (R < 0.4)
        R = 0.3;
    double r = sqrt(R * R + Z * Z);
    double rp = sqrt(R * R + pow(Z / 0.5, 2));
    double rhob = rho0b * exp(-pow(rp / rcut, 2)) * pow(1. + rp / r0, -1.8);
    double rhodthin = 0.5 * Sigma0thin / zdthin * exp(-fabs(Z) / zdthin - R / Rdthin);
    double rhodthick = 0.5 * Sigma0thick / zdthick * exp(-fabs(Z) / zdthick - R / Rdthick);
    double rhodgas1 = 0.25 * Sigma0gas1 / zdgas1 * exp(-Rm1 / R - R / Rdgas1) * pow(cosh(0.5 * Z / zdgas1), -2);
    double rhodgas2 = 0.25 * Sigma0gas2 / zdgas2 * exp(-Rm2 / R - R / Rdgas2) * pow(cosh(0.5 * Z / zdgas2), -2);
    return rhob + rhodthin + rhodthick + rhodgas1 + rhodgas2;
}

//  CDM
double CDM(double R, double Z)
{
    double r = sqrt(R * R + pow(Z / 0.95, 2));
    if (r < 0.4)
        r = 0.3;
    double rhoDM = rho0h * pow(r / rh, -0.95) * pow(1 + pow(r / rh, 1.19), -1.68);
    return rhoDM;
}

// char *MODEL = "M17"; //  name of the model

// // model parameters
// double r0 = 0.075;
// double rcut = 2.1;
// double rho0b = 10.02e10;
// double Sigma0thin = 952e6;
// double Rdthin = 2.40037;
// double zdthin = 0.3;
// double Sigma0thick = 119e6;
// double Rdthick = 3.47151;
// double zdthick = 0.9;
// double Sigma0gas1 = 53.1e6;
// double Rdgas1 = 7.0;
// double zdgas1 = 0.085;
// double Rm1 = 4.0;
// double Sigma0gas2 = 2180e6;
// double Rdgas2 = 1.5;
// double zdgas2 = 0.045;
// double Rm2 = 12.0;
// double rh = 21.2155;
// double rho0h = 0.006982e9;

// // baryonic
// double Baryonic(double R, double Z)
// {
// 	if (R < 0.4)
// 	    R = 0.3;
// 	double r = sqrt(R*R + Z*Z);
// 	double rp = sqrt(R*R + pow(Z/0.5, 2));
// 	double rhob = rho0b*exp(-pow(rp/rcut, 2))*pow(1. + rp/r0, -1.8);
// 	double rhodthin = 0.5*Sigma0thin/zdthin*exp(-fabs(Z)/zdthin - R/Rdthin);
// 	double rhodthick = 0.5*Sigma0thick/zdthick*exp(-fabs(Z)/zdthick - R/Rdthick);
// 	double rhodgas1 = 0.25*Sigma0gas1/zdgas1*exp(-Rm1/R - R/Rdgas1)*pow(cosh(0.5*Z/zdgas1), -2);
// 	double rhodgas2 = 0.25*Sigma0gas2/zdgas2*exp(-Rm2/R - R/Rdgas2)*pow(cosh(0.5*Z/zdgas2), -2);
// 	return rhob + rhodthin + rhodthick + rhodgas1 + rhodgas2;
// }

// // CDM
// double CDM(double R, double Z)
// {
// 	double r = sqrt(R*R + Z*Z);
//     if (r < 0.4)
//        	r = 0.3;
//     r/=rh;
// 	return rho0h/r/pow(1.0 + r, 2);
// }

typedef struct
{
    double x;
    double y;
    double z; // coordinate of x,y,z

    // double r;//distance from center

    double db; // baryonic matter
    double dm; // dark matter
    double dp; // phantom matter
    double dg; // effective DM in MOG

    double pn; // newton potential
    double pc; // CDM potential
    double pp; // phantom potential
    double pg; // MOG potential
} Grid;

#define GAL galaxy[i * amounty * amountz + j * amountz + k]

Grid *Init_Grid(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{
    int i;
    printf("-->Initialization...\t");
    fflush(stdout);
    clock_t start, finish;
    start = clock();
    galaxy = (Grid *)malloc(sizeof(Grid) * amountx * amounty * amountz);
    if (galaxy == NULL)
    {
        printf("out of memory!\n");
        fflush(stdout);
        exit(0);
    }
    for (i = 0; i < amountx; i++)
    {
        int j;
#pragma omp parallel for
        for (j = 0; j < amounty; j++)
        {
            int k;
            for (k = 0; k < amountz; k++)
            {
                GAL.x = (i - amountx / 2) * h_stepx;
                GAL.y = (j - amounty / 2) * h_stepy;
                GAL.z = (k - amountz / 2) * h_stepz;
                // GAL.r = sqrt(GAL.x*GAL.x + GAL.y*GAL.y);
                double R = sqrt(GAL.x * GAL.x + GAL.y * GAL.y);
                GAL.db = Baryonic(R, GAL.z);
                GAL.dm = CDM(R, GAL.z);
                GAL.dp = 0;
                GAL.dg = 0;
                GAL.pn = 0;
                GAL.pp = 0;
                GAL.pc = 0;
                GAL.pg = 0;
            }
        }
    }
    finish = clock();
    printf("Finished! Spent %.2fs (CPU).\n", (double)(finish - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    return galaxy;
} //初始化网格

Grid *evaluate_potential(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{
    int i;
    double VG = G * h_stepx * h_stepy * h_stepz;
    int full_length = amountx * amounty * amountz;

    // MOG
    const double alpha = 8.89;
    const double mu = 0.042;
    const double VGINF = (1 + alpha) * G * h_stepx * h_stepy * h_stepz; // (1 + alpha)*G_N*V
    // MOG
    printf("-->Newton+CDM+MOG...\t");
    fflush(stdout);
    clock_t start, finish;
    start = clock();
    for (i = round(amountx / 2) - 1; i < amountx; i++)
    {
        int j = round(amounty / 2);
        int k;
#pragma omp parallel for
        for (k = round(amountz / 2) - 1; k < amountz; k++)
        {
            int I = i * amounty * amountz + j * amountz + k;
            double tmpN = 0; // Newton
            double tmpC = 0; // CDM
            double tmpG = 0; // MOG
            int J;
            double x = galaxy[I].x;
            double y = galaxy[I].y;
            double z = galaxy[I].z;
            for (J = 0; J < full_length; J++)
            {
                if (I == J)
                {
                    continue;
                }
                double r = sqrt(pow(x - galaxy[J].x, 2) + pow(y - galaxy[J].y, 2) + pow(z - galaxy[J].z, 2));
                tmpC -= (galaxy[J].db + galaxy[J].dm) / r;
                tmpN -= (galaxy[J].db) / r;

                // MOG
                tmpG += -galaxy[J].db / r + alpha / (1 + alpha) * exp(-mu * r) * galaxy[J].db / r;
            }
            galaxy[I].pn = tmpN * VG;
            galaxy[I].pc = tmpC * VG;
            galaxy[I].pg = tmpG * VGINF;
        }
    }
    finish = clock();
    printf("Finished! Spent %.2fs (CPU).\n", (double)(finish - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    return galaxy;
}

Grid *evaluate_potential_QUMOND(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{
    int i;
    double VG = G * h_stepx * h_stepy * h_stepz;
    int full_length = amountx * amounty * amountz;
    printf("-->QUMOND potential...\t");
    fflush(stdout);
    clock_t start, finish;
    start = clock();
    for (i = round(amountx / 2) - 1; i < amountx; i++)
    {
        int j = round(amounty / 2);
        int k;
#pragma omp parallel for
        for (k = round(amountz / 2) - 1; k < amountz; k++)
        {
            int I = i * amounty * amountz + j * amountz + k;
            double tmpM = 0; // QUMOND
            int J;
            double x = galaxy[I].x;
            double y = galaxy[I].y;
            double z = galaxy[I].z;
            for (J = 0; J < full_length; J++)
            {
                if (I == J)
                {
                    continue;
                }
                double r = sqrt(pow(x - galaxy[J].x, 2) + pow(y - galaxy[J].y, 2) + pow(z - galaxy[J].z, 2));
                tmpM -= (galaxy[J].db + galaxy[J].dp) / r;
            }
            galaxy[I].pp = tmpM * VG;
        }
    }
    finish = clock();
    printf("Finished! Spent %.2fs (CPU).\n", (double)(finish - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    return galaxy;
}

Grid *broadcast_potential_pn(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{
    printf("-->broadcasting pn...\t");
    fflush(stdout);
    clock_t start, finish;
    start = clock();

    size_t nx = amountx / 2;
    size_t nz = amountz / 2;
    double *Rlist = calloc(nx, sizeof(double));
    double *zlist = calloc(nz, sizeof(double));
    double *p = calloc(nx * nz, sizeof(double));
    int ii;
    for (ii = 0; ii < nx; ii++)
    {
        Rlist[ii] = ii * h_stepx;
    }
    for (ii = 0; ii < nz; ii++)
    {
        zlist[ii] = ii * h_stepz;
    }

    for (ii = 0; ii < nx; ii++)
    {
        int i = ii + amountx / 2;
        int j = round(amounty / 2);
        int kk;
#pragma omp parallel for
        for (kk = 0; kk < nz; kk++)
        {
            int k = kk + amountz / 2;
            p[kk * nx + ii] = GAL.pn; // ! be careful with the transpose!
        }
    }
    const gsl_interp2d_type *T = gsl_interp2d_bilinear;
    gsl_spline2d *spline = gsl_spline2d_alloc(T, nx, nz);
    gsl_interp_accel *xacc = gsl_interp_accel_alloc();
    gsl_interp_accel *zacc = gsl_interp_accel_alloc();
    gsl_spline2d_init(spline, Rlist, zlist, p, nx, nz);

    int i;
    for (i = 0; i < amountx; i++)
    {
        int j;
#pragma omp parallel for
        for (j = 0; j < amounty; j++)
        {
            int k;
            for (k = 0; k < amountz; k++)
            {
                double R = sqrt(GAL.x * GAL.x + GAL.y * GAL.y);
                GAL.pn = gsl_spline2d_eval_extrap(spline, R, fabs(GAL.z), xacc, zacc);
            }
        }
    }

    finish = clock();
    printf("Finished! Spent %.2fs (CPU).\n", (double)(finish - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    free(p);
    free(Rlist);
    free(zlist);
    free(spline);
    return galaxy;
}

Grid *broadcast_potential_pc(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{
    printf("-->broadcasting pc...\t");
    fflush(stdout);
    clock_t start, finish;
    start = clock();

    size_t nx = amountx / 2;
    size_t nz = amountz / 2;
    double *Rlist = calloc(nx, sizeof(double));
    double *zlist = calloc(nz, sizeof(double));
    double *p = calloc(nx * nz, sizeof(double));
    int ii;
    for (ii = 0; ii < nx; ii++)
    {
        Rlist[ii] = ii * h_stepx;
    }
    for (ii = 0; ii < nz; ii++)
    {
        zlist[ii] = ii * h_stepz;
    }

    for (ii = 0; ii < nx; ii++)
    {
        int i = ii + amountx / 2;
        int j = round(amounty / 2);
        int kk;
#pragma omp parallel for
        for (kk = 0; kk < nz; kk++)
        {
            int k = kk + amountz / 2;
            p[kk * nx + ii] = GAL.pc;
        }
    }
    const gsl_interp2d_type *T = gsl_interp2d_bilinear;
    gsl_spline2d *spline = gsl_spline2d_alloc(T, nx, nz);
    gsl_interp_accel *xacc = gsl_interp_accel_alloc();
    gsl_interp_accel *zacc = gsl_interp_accel_alloc();
    gsl_spline2d_init(spline, Rlist, zlist, p, nx, nz);

    int i;
    for (i = 0; i < amountx; i++)
    {
        int j;
#pragma omp parallel for
        for (j = 0; j < amounty; j++)
        {
            int k;
            for (k = 0; k < amountz; k++)
            {
                double R = sqrt(GAL.x * GAL.x + GAL.y * GAL.y);
                GAL.pc = gsl_spline2d_eval_extrap(spline, R, fabs(GAL.z), xacc, zacc);
            }
        }
    }

    finish = clock();
    printf("Finished! Spent %.2fs (CPU).\n", (double)(finish - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    free(p);
    free(Rlist);
    free(zlist);
    free(spline);
    return galaxy;
}

Grid *broadcast_potential_pp(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{
    printf("-->broadcasting pp...\t");
    fflush(stdout);
    clock_t start, finish;
    start = clock();

    size_t nx = amountx / 2;
    size_t nz = amountz / 2;
    double *Rlist = calloc(nx, sizeof(double));
    double *zlist = calloc(nz, sizeof(double));
    double *p = calloc(nx * nz, sizeof(double));
    int ii;
    for (ii = 0; ii < nx; ii++)
    {
        Rlist[ii] = ii * h_stepx;
    }
    for (ii = 0; ii < nz; ii++)
    {
        zlist[ii] = ii * h_stepz;
    }

    for (ii = 0; ii < nx; ii++)
    {
        int i = ii + amountx / 2;
        int j = round(amounty / 2);
        int kk;
#pragma omp parallel for
        for (kk = 0; kk < nz; kk++)
        {
            int k = kk + amountz / 2;
            p[kk * nx + ii] = GAL.pp;
        }
    }
    const gsl_interp2d_type *T = gsl_interp2d_bilinear;
    gsl_spline2d *spline = gsl_spline2d_alloc(T, nx, nz);
    gsl_interp_accel *xacc = gsl_interp_accel_alloc();
    gsl_interp_accel *zacc = gsl_interp_accel_alloc();
    gsl_spline2d_init(spline, Rlist, zlist, p, nx, nz);

    int i;
    for (i = 0; i < amountx; i++)
    {
        int j;
#pragma omp parallel for
        for (j = 0; j < amounty; j++)
        {
            int k;
            for (k = 0; k < amountz; k++)
            {
                double R = sqrt(GAL.x * GAL.x + GAL.y * GAL.y);
                GAL.pp = gsl_spline2d_eval_extrap(spline, R, fabs(GAL.z), xacc, zacc);
            }
        }
    }

    finish = clock();
    printf("Finished! Spent %.2fs (CPU).\n", (double)(finish - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    free(p);
    free(Rlist);
    free(zlist);
    free(spline);
    return galaxy;
}

Grid *broadcast_potential_pg(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{
    printf("-->broadcasting pg...\t");
    fflush(stdout);
    clock_t start, finish;
    start = clock();

    size_t nx = amountx / 2;
    size_t nz = amountz / 2;
    double *Rlist = calloc(nx, sizeof(double));
    double *zlist = calloc(nz, sizeof(double));
    double *p = calloc(nx * nz, sizeof(double));
    int ii;
    for (ii = 0; ii < nx; ii++)
    {
        Rlist[ii] = ii * h_stepx;
    }
    for (ii = 0; ii < nz; ii++)
    {
        zlist[ii] = ii * h_stepz;
    }

    for (ii = 0; ii < nx; ii++)
    {
        int i = ii + amountx / 2;
        int j = round(amounty / 2);
        int kk;
#pragma omp parallel for
        for (kk = 0; kk < nz; kk++)
        {
            int k = kk + amountz / 2;
            p[kk * nx + ii] = GAL.pg;
        }
    }
    const gsl_interp2d_type *T = gsl_interp2d_bilinear;
    gsl_spline2d *spline = gsl_spline2d_alloc(T, nx, nz);
    gsl_interp_accel *xacc = gsl_interp_accel_alloc();
    gsl_interp_accel *zacc = gsl_interp_accel_alloc();
    gsl_spline2d_init(spline, Rlist, zlist, p, nx, nz);

    int i;
    for (i = 0; i < amountx; i++)
    {
        int j;
#pragma omp parallel for
        for (j = 0; j < amounty; j++)
        {
            int k;
            for (k = 0; k < amountz; k++)
            {
                double R = sqrt(GAL.x * GAL.x + GAL.y * GAL.y);
                GAL.pg = gsl_spline2d_eval_extrap(spline, R, fabs(GAL.z), xacc, zacc);
            }
        }
    }

    finish = clock();
    printf("Finished! Spent %.2fs (CPU).\n", (double)(finish - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    free(p);
    free(Rlist);
    free(zlist);
    free(spline);
    return galaxy;
}

double v(double y)
{
    if (y < 1e-6)
    {
        return 0.5 * sqrt(1.0 + 4.0 / 1e-6) + 0.5;
    }
    return 0.5 * sqrt(1.0 + 4.0 / y) + 0.5;
}

double v_(double y)
{
    return v(y) - 1.0;
} // v-function;

// calculate PDM distribution
int pdm_dis(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{

    double vax, vay, vaz, vbx, vby, vbz, axx, axy, axz, ayx, ayy, ayz, azx, azy, azz;
    double bxx, bxy, bxz, byx, byy, byz, bzx, bzy, bzz, pdm;
    clock_t start, finish;
    int i, j, k;
    start = clock();
    printf("-->Calculating PDM...\t");
    fflush(stdout);
    for (i = 2; i < amountx - 2; i++)
    {
#pragma omp parallel for
        for (j = 2; j < amounty - 2; j++)
        {
            for (k = 2; k < amountz - 2; k++)
            {
                axx = (galaxy[(i - 2) * amounty * amountz + j * amountz + k].pn - 27 * galaxy[(i - 1) * amounty * amountz + j * amountz + k].pn + 27 * galaxy[i * amounty * amountz + j * amountz + k].pn - galaxy[(i + 1) * amounty * amountz + j * amountz + k].pn) / (12 * h_stepx + 12 * h_stepx);
                axy = (galaxy[(i - 1) * amounty * amountz + (j - 2) * amountz + k].pn - 8 * galaxy[(i - 1) * amounty * amountz + (j - 1) * amountz + k].pn + 8 * galaxy[(i - 1) * amounty * amountz + (j + 1) * amountz + k].pn - galaxy[(i - 1) * amounty * amountz + (j + 2) * amountz + k].pn + galaxy[i * amounty * amountz + (j - 2) * amountz + k].pn - 8 * galaxy[i * amounty * amountz + (j - 1) * amountz + k].pn + 8 * galaxy[i * amounty * amountz + (j + 1) * amountz + k].pn - galaxy[i * amounty * amountz + (j + 2) * amountz + k].pn) / (12 * h_stepx + 12 * h_stepy);
                axz = (galaxy[(i - 1) * amounty * amountz + j * amountz + (k - 2)].pn - 8 * galaxy[(i - 1) * amounty * amountz + j * amountz + (k - 1)].pn + 8 * galaxy[(i - 1) * amounty * amountz + j * amountz + (k + 1)].pn - galaxy[(i - 1) * amounty * amountz + j * amountz + (k + 2)].pn + galaxy[i * amounty * amountz + j * amountz + (k - 2)].pn - 8 * galaxy[i * amounty * amountz + j * amountz + (k - 1)].pn + 8 * galaxy[i * amounty * amountz + j * amountz + (k + 1)].pn - galaxy[i * amounty * amountz + j * amountz + (k + 2)].pn) / (12 * h_stepx + 12 * h_stepz);
                ayx = (galaxy[(i - 2) * amounty * amountz + (j - 1) * amountz + k].pn - 8 * galaxy[(i - 1) * amounty * amountz + (j - 1) * amountz + k].pn + 8 * galaxy[(i + 1) * amounty * amountz + (j - 1) * amountz + k].pn - galaxy[(i + 2) * amounty * amountz + (j - 1) * amountz + k].pn + galaxy[(i - 2) * amounty * amountz + j * amountz + k].pn - 8 * galaxy[(i - 1) * amounty * amountz + j * amountz + k].pn + 8 * galaxy[(i + 1) * amounty * amountz + j * amountz + k].pn - galaxy[(i + 2) * amounty * amountz + j * amountz + k].pn) / (12 * h_stepy + 12 * h_stepx);
                ayy = (galaxy[i * amounty * amountz + (j - 2) * amountz + k].pn - 27 * galaxy[i * amounty * amountz + (j - 1) * amountz + k].pn + 27 * galaxy[i * amounty * amountz + j * amountz + k].pn - galaxy[i * amounty * amountz + (j + 1) * amountz + k].pn) / (12 * h_stepy + 12 * h_stepy);
                ayz = (galaxy[i * amounty * amountz + (j - 1) * amountz + (k - 2)].pn - 8 * galaxy[i * amounty * amountz + (j - 1) * amountz + (k - 1)].pn + 8 * galaxy[i * amounty * amountz + (j - 1) * amountz + (k + 1)].pn - galaxy[i * amounty * amountz + (j - 1) * amountz + (k + 2)].pn + galaxy[i * amounty * amountz + j * amountz + (k - 2)].pn - 8 * galaxy[i * amounty * amountz + j * amountz + (k - 1)].pn + 8 * galaxy[i * amounty * amountz + j * amountz + (k + 1)].pn - galaxy[i * amounty * amountz + j * amountz + (k + 2)].pn) / (12 * h_stepy + 12 * h_stepz);
                azx = (galaxy[(i - 2) * amounty * amountz + j * amountz + (k - 1)].pn - 8 * galaxy[(i - 1) * amounty * amountz + j * amountz + (k - 1)].pn + 8 * galaxy[(i + 1) * amounty * amountz + j * amountz + (k - 1)].pn - galaxy[(i + 2) * amounty * amountz + j * amountz + (k - 1)].pn + galaxy[(i - 2) * amounty * amountz + j * amountz + k].pn - 8 * galaxy[(i - 1) * amounty * amountz + j * amountz + k].pn + 8 * galaxy[(i + 1) * amounty * amountz + j * amountz + k].pn - galaxy[(i + 2) * amounty * amountz + j * amountz + k].pn) / (12 * h_stepx + 12 * h_stepz);
                azy = (galaxy[i * amounty * amountz + (j - 2) * amountz + (k - 1)].pn - 8 * galaxy[i * amounty * amountz + (j - 1) * amountz + (k - 1)].pn + 8 * galaxy[i * amounty * amountz + (j + 1) * amountz + (k - 1)].pn - galaxy[i * amounty * amountz + (j + 2) * amountz + (k - 1)].pn + galaxy[i * amounty * amountz + (j - 2) * amountz + k].pn - 8 * galaxy[i * amounty * amountz + (j - 1) * amountz + k].pn + 8 * galaxy[i * amounty * amountz + (j + 1) * amountz + k].pn - galaxy[i * amounty * amountz + (j + 2) * amountz + k].pn) / (12 * h_stepy + 12 * h_stepz);
                azz = (galaxy[i * amounty * amountz + j * amountz + (k - 2)].pn - 27 * galaxy[i * amounty * amountz + j * amountz + (k - 1)].pn + 27 * galaxy[i * amounty * amountz + j * amountz + k].pn - galaxy[i * amounty * amountz + j * amountz + (k + 1)].pn) / (12 * h_stepz + 12 * h_stepz);

                bxx = -(galaxy[(i + 2) * amounty * amountz + j * amountz + k].pn - 27 * galaxy[(i + 1) * amounty * amountz + j * amountz + k].pn + 27 * galaxy[i * amounty * amountz + j * amountz + k].pn - galaxy[(i - 1) * amounty * amountz + j * amountz + k].pn) / (12 * h_stepx + 12 * h_stepx);
                bxy = (galaxy[(i + 1) * amounty * amountz + (j - 2) * amountz + k].pn - 8 * galaxy[(i + 1) * amounty * amountz + (j - 1) * amountz + k].pn + 8 * galaxy[(i + 1) * amounty * amountz + (j + 1) * amountz + k].pn - galaxy[(i + 1) * amounty * amountz + (j + 2) * amountz + k].pn + galaxy[i * amounty * amountz + (j - 2) * amountz + k].pn - 8 * galaxy[i * amounty * amountz + (j - 1) * amountz + k].pn + 8 * galaxy[i * amounty * amountz + (j + 1) * amountz + k].pn - galaxy[i * amounty * amountz + (j + 2) * amountz + k].pn) / (12 * h_stepx + 12 * h_stepy);
                bxz = (galaxy[(i + 1) * amounty * amountz + j * amountz + (k - 2)].pn - 8 * galaxy[(i + 1) * amounty * amountz + j * amountz + (k - 1)].pn + 8 * galaxy[(i + 1) * amounty * amountz + j * amountz + (k + 1)].pn - galaxy[(i + 1) * amounty * amountz + j * amountz + (k + 2)].pn + galaxy[i * amounty * amountz + j * amountz + (k - 2)].pn - 8 * galaxy[i * amounty * amountz + j * amountz + (k - 1)].pn + 8 * galaxy[i * amounty * amountz + j * amountz + (k + 1)].pn - galaxy[i * amounty * amountz + j * amountz + (k + 2)].pn) / (12 * h_stepx + 12 * h_stepz);
                byx = (galaxy[(i - 2) * amounty * amountz + (j + 1) * amountz + k].pn - 8 * galaxy[(i - 1) * amounty * amountz + (j + 1) * amountz + k].pn + 8 * galaxy[(i + 1) * amounty * amountz + (j + 1) * amountz + k].pn - galaxy[(i + 2) * amounty * amountz + (j + 1) * amountz + k].pn + galaxy[(i - 2) * amounty * amountz + j * amountz + k].pn - 8 * galaxy[(i - 1) * amounty * amountz + j * amountz + k].pn + 8 * galaxy[(i + 1) * amounty * amountz + j * amountz + k].pn - galaxy[(i + 2) * amounty * amountz + j * amountz + k].pn) / (12 * h_stepy + 12 * h_stepx);
                byy = -(galaxy[i * amounty * amountz + (j + 2) * amountz + k].pn - 27 * galaxy[i * amounty * amountz + (j + 1) * amountz + k].pn + 27 * galaxy[i * amounty * amountz + j * amountz + k].pn - galaxy[i * amounty * amountz + (j - 1) * amountz + k].pn) / (12 * h_stepy + 12 * h_stepy);
                byz = (galaxy[i * amounty * amountz + (j + 1) * amountz + (k - 2)].pn - 8 * galaxy[i * amounty * amountz + (j + 1) * amountz + (k - 1)].pn + 8 * galaxy[i * amounty * amountz + (j + 1) * amountz + (k + 1)].pn - galaxy[i * amounty * amountz + (j + 1) * amountz + (k + 2)].pn + galaxy[i * amounty * amountz + j * amountz + (k - 2)].pn - 8 * galaxy[i * amounty * amountz + j * amountz + (k - 1)].pn + 8 * galaxy[i * amounty * amountz + j * amountz + (k + 1)].pn - galaxy[i * amounty * amountz + j * amountz + (k + 2)].pn) / (12 * h_stepy + 12 * h_stepz);
                bzx = (galaxy[(i - 2) * amounty * amountz + j * amountz + (k + 1)].pn - 8 * galaxy[(i - 1) * amounty * amountz + j * amountz + (k + 1)].pn + 8 * galaxy[(i + 1) * amounty * amountz + j * amountz + (k + 1)].pn - galaxy[(i + 2) * amounty * amountz + j * amountz + (k + 1)].pn + galaxy[(i - 2) * amounty * amountz + j * amountz + k].pn - 8 * galaxy[(i - 1) * amounty * amountz + j * amountz + k].pn + 8 * galaxy[(i + 1) * amounty * amountz + j * amountz + k].pn - galaxy[(i + 2) * amounty * amountz + j * amountz + k].pn) / (12 * h_stepz + 12 * h_stepx);
                bzy = (galaxy[i * amounty * amountz + (j - 2) * amountz + (k + 1)].pn - 8 * galaxy[i * amounty * amountz + (j - 1) * amountz + (k + 1)].pn + 8 * galaxy[i * amounty * amountz + (j + 1) * amountz + (k + 1)].pn - galaxy[i * amounty * amountz + (j + 2) * amountz + (k + 1)].pn + galaxy[i * amounty * amountz + (j - 2) * amountz + k].pn - 8 * galaxy[i * amounty * amountz + (j - 1) * amountz + k].pn + 8 * galaxy[i * amounty * amountz + (j + 1) * amountz + k].pn - galaxy[i * amounty * amountz + (j + 2) * amountz + k].pn) / (12 * h_stepz + 12 * h_stepy);
                bzz = -(galaxy[i * amounty * amountz + j * amountz + (k + 2)].pn - 27 * galaxy[i * amounty * amountz + j * amountz + (k + 1)].pn + 27 * galaxy[i * amounty * amountz + j * amountz + k].pn - galaxy[i * amounty * amountz + j * amountz + (k - 1)].pn) / (12 * h_stepz + 12 * h_stepz);

                vax = v_(sqrt(axx * axx + axy * axy + axz * axz) / a0);
                vay = v_(sqrt(ayx * ayx + ayy * ayy + ayz * ayz) / a0);
                vaz = v_(sqrt(azx * azx + azy * azy + azz * azz) / a0);
                vbx = v_(sqrt(bxx * bxx + bxy * bxy + bxz * bxz) / a0);
                vby = v_(sqrt(byx * byx + byy * byy + byz * byz) / a0);
                vbz = v_(sqrt(bzx * bzx + bzy * bzy + bzz * bzz) / a0);
                pdm = (vbx * bxx - vax * axx + vby * byy - vay * ayy + vbz * bzz - vaz * azz) / (4.0 * M_PI * 0.333333333333 * (h_stepx + h_stepy + h_stepz) * G);
                GAL.dp = pdm;
            }
        }
    }
    finish = clock();
    printf("Finished! Spent %.2fs (CPU).\n", (double)(finish - start) / CLOCKS_PER_SEC);
    return 0;
}

// calculate effective dark matter in MOG
int pg_dis(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{
    clock_t start, finish;
    int i, j, k;
    start = clock();
    printf("-->Calculating effective DM in MOG...\t");
    fflush(stdout);
    for (i = 2; i < amountx - 2; i++)
    {
#pragma omp parallel for
        for (j = 2; j < amounty - 2; j++)
        {
            for (k = 2; k < amountz - 2; k++)
            {
                double dg = 0;
                dg += (galaxy[(i - 1) * amounty * amountz + j * amountz + k].pg - 2 * galaxy[i * amounty * amountz + j * amountz + k].pg + galaxy[(i + 1) * amounty * amountz + j * amountz + k].pg) / h_stepx / h_stepx;
                dg += (galaxy[i * amounty * amountz + (j - 1) * amountz + k].pg - 2 * galaxy[i * amounty * amountz + j * amountz + k].pg + galaxy[i * amounty * amountz + (j + 1) * amountz + k].pg) / h_stepy / h_stepy;
                dg += (galaxy[i * amounty * amountz + j * amountz + (k - 1)].pg - 2 * galaxy[i * amounty * amountz + j * amountz + k].pg + galaxy[i * amounty * amountz + j * amountz + (k + 1)].pg) / h_stepz / h_stepz;
                dg /= 4.0 * M_PI * G;
                GAL.dg = dg - GAL.db;
            }
        }
    }
    finish = clock();
    printf("Finished! Spent %.2fs (CPU).\n", (double)(finish - start) / CLOCKS_PER_SEC);
    return 0;
}

int write_PDM(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{
    int i, j, k;
    char filename[64] = "dspdm-";
    strcat(filename, MODEL);
    strcat(filename, ".txt");
    FILE *ptr = fopen(filename, "w");
    fprintf(ptr, "#x[kpc]\tz[kpc]\tPDM[GeV/cm3]\tCDM[GeV/cm3]\tMOG[GeV/cm3]\n");
    for (i = 0; i < amountx; i++)
    {
        // for (j = round(amounty/2)-3; j < round(amounty/2)+4; j++)
        j = round(amounty / 2);
        for (k = 0; k < amountz; k++)
        {
            double x = GAL.x;
            double z = GAL.z;
            double PDM = GAL.dp * toGeV;
            double CDM = GAL.dm * toGeV;
            double GDM = GAL.dg * toGeV;
            fprintf(ptr, "%f\t%f\t%g\t%g\t%g\n", x, z, PDM, CDM, GDM);
        }
    }
    fclose(ptr);
    printf("    -File wrote to %s.\n", filename);
    return 0;
}

int write_potential(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{
    int i, j, k;
    char filename[64] = "dspn-";
    strcat(filename, MODEL);
    strcat(filename, ".txt");
    FILE *ptr = fopen(filename, "w");
    fprintf(ptr, "#x[kpc]\tz[kpc]\tPN\tPCDM\tPQUMOND\tPMOG\n");
    for (i = 0; i < amountx; i++)
    {
        // for (j = round(amounty/2)-3; j < round(amounty/2)+4; j++)
        j = round(amounty / 2);
        for (k = 0; k < amountz; k++)
        {
            double x = GAL.x;
            double z = GAL.z;
            double PN = GAL.pn;
            double PCDM = GAL.pc;
            double PQUMOND = GAL.pp;
            double PMOG = GAL.pg;
            fprintf(ptr, "%f\t%f\t%g\t%g\t%g\t%g\n", x, z, PN, PCDM, PQUMOND, PMOG);
        }
    }
    fclose(ptr);
    printf("    -File wrote to %s.\n", filename);
    return 0;
}


// write matrix for Jeans Analysis
int write_JeansMat(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{
    int i, j, k;
    char filename[64] = "dsmatrix-GAIA-";
    strcat(filename, MODEL);
    strcat(filename, ".ds");
    FILE *ptr = fopen(filename, "wb");
    int zbound = (int)(2.5/h_stepz + amountz/2);
    int N = (amountx-1 - amountx / 2)*(zbound - amountz / 2);
    double *Rlist = malloc(sizeof(double)*N);
    double *zlist = malloc(sizeof(double)*N);
    double *QumondPhi = malloc(sizeof(double)*N);
    double *ColddmPhi = malloc(sizeof(double)*N);
    double *NewtonPhi = malloc(sizeof(double)*N);
    double *MoffatPhi = malloc(sizeof(double)*N);
    double *rho = malloc(sizeof(double)*N);
    assert(Rlist!=NULL);
    assert(rho!=NULL);
    int count = 0;
    for (i = amountx / 2; i < amountx-1; i++)
    {
        // for (j = round(amounty/2)-3; j < round(amounty/2)+4; j++)
        j = round(amounty / 2);
        for (k = amountz / 2; k < zbound; k++)
        {
            double R = sqrt(pow(GAL.x, 2) + pow(GAL.y, 2));
            double z = GAL.z;
            if (z>2.5)
            {
                break;
            }
            Rlist[count] = R;
            zlist[count] = z;
            QumondPhi[count]  = GAL.pp;
            ColddmPhi[count]  = GAL.pc;
            NewtonPhi[count]  = GAL.pn;
            MoffatPhi[count]  = GAL.pg;
            rho[count]  = GAL.db;
            count++;
        }
    }
    assert(count == N);
    fwrite(&N, sizeof(N), 1, ptr);
    for (count=0; count<N; count++) fwrite(&(Rlist[count]), sizeof(double), 1, ptr);
    for (count=0; count<N; count++) fwrite(&(zlist[count]), sizeof(double), 1, ptr);
    for (count=0; count<N; count++) fwrite(&(QumondPhi[count]), sizeof(double), 1, ptr);
    for (count=0; count<N; count++) fwrite(&(ColddmPhi[count]), sizeof(double), 1, ptr);
    for (count=0; count<N; count++) fwrite(&(NewtonPhi[count]), sizeof(double), 1, ptr);
    for (count=0; count<N; count++) fwrite(&(MoffatPhi[count]), sizeof(double), 1, ptr);
    for (count=0; count<N; count++) fwrite(&(rho[count]), sizeof(double), 1, ptr);
    fclose(ptr);
    printf("    -File wrote to %s.\n", filename);
    return 0;
}

int write_RC(Grid *galaxy, int amountx, int amounty, int amountz, double h_stepx, double h_stepy, double h_stepz)
{
    int i, j, k;
    char filename[64] = "dsvel-";
    strcat(filename, MODEL);
    strcat(filename, ".txt");
    FILE *ptr = fopen(filename, "w");
    fprintf(ptr, "#R[kpc]\tBaryon[km/s]\tCDM[km/s]\tQUMOND[km/s]\tMOG[km/s]\n");
    for (i = round(amountx / 2); i < amountx - 1; i++)
    {
        // for (j = round(amounty/2)-3; j < round(amounty/2)+4; j++)
        j = round(amounty / 2);
        k = round(amountz / 2);
        double R = sqrt(pow(GAL.x, 2) + pow(GAL.y, 2));
        double z = GAL.z;
        double dphi_dr = fabs(galaxy[(i + 1) * amounty * amountz + j * amountz + k].pn - galaxy[i * amounty * amountz + j * amountz + k].pn) / h_stepx;
        double Nvc = sqrt(R * dphi_dr) * KPCpMYRtKMpS;

        dphi_dr = fabs(galaxy[(i + 1) * amounty * amountz + j * amountz + k].pc - galaxy[i * amounty * amountz + j * amountz + k].pc) / h_stepx;
        double Cvc = sqrt(R * dphi_dr) * KPCpMYRtKMpS;

        dphi_dr = fabs(galaxy[(i + 1) * amounty * amountz + j * amountz + k].pp - galaxy[i * amounty * amountz + j * amountz + k].pp) / h_stepx;
        double Pvc = sqrt(R * dphi_dr) * KPCpMYRtKMpS;

        dphi_dr = fabs(galaxy[(i + 1) * amounty * amountz + j * amountz + k].pg - galaxy[i * amounty * amountz + j * amountz + k].pg) / h_stepx;
        double Gvc = sqrt(R * dphi_dr) * KPCpMYRtKMpS;
        fprintf(ptr, "%f\t%f\t%f\t%f\t%f\n", R, Nvc, Cvc, Pvc, Gvc);
    }
    fclose(ptr);
    printf("    -File wrote to %s.\n", filename);
    return 0;
}

int main()
{
    time_t begin = time(NULL);
    printf("\t--------------------------\n");
    printf("\t  PHANTOM v2.2 Directsum  \n");
    printf("\t--------------------------\n");
    printf("-->Model: %s\n", MODEL);

    // initialize the gird
    Grid *galaxy = NULL;
    int Nx = 256;
    int Ny = 256;
    int Nz = 256;
    double h = 0.3;
    galaxy = Init_Grid(galaxy, Nx, Ny, Nz, h, h, h);

    // calculate Newtonian, CDM, MOG potential
    evaluate_potential(galaxy, Nx, Ny, Nz, h, h, h);
    // broadcast Newtonian potential
    broadcast_potential_pn(galaxy, Nx, Ny, Nz, h, h, h);
    // calculate PDM distribution
    pdm_dis(galaxy, Nx, Ny, Nz, h, h, h);
    // calculate QUMOND potential
    evaluate_potential_QUMOND(galaxy, Nx, Ny, Nz, h, h, h);
    // calculate RC
    write_RC(galaxy, Nx, Ny, Nz, h, h, h);
    // write Jeans Matrix
    write_JeansMat(galaxy, Nx, Ny, Nz, h, h, h);
    // write potential
    broadcast_potential_pc(galaxy, Nx, Ny, Nz, h, h, h);
    broadcast_potential_pp(galaxy, Nx, Ny, Nz, h, h, h);
    broadcast_potential_pg(galaxy, Nx, Ny, Nz, h, h, h);
    // calculate effective DM distribution
    pg_dis(galaxy, Nx, Ny, Nz, h, h, h);
    // write PDM
    write_PDM(galaxy, Nx, Ny, Nz, h, h, h);
    write_potential(galaxy, Nx, Ny, Nz, h, h, h);
    // end
    free(galaxy);
    time_t end = time(NULL);
    printf("* Total elapsed time is %d seconds (Wall).\n", (int)(end - begin));
    return 0;
}
