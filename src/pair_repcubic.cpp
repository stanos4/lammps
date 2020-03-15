/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 Contributing author: Aidan Thompson (SNL)
 ------------------------------------------------------------------------- */

#include "pair_repcubic.h"
#include <mpi.h>
#include <cmath>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace PairRepCubicConstants;

/* ---------------------------------------------------------------------- */

PairRepCubic::PairRepCubic(LAMMPS *lmp) :
    Pair(lmp) {
}

/* ---------------------------------------------------------------------- */

PairRepCubic::~PairRepCubic() {
//  printf("RepCubic destructor\n");
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

//    memory->destroy(cut);
//    memory->destroy(cut_inner);
//    memory->destroy(cut_inner_sq);
//    memory->destroy(epsilon);
//    memory->destroy(sigma);
//    memory->destroy(lj1);
//    memory->destroy(lj2);
//    memory->destroy(lj3);
//    memory->destroy(lj4);
    memory->destroy(cut);
    memory->destroy(gamma_fc);
  }
//  printf("RepCubic destructor\n");
}

/* ---------------------------------------------------------------------- */

void PairRepCubic::compute(int eflag, int vflag) {
//  printf("starting RepCubic compute\n");
  int i, j, j_gamma, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, r;
  double dr, tk, epair, fpair;
  double rsq, factor_ij, cut_ij;
  double fx, fy, fz;
  int *ilist, *jlist, *numneigh, **firstneigh;

  dr = 0.0;
  tk = 0.0;
  epair = 0.0;
  fpair = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

//  printf("    inum: %d\n",inum);
//  printf("gamma[0;0]: %f; %f; %f; %f; %f \n",gamma_fc[0][0][0],gamma_fc[0][0][1],gamma_fc[0][0][2],gamma_fc[0][0][3], cut[0][0]);
//  printf("gamma[1;0]: %f; %f; %f; %f; %f \n",gamma_fc[1][0][0],gamma_fc[1][0][1],gamma_fc[1][0][2],gamma_fc[1][0][3], cut[1][0]);
//  printf("gamma[0;1]: %f; %f; %f; %f; %f \n",gamma_fc[0][1][0],gamma_fc[0][1][1],gamma_fc[0][1][2],gamma_fc[0][1][3], cut[0][1]);
//
//  printf("gamma[1;1]: %f; %f; %f; %f; %f \n",gamma_fc[1][1][0],gamma_fc[1][1][1],gamma_fc[1][1][2],gamma_fc[1][1][3], cut[1][1]);
//
//  printf("gamma[2;1]: %f; %f; %f; %f; %f \n",gamma_fc[2][1][0],gamma_fc[2][1][1],gamma_fc[2][1][2],gamma_fc[2][1][3], cut[2][1]);
//  printf("gamma[1;2]: %f; %f; %f; %f; %f \n",gamma_fc[1][2][0],gamma_fc[1][2][1],gamma_fc[1][2][2],gamma_fc[1][2][3], cut[1][2]);
//
//  printf("gamma[2;2]: %f; %f; %f; %f; %f \n",gamma_fc[2][2][0],gamma_fc[2][2][1],gamma_fc[2][2][2],gamma_fc[2][2][3], cut[2][2]);

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

//    printf("    jnum: %d\n",jnum);
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j_gamma = j;
      factor_ij = special_lj[sbmask(j)];
//      printf("\n\nfactor index: %d; factor_lj: %f\n",sbmask(j),factor_ij);
//      printf("RepCubic %d-%d: %f\n", i,j,factor_ij);
      j &= NEIGHMASK;

      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx * delx + dely * dely + delz * delz;
//      printf("      rsq: %f\n",rsq);
      jtype = type[j];
//      printf("      itype-jtype: %d-%d\n",itype,jtype);
//      debug print stano
//      printf("RepCubic compute pair %d-%d\n",i,j);
//      printf("%f %f %f\n", x[j][0], x[j][1], x[j][2]);
//      printf("    %f\n",factor_lj);


      if (factor_ij != 0.0){
        cut_ij = cutsq[itype][jtype] * factor_ij * factor_ij;
        if (rsq < cut_ij) {
          r = sqrt(rsq);
//          printf("      r: %f\n",r);
          cut_ij = sqrt(cut_ij);
          dr = cut_ij - r;
//          printf("      neightype; cutij; dr; gamma: %d; %f; %f; %f\n",sbmask(j_gamma),cut_ij,dr, gamma_fc[itype][jtype][sbmask(j_gamma)]);
          tk = gamma_fc[itype][jtype][sbmask(j_gamma)] * dr * dr;
          printf("i-j; dr; tk; tk*dr: %d-%d; %f; %f; %f\n",i,j,dr,tk, tk*dr);
//          printf("      itype-jtype; r; dr; gamma; tk*dr: %d-%d; %f; %f; %f; %f\n", itype, jtype, r, dr, gamma_fc[itype][jtype][sbmask(j_gamma)], tk*dr);
//          printf("itype-jtype: cutoff, gamma = %d-%d:%f, %f\n",itype,jtype, cut_ij, gamma_fc[itype][jtype][sbmask(j_gamma)]);

          fpair = 3*tk*factor_ij/r;
        }

      }
      double fx = fpair*delx;
      double fy = fpair*dely;
      double fz = fpair*delz;
      f[i][0] -= fx;
      f[i][1] -= fy;
      f[i][2] -= fz;
      if (newton_pair || j < nlocal){
        f[j][0] += fx;
        f[j][1] += fy;
        f[j][2] += fz;
//        printf("      fix; fiy; fiz; fjx; fjy; fjz: %f; %f; %f;    %f; %f; %f\n",-fx,-fy,-fz,+fx,+fy,+fz);
      }

      if (eflag) {
        epair = tk * dr;
        if (evflag) {
          ev_tally(i,j,nlocal,newton_pair,epair,0.0,fpair,delx,dely,delz);
//          printf("      newton_p; epair; fpair; delx; dely; delz: %d; %f; %f; %f; %f; %f\n",newton_pair, epair, fpair, delx, dely, delz);
        }
      }
//      debug print stano
//      printf("RepCubic compute pair %d-%d\n",i,j);
//      printf("    %f\n",factor_lj);

    }
  }

  if (vflag_fdotr)
    virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairRepCubic::allocate() {
//  printf("RepCubic allocate\n");
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(gamma_fc, n + 1, n + 1, 4, "pair:cut");
//  printf("RepCubic allocate\n");
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairRepCubic::settings(int narg, char**/*arg*/) {
  printf("RepCubic settings\n");
  if (narg != 0)
    error->all(FLERR, "Illegal pair_style command");

  // NOTE: lj/cubic has no global cutoff. instead the cutoff is
  // inferred from the lj parameters. so we must not reset cutoffs here.
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairRepCubic::coeff(int narg, char **arg) {
//  printf("I got to RepCubic: %d\n\n",narg);
  printf("RepCubic coeff\n");
  if (narg != 6)
    error->all(FLERR, "Incorrect args for pair coefficients RepCubic start");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(FLERR, arg[0], atom->ntypes, ilo, ihi);
  force->bounds(FLERR, arg[1], atom->ntypes, jlo, jhi);

//  double epsilon_one = force->numeric(FLERR, arg[2]);
//  double sigma_one = force->numeric(FLERR, arg[3]);
//  double rmin = sigma_one * RT6TWO;

  double cut_one_12 = force->numeric(FLERR, arg[2]);
//  double cut_one_13 = force->numeric(FLERR, arg[3]);
//  double cut_one_14 = force->numeric(FLERR, arg[4]);
  double gamma_fc_one_12 = force->numeric(FLERR, arg[3]);
  double gamma_fc_one_13 = force->numeric(FLERR, arg[4]);
  double gamma_fc_one_14 = force->numeric(FLERR, arg[5]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {

      cut[i][j] = cut[j][i] = cut_one_12;
      gamma_fc[i][j][1] = gamma_fc[j][i][1] = gamma_fc_one_12;
      gamma_fc[i][j][2] = gamma_fc[j][i][2] = gamma_fc_one_13;
      gamma_fc[i][j][3] = gamma_fc[j][i][3] = gamma_fc_one_14;
      gamma_fc[i][j][0] = gamma_fc[j][i][0] = gamma_fc_one_14;
      printf("cut_one, i-j: %d-%d: %f\n",i,j,cut_one_12);
      printf("gamma, i-j: %d-%d: %f  %f  %f\n",i, j, gamma_fc_one_12, gamma_fc_one_13, gamma_fc_one_14);
      setflag[i][j] = setflag[j][i] = 1;


//      epsilon[i][j] = epsilon_one;
//      sigma[i][j] = sigma_one;
//      cut_inner[i][j] = rmin * SS;
//      cut[i][j] = rmin * SM;
//      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR, "Incorrect args for pair coefficients RepCubic end");
//  printf("finished RepCubic coeff reading\n");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

//double PairRepCubic::init_one(int i, int j) {
//  if (setflag[i][j] == 0) {
//    epsilon[i][j] = mix_energy(epsilon[i][i], epsilon[j][j], sigma[i][i],
//        sigma[j][j]);
//    sigma[i][j] = mix_distance(sigma[i][i], sigma[j][j]);
//    cut_inner[i][j] = mix_distance(cut_inner[i][i], cut_inner[j][j]);
//    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
//  }
//
//  cut_inner_sq[i][j] = cut_inner[i][j] * cut_inner[i][j];
//  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
//  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);
//  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
//  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);
//
//  cut_inner[j][i] = cut_inner[i][j];
//  cut_inner_sq[j][i] = cut_inner_sq[i][j];
//  lj1[j][i] = lj1[i][j];
//  lj2[j][i] = lj2[i][j];
//  lj3[j][i] = lj3[i][j];
//  lj4[j][i] = lj4[i][j];
//
//  return cut[i][j];
//}
double PairRepCubic::init_one(int i, int j) {
  printf("RepCubic init_one\n");
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  return cut[i][j];
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void PairRepCubic::write_restart(FILE *fp) {
  printf("RepCubic write_restart\n");
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
//        fwrite(&epsilon[i][j], sizeof(double), 1, fp);
//        fwrite(&sigma[i][j], sizeof(double), 1, fp);
//        fwrite(&cut_inner[i][j], sizeof(double), 1, fp);
//        fwrite(&cut[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
        fwrite(&gamma_fc[i][j], sizeof(double), 1, fp);
      }
    }
//  printf("RepCubic write_restart\n");
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void PairRepCubic::read_restart(FILE *fp) {
  printf("RepCubic read_restart\n");
  read_restart_settings(fp);
  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0)
        fread(&setflag[i][j], sizeof(int), 1, fp);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
//          fread(&epsilon[i][j], sizeof(double), 1, fp);
//          fread(&sigma[i][j], sizeof(double), 1, fp);
//          fread(&cut_inner[i][j], sizeof(double), 1, fp);
//          fread(&cut[i][j], sizeof(double), 1, fp);
          fread(&cut[i][j], sizeof(double), 1, fp);
          fread(&gamma_fc[i][j], sizeof(double), 1, fp);
        }
//        MPI_Bcast(&epsilon[i][j], 1, MPI_DOUBLE, 0, world);
//        MPI_Bcast(&sigma[i][j], 1, MPI_DOUBLE, 0, world);
//        MPI_Bcast(&cut_inner[i][j], 1, MPI_DOUBLE, 0, world);
//        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&gamma_fc[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
//  printf("RepCubic read_restart\n");
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void PairRepCubic::write_restart_settings(FILE *fp) {
  printf("RepCubic write_restart_settings\n");
  fwrite(&mix_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void PairRepCubic::read_restart_settings(FILE *fp) {
  printf("RepCubic read_restart_sett\n");
  int me = comm->me;
  if (me == 0) {
    fread(&mix_flag, sizeof(int), 1, fp);
  }
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
}

/* ---------------------------------------------------------------------- */

double PairRepCubic::single(int /*i*/, int /*j*/, int itype, int jtype,
    double rsq, double /*factor_coul*/, double factor_lj, double &fforce) {
  printf("RepCubic single\n");
//  double r2inv, r6inv, forcelj, philj;
//  double r, t;
//  double rmin;
//
//  r2inv = 1.0 / rsq;
//  if (rsq <= cut_inner_sq[itype][jtype]) {
//    r6inv = r2inv * r2inv * r2inv;
//    forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
//  } else {
//    r = sqrt(rsq);
//    rmin = sigma[itype][jtype] * RT6TWO;
//    t = (r - cut_inner[itype][jtype]) / rmin;
//    forcelj = epsilon[itype][jtype] * (-DPHIDS + A3 * t * t / 2.0) * r / rmin;
//  }
//  fforce = factor_lj * forcelj * r2inv;
//
//  if (rsq <= cut_inner_sq[itype][jtype])
//    philj = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]);
//  else
//    philj = epsilon[itype][jtype] * (PHIS + DPHIDS * t - A3 * t * t * t / 6.0);
//
//  return factor_lj * philj;
//  printf("RepCubic single\n");
  return 0;
}
