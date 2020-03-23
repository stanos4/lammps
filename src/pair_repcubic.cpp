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
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(gamma_fc);
  }
}

/* ---------------------------------------------------------------------- */

void PairRepCubic::compute(int eflag, int vflag) {
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
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j_gamma = j;
      factor_ij = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      if (factor_ij != 0.0){
        cut_ij = cutsq[itype][jtype] * factor_ij * factor_ij;
        if (rsq < cut_ij) {
          r = sqrt(rsq);
          cut_ij = sqrt(cut_ij);
          dr = cut_ij - r;
          tk = gamma_fc[itype][jtype][sbmask(j_gamma)] * dr * dr * factor_ij;
          fpair = 3*tk/r;
        }

      }
      double fx = -fpair*delx;
      double fy = -fpair*dely;
      double fz = -fpair*delz;
      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
      if (newton_pair || j < nlocal){
        f[j][0] -= fx;
        f[j][1] -= fy;
        f[j][2] -= fz;
      }

      if (eflag) {
        epair = tk * dr;

      }
      if (evflag) {
        ev_tally(i,j,nlocal,newton_pair,epair,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairRepCubic::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(gamma_fc, n + 1, n + 1, 4, "pair:cut");
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairRepCubic::settings(int narg, char**/*arg*/) {
  if (narg != 0)
    error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairRepCubic::coeff(int narg, char **arg) {
  if (narg != 6)
    error->all(FLERR, "Incorrect args for pair coefficients RepCubic start");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(FLERR, arg[0], atom->ntypes, ilo, ihi);
  force->bounds(FLERR, arg[1], atom->ntypes, jlo, jhi);

  double cut_one = force->numeric(FLERR, arg[2]);
  double gamma_fc_one_12 = force->numeric(FLERR, arg[3]);
  double gamma_fc_one_13 = force->numeric(FLERR, arg[4]);
  double gamma_fc_one_14 = force->numeric(FLERR, arg[5]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {

      cut[i][j] = cut[j][i] = cut_one;
      gamma_fc[i][j][1] = gamma_fc[j][i][1] = gamma_fc_one_12;
      gamma_fc[i][j][2] = gamma_fc[j][i][2] = gamma_fc_one_13;
      gamma_fc[i][j][3] = gamma_fc[j][i][3] = gamma_fc_one_14;
      gamma_fc[i][j][0] = gamma_fc[j][i][0] = gamma_fc_one_14;
      setflag[i][j] = setflag[j][i] = 1;


      count++;
    }
  }

  if (count == 0)
    error->all(FLERR, "Incorrect args for pair coefficients RepCubic end");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairRepCubic::init_one(int i, int j) {
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  return cut[i][j];
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void PairRepCubic::write_restart(FILE *fp) {
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&cut[i][j], sizeof(double), 1, fp);
        fwrite(&gamma_fc[i][j], sizeof(double), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void PairRepCubic::read_restart(FILE *fp) {
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
          fread(&cut[i][j], sizeof(double), 1, fp);
          fread(&gamma_fc[i][j], sizeof(double), 1, fp);
        }
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&gamma_fc[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void PairRepCubic::write_restart_settings(FILE *fp) {
  fwrite(&mix_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void PairRepCubic::read_restart_settings(FILE *fp) {
  int me = comm->me;
  if (me == 0) {
    fread(&mix_flag, sizeof(int), 1, fp);
  }
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
}

/* ---------------------------------------------------------------------- */

double PairRepCubic::single(int /*i*/, int j, int itype, int jtype,
    double rsq, double /*factor_coul*/, double factor_lj, double &fforce) {
  double cut_ij,r,dr,tk;
  fforce = 0.;
  tk = 0.;
  dr = 0.;
  if (factor_lj != 0.0) {
    cut_ij = cutsq[itype][jtype] * factor_lj * factor_lj;
    if (rsq < cut_ij){
      r = sqrt(rsq);
      dr = cut_ij - r;
      tk = gamma_fc[itype][jtype][sbmask(j)] * dr * dr * factor_lj;
      fforce = 3*tk/r;
    }
  }
  return tk*dr;
}
