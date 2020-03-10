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
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "angle_vfm.h"
#include <mpi.h>
#include <cmath>
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.001

/* ---------------------------------------------------------------------- */

AngleVFM::AngleVFM(LAMMPS *lmp) : Angle(lmp) {}

/* ---------------------------------------------------------------------- */

AngleVFM::~AngleVFM()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(theta0);
    memory->destroy(rij0);
    memory->destroy(rik0);
    memory->destroy(beta);
  }
}

/* ---------------------------------------------------------------------- */

void AngleVFM::compute(int eflag, int vflag)
{
  int i1,i2,i3,n,type;
  double delx1,dely1,delz1,delx2,dely2,delz2;
  double eangle,f1[3],f3[3];
  double tk,dr;
  double r123,c0,a;

  eangle = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nanglelist; n++) {
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];

    // 1st bond

    delx1 = x[i1][0] - x[i2][0];
    dely1 = x[i1][1] - x[i2][1];
    delz1 = x[i1][2] - x[i2][2];

    // 2nd bond

    delx2 = x[i3][0] - x[i2][0];
    dely2 = x[i3][1] - x[i2][1];
    delz2 = x[i3][2] - x[i2][2];

    // Keating force & energy
    c0 = cos(theta0[type]);
    r123 = delx1*delx2 + dely1*dely2 + delz1*delz2;
    dr = r123 - rij0[type] * rik0[type] * c0;
    tk = dr * beta[type] / rij0[type] / rik0[type];

    if (eflag) eangle = dr * tk;

    a = -2.0 * tk;

    f1[0] = a*delx2;
    f1[1] = a*dely2;
    f1[2] = a*delz2;

//    f2[0] = -a*delx2 - a*delx1;
//    f2[1] = -a*dely2 - a*dely1;
//    f2[2] = -a*delz2 - a*delz1;

    f3[0] = a*delx1;
    f3[1] = a*dely1;
    f3[2] = a*delz1;


    // apply force to each of 3 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += f1[0];
      f[i1][1] += f1[1];
      f[i1][2] += f1[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= f1[0] + f3[0];
      f[i2][1] -= f1[1] + f3[1];
      f[i2][2] -= f1[2] + f3[2];
    }

    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0];
      f[i3][1] += f3[1];
      f[i3][2] += f3[2];
    }

    if (evflag) ev_tally(i1,i2,i3,nlocal,newton_bond,eangle,f1,f3,
                         delx1,dely1,delz1,delx2,dely2,delz2);
  }
}

/* ---------------------------------------------------------------------- */

void AngleVFM::allocate()
{
  allocated = 1;
  int n = atom->nangletypes;

  memory->create(theta0,n+1,"angle:theta0");
  memory->create(rij0,n+1,"angle:rij0");
  memory->create(rik0,n+1,"angle:rik0");
  memory->create(beta,n+1,"angle:beta");
  memory->create(setflag,n+1,"angle:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

void AngleVFM::coeff(int narg, char **arg)
{
  if (narg != 5) error->all(FLERR,"Incorrect args for angle coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nangletypes,ilo,ihi);

  double theta0_one = force->numeric(FLERR,arg[2]);
  double rij0_one = force->numeric(FLERR,arg[3]);
  double rik0_one = force->numeric(FLERR,arg[4]);
  double beta_one = force->numeric(FLERR,arg[5]);

  // convert theta0 from degrees to radians

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    theta0[i] = theta0_one/180.0 * MY_PI;
    rij0[i] = rij0_one;
    rik0[i] = rik0_one;
    beta[i] = beta_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for angle coefficients");
}

/* ---------------------------------------------------------------------- */

double AngleVFM::equilibrium_angle(int i)
{
  return theta0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

//void AngleCharmm::write_restart(FILE *fp)
//{
//  fwrite(&k[1],sizeof(double),atom->nangletypes,fp);
//  fwrite(&theta0[1],sizeof(double),atom->nangletypes,fp);
//  fwrite(&k_ub[1],sizeof(double),atom->nangletypes,fp);
//  fwrite(&r_ub[1],sizeof(double),atom->nangletypes,fp);
//}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void AngleVFM::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR,&theta0[1],sizeof(double),atom->nangletypes,fp,NULL,error);
    utils::sfread(FLERR,&rij0[1],sizeof(double),atom->nangletypes,fp,NULL,error);
    utils::sfread(FLERR,&rik0[1],sizeof(double),atom->nangletypes,fp,NULL,error);
    utils::sfread(FLERR,&beta[1],sizeof(double),atom->nangletypes,fp,NULL,error);
  }
  MPI_Bcast(&theta0[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&rij0[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&rik0[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&beta[1],atom->nangletypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nangletypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void AngleVFM::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nangletypes; i++)
    fprintf(fp,"%d %g %g %g %g\n",
            i,theta0[i]/MY_PI*180.0,rij0[i],rik0[i],beta[i]);
}

/* ---------------------------------------------------------------------- */

double AngleVFM::single(int type, int i1, int i2, int i3)
{
  double **x = atom->x;

  double delx1 = x[i1][0] - x[i2][0];
  double dely1 = x[i1][1] - x[i2][1];
  double delz1 = x[i1][2] - x[i2][2];
  domain->minimum_image(delx1,dely1,delz1);

  double delx2 = x[i3][0] - x[i2][0];
  double dely2 = x[i3][1] - x[i2][1];
  double delz2 = x[i3][2] - x[i2][2];
  domain->minimum_image(delx2,dely2,delz2);

  double c0 = cos(theta0[type]);
  double r123 = delx1*delx2 + dely1*dely2 + delz1*delz2;
  double dr = r123 - rij0[type] * rik0[type] * c0;
  double tk = dr * beta[type] / rij0[type] / rik0[type];

  return (dr*tk);
}
