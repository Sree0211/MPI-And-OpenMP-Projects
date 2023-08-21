/* Project - part2 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define NPROC0 4
#define NPROC1 4
#define L0 20
#define L1 20
#define L2 20
#define VOLUME L0 *L1 *L2
#define iL0 L0 / NPROC0
#define iL1 L1 / NPROC1
#define alpha 0.5

#define LEFT 0
#define RIGHT 1
#define UP 2
#define DOWN 3

typedef struct GRID_INFO_T
{
    int p;
    MPI_Comm comm;
    int q;
    int boundary;
    int grid_rank;
    int cpr[2];
    int npr[4];
    int face0;
    int face1;
    int ipt[iL0 * iL1 * L2];
    int iup[iL0 * iL1 * L2][2];
    int idn[iL0 * iL1 * L2][2];
    int *map;
} GRID_INFO_T;

typedef struct field
{
    double phi1;
    double phi2;
}field;


void Setup_grid(GRID_INFO_T *grid)
{
    grid->face0 = (L2 * iL0);
    grid->face1 = (L2 * iL1);

    grid->boundary = 2 * (grid->face0 + grid->face1);
    grid->map = (int *)malloc(grid->boundary * sizeof(int));

    int periods[2] = {1, 1}; /* Set the lattice to be periodic */

    MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));
    grid->q = sqrt((NPROC0 * NPROC1));

    if (grid->p < 2)
    {
        printf("Number of %d processors not enough\n", grid->p);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Create a 2D grid structure */
    int dims[2] = {grid->q, grid->q};

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &(grid->comm));
    /* MPI_Comm_rank(grid->comm, &(grid->my_rank)); */

    /* Get the rank and coordinates of the process in the lattice */
    MPI_Comm_rank(grid->comm, &(grid->grid_rank));
    MPI_Cart_coords(grid->comm, grid->grid_rank, 2, grid->cpr);

    // /* Get the ranks of the neighboring processes */ OLDDD
    MPI_Cart_shift(grid->comm, 0, 1, &(grid->npr[DOWN]), &(grid->npr[UP]));
    MPI_Cart_shift(grid->comm, 1, 1, &(grid->npr[LEFT]), &(grid->npr[RIGHT]));
    // #define UP 0
    // #define DOWN 1
    // #define LEFT 2
    // #define RIGHT 3
    /* MPI_Comm_free(&grid->comm); */
}

// void user_input(my_rank)
// {
//     if (my_rank==0)   
//     {
//         scanf("%d", &NPROC0);
//         scanf("%d", &NPROC1);
//         scanf("%d", &L0);
//         scanf("%d", &L1);
//         scanf("%d", &L2);
//     }
//     MPI_Bcast(&NPROC0, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&NPROC1, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&L0, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&L1, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&L2, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Barrier(MPI_COMM_WORLD);
//     VOLUME=L0 *L1 *L2;
//     iL0=L0 / NPROC0;
//     iL1=L1 / NPROC1;
// }

/* Function to define Action*/
void define_action(GRID_INFO_T *grid, field *fi)
{
    int vol;
    vol = iL0 * iL1 * L2;

    double k = 0.1;
    double lambda = 0.2;

    double S=0;
    double sum_phi=0;

    for(int i=0;i<vol;i++)  // local volume is considered
    {
        sum_phi += (fi[i].phi1*fi[grid->iup[i][0]].phi1) + (fi[i].phi1*fi[grid->iup[i][1]].phi1)  + (fi[i].phi2*fi[grid->iup[i][0]].phi2) + (fi[i].phi2*fi[grid->iup[i][1]].phi2));
        S += -2*k*sum_phi +  (fi[i].phi1)^2 + (fi[i].phi2)^2 + lambda*((fi[i].phi1)^2+ (fi[i].phi2)^2 -1);
    }

}

void hor_update(GRID_INFO_T *grid, field *fi, double lambda, double k, int vol)
{
    // Choose a random lattice position
    srand(time(NULL));
    int ix = rand() % vol;

    // local magnetic field 'b'
    double *B_phi1;
    double *B_phi2;

    B_phi1 = (double *)malloc(vol * sizeof(double));
    B_phi1 = (double *)malloc(vol * sizeof(double));

    // Setting values of local magnetic field 'B'
    for(int i=0;i<vol;i++)
    {
        B_phi1[i] = k*(fi[grid->iup[i][0]].phi1 + fi[grid->iup[i][1]].phi1 + fi[grid->idn[i][0]].phi1 + fi[grid->idn[i][1]].phi1);
        B_phi2[i] = k*(fi[grid->iup[i][0]].phi2 + fi[grid->iup[i][1]].phi2 + fi[grid->idn[i][0]].phi2 + fi[grid->idn[i][1]].phi2); 
    }

    // Change in action
    double delta_S = 0;
    for(int mu=0;mu<2;mu++)
    {
        delta_S += -2*k*(fi[ix].phi1*fi[grid->iup[ix][mu]].phi1 + fi[ix].phi2*fi[grid->iup[ix][mu]].phi2 + fi[ix].phi1*fi[grid->idn[ix][mu]].phi1 + fi[ix].phi2*fi[grid->idn[ix][mu]].phi2 );
        delta_S += 2*mu^2*(fi[ix].phi1^2 + fi[ix].phi2^2) + lambda* (fi[ix].phi1^2 + fi[ix].phi2^2)^2;
    }

    // Overrelaxation parameters
    double v_alpha2 = 1 + ((alpha - 1)/2*lambda);
    double c_alpha = 0;
    double delta_V = 0; 

    c_alpha = lambda * (v_alpha2^4 - 1) + (alpha^-1 - 1) * (B_phi1[ix] * B_phi1[ix] + B_phi2[ix] * B_phi2[ix]);

    // Updated phi values
    double phi1_updated = 0;
    double phi2_updated = 0;

    phi1_updated = (1 + alpha) * fi[ix].phi1 - alpha * B_phi1[ix] - (alpha / (2 * lambda)) * delta_S;
    phi2_updated = (1 + alpha) * fi[ix].phi2 - alpha * B_phi2[ix] - (alpha / (2 * lambda)) * delta_S;
    
    // Calculating the scalar potential 
    delta_V = (phi1_updated - fi[ix].phi1)^2 + (phi2_updated - fi[ix].phi2)^2 + lambda * (phi1_updated^2 + phi2_updated^2 - fi[ix].phi1^2 - fi[ix].phi2^2- v_alpha2)^2 - c_alpha;

}

/* Scalar field definition */
void define_fi(GRID_INFO_T *grid,field *fi)
{   
    int vol;
    vol = iL0 * iL1 * L2;
    int bndry = 2 * (grid->face0 + grid->face1);
    int i,j;
    int even_l = 0;
    int odd_l  = vol/2;
    int even_b = vol;
    int odd_b  = vol+bndry/2;

    srand(time(NULL)*(grid->grid_rank));
    /* Setting random field values based on even or odd indices in the lattice*/
    for(i=0;i<vol;i++)
    {
    /* Values inside local lattice*/
        if(grid->ipt[i] < vol/2) /* ipt is used*/  //even
        {
            fi[even_l].phi1 = (double) rand() / RAND_MAX*vol;
            fi[even_l].phi2 = (double) rand() / RAND_MAX*vol;
            even_l++;
        }
        else
        {
            fi[odd_l].phi1 = (double) rand() / RAND_MAX*vol;
            fi[odd_l].phi2 = (double) rand() / RAND_MAX*vol;
            odd_l++;   
        }
        if (grid->idn[i][0]>=vol+bndry/2 && grid->idn[i][0]<vol+bndry/2+grid->face0/2) // -0 odd
        {
            fi[odd_b].phi1 = (double) rand() / RAND_MAX*vol;
            fi[odd_b].phi2 = (double) rand() / RAND_MAX*vol;
            odd_b++;
        }
        if (grid->iup[i][0]>=vol+bndry/2 + grid->face0/2 && grid->iup[i][0]<vol+ bndry/2 +grid->face0) //+0 odd
        { 
            fi[odd_b].phi1 = (double) rand() / RAND_MAX*vol; 
            fi[odd_b].phi2 = (double) rand() / RAND_MAX*vol;
            odd_b++;
        }
        if (grid->idn[i][1]>=vol+bndry/2+grid->face0 && grid->idn[i][1]<vol+bndry/2+grid->face0+grid->face1/2) //-1 odd
        {
            fi[odd_b].phi1 = (double) rand() / RAND_MAX*vol;
            fi[odd_b].phi2 = (double) rand() / RAND_MAX*vol;
            odd_b++;
        }
        if (grid->iup[i][1]>=vol+bndry/2 + grid->face0 + grid->face1/2 && grid->iup[i][1]<vol+bndry) //+1 odd
        {
            fi[odd_b].phi1 = (double) rand() / RAND_MAX*vol;
            fi[odd_b].phi2 = (double) rand() / RAND_MAX*vol;
            odd_b++;
        }
        if (grid->idn[i][0]>=vol && grid->idn[i][0]<vol+grid->face0/2 ) //-0 even
        {
            fi[even_b].phi1 = (double) rand() / RAND_MAX*vol;
            fi[even_b].phi2 = (double) rand() / RAND_MAX*vol;
            even_b++;
        }
        if (grid->iup[i][0]>=vol + grid->face0/2 && grid->iup[i][0]<vol + grid->face0 )  //+0 even
        { 
            fi[even_b].phi1 = (double) rand() / RAND_MAX*vol;
            fi[even_b].phi2 = (double) rand() / RAND_MAX*vol;
            even_b++;
        }
        if (grid->idn[i][1]>=vol+grid->face0 &&  grid->idn[i][1]<vol+grid->face0+grid->face1/2) //-1 even
        {
            fi[even_b].phi1 = (double) rand() / RAND_MAX*vol; 
            fi[even_b].phi2 = (double) rand() / RAND_MAX*vol;
            even_b++;
        }
        if (grid->iup[i][1]>=vol + grid->face0+ grid->face1/2 && grid->iup[i][1]<vol + bndry/2) //+1 even
        {
            fi[even_b].phi1 = (double) rand() / RAND_MAX*vol; 
            fi[even_b].phi2 = (double) rand() / RAND_MAX*vol;
            even_b++;
        } 
    }
}

void point_communications(GRID_INFO_T *grid, field *fi)
{
    int j;
    int nl;
    int loc_vol;
    int bnd0=0;
    int bnd1=0;
    int bnd2=0;
    int bnd3=0;

    int bnd0_upd=0;
    int bnd1_upd=0;
    int bnd2_upd=0;
    int bnd3_upd=0;

    struct field *temporary_bndry0=(field*)malloc(iL1*L2 *sizeof(field));
    struct field *temporary_bndry1=(field*)malloc(iL1*L2 *sizeof(field));
    struct field *temporary_bndry2=(field*)malloc(iL0*L2 *sizeof(field));
    struct field *temporary_bndry3=(field*)malloc(iL0*L2 *sizeof(field));

    MPI_Status status;
    loc_vol=iL0*iL1*L2;

    /* MPI communications */
    for (j=0;j<loc_vol; j++)
    {
        if(grid->iup[j][0]>=loc_vol)  /* right boundary */
        {
            temporary_bndry1[bnd1].phi1=fi[grid->ipt[j]].phi1;
            temporary_bndry1[bnd1].phi2=fi[grid->ipt[j]].phi2;
            bnd1++;
        }
        if(grid->idn[j][0]>=loc_vol) /* left boundary */
        {
            temporary_bndry0[bnd0].phi1=fi[grid->ipt[j]].phi1;
            temporary_bndry0[bnd0].phi2=fi[grid->ipt[j]].phi2;
            bnd0++;
        }
        if(grid->iup[j][1]>=loc_vol) /* up boundary */
        {
            temporary_bndry2[bnd2].phi1=fi[grid->ipt[j]].phi1;
            temporary_bndry2[bnd2].phi2=fi[grid->ipt[j]].phi2;
            bnd2++;
        }
        if(grid->idn[j][1]>=loc_vol) /* down boundary */
        {
            temporary_bndry3[bnd3].phi1=fi[grid->ipt[j]].phi1;
            temporary_bndry3[bnd3].phi2=fi[grid->ipt[j]].phi2;
            bnd3++;  
        }
    }
    MPI_Barrier(grid->comm);

    if (grid->grid_rank==0)
    {
        printf("Field values of Process %d before change: \n",grid->grid_rank);
        printf("\n");
        int i;
        for (i=loc_vol; i<grid->boundary+loc_vol; i++)
        {
            printf("Phi1 is: %f  Phi2: %f\n",fi[i].phi1,fi[i].phi2);
        }
        printf("\n");
    }

    MPI_Sendrecv_replace(temporary_bndry0, 2*iL1*L2, MPI_DOUBLE, grid->npr[0], 0, grid->npr[1], 0, grid->comm, &status); /* in LEFT  */
    MPI_Sendrecv_replace(temporary_bndry1, 2*iL1*L2, MPI_DOUBLE, grid->npr[1], 1, grid->npr[0], 1, grid->comm, &status); /* in RIGHT */
    MPI_Sendrecv_replace(temporary_bndry2, 2*iL1*L2, MPI_DOUBLE, grid->npr[2], 2, grid->npr[3], 2, grid->comm, &status); /* in UP    */
    MPI_Sendrecv_replace(temporary_bndry3, 2*iL1*L2, MPI_DOUBLE, grid->npr[3], 3, grid->npr[2], 3, grid->comm, &status); /* in DOWN  */ 
    /* #define LEFT 0  */ 
    /* #define RIGHT 1 */ 
    /* #define UP 2    */ 
    /* #define DOWN 3  */ 

    /* update_fi_boundaries */
    for(j=0; j<loc_vol;j++)
    {
        if(grid->iup[j][0]>=loc_vol) /* right */
        {
            nl = grid->map[grid->iup[j][0]-loc_vol];
            fi[grid->idn[nl][0]].phi1= temporary_bndry1[bnd1_upd].phi1;
            fi[grid->idn[nl][0]].phi2= temporary_bndry1[bnd1_upd].phi2;
            bnd1_upd++;
        }
        if(grid->idn[j][0]>=loc_vol) /* left boundary */
        {
            nl= grid->map[grid->idn[j][0]-loc_vol];
            fi[grid->iup[nl][0]].phi1=temporary_bndry0[bnd0_upd].phi1;
            fi[grid->iup[nl][0]].phi2=temporary_bndry0[bnd0_upd].phi2;
            bnd0_upd++;
        }
        if(grid->iup[j][1]>=loc_vol) /* up boundary */
        {
            nl=grid->map[grid->iup[j][1]-loc_vol];
            fi[grid->idn[nl][1]].phi1=temporary_bndry2[bnd2_upd].phi1;
            fi[grid->idn[nl][1]].phi2=temporary_bndry2[bnd2_upd].phi2;
            bnd2_upd++;
        }
        if(grid->idn[j][1]>=loc_vol) /* down boundary */
        {
            nl=grid->map[grid->idn[j][1]-loc_vol];
            fi[grid->iup[nl][1]].phi1=temporary_bndry3[bnd3_upd].phi1;
            fi[grid->iup[nl][1]].phi2=temporary_bndry3[bnd3_upd].phi2;
            bnd3_upd++;
        }
    }
    if (grid->grid_rank==0)
    {
        printf("Field values of Process %d after exchange: \n",grid->grid_rank);
        printf("\n");
        int l;
        for (l=loc_vol; l<grid->boundary+loc_vol; l++)
        {
            printf("Phi1 is: %f  Phi2: %f\n",fi[l].phi1,fi[l].phi2);
        }
        printf("\n");
    }
    free(temporary_bndry0);
    free(temporary_bndry1);
    free(temporary_bndry2);
    free(temporary_bndry3);
}

void define_ipt(GRID_INFO_T *grid) /* Function to calculate ipt */
{
    int vol;
    vol = iL0 * iL1 * L2;
    int beg = 0;
    int mid = (iL0 * iL1 * L2) / 2;

    for (int i = 0; i < L2; i++)
    {
        for (int j = 0; j < iL1; j++)
        {
            for (int k = 0; k < iL0; k++)
            {
                if ((i + j + k) % 2 == 0)
                {
                    grid->ipt[k + j * iL0 + i * iL0 * iL1] = beg;
                    beg++;
                }
                else
                {
                    grid->ipt[k + j * iL0 + i * iL0 * iL1] = mid;
                    mid++;
                }
            }
        }
    }
}

void define_iup(GRID_INFO_T *grid)
{
    int vol;
    vol = iL0 * iL1 * L2;
    int bndry = 2 * (grid->face0 + grid->face1);
    int b = 0;
    int count[4] = {0, 0, 0, 0}; /* for updating every element in iup for all 4 bounday conditions */

    /* Setting up the index values for positive 0 and 1 direction for even and odd points */
    int *pos0_o, *pos0_e;
    int *pos1_o, *pos1_e;

    pos0_o = (int *)malloc(grid->face0 / 2 * sizeof(int));
    pos0_e = (int *)malloc(grid->face0 / 2 * sizeof(int));
    pos1_o = (int *)malloc(grid->face1 / 2 * sizeof(int));
    pos1_e = (int *)malloc(grid->face1 / 2 * sizeof(int));

    int cnt = 0;
    for (int i = vol + grid->face0 / 2; i < vol + grid->face0; i++)
    {
        pos0_e[cnt] = i;
        cnt++;
    }
    cnt = 0;
    for (int i = vol + grid->face0 + grid->face1 / 2; i < vol + bndry / 2; i++)
    {
        pos1_e[cnt] = i;
        cnt++;
    }
    cnt = 0;
    for (int i = vol + bndry / 2 + grid->face0 / 2; i < vol + bndry / 2 + grid->face0; i++)
    {
        pos0_o[cnt] = i;
        cnt++;
    }
    cnt = 0;
    for (int i = vol + bndry / 2 + grid->face0 + grid->face1 / 2; i < vol + bndry; i++)
    {
        pos1_o[cnt] = i;
        cnt++;
    }

    int sizet = sizeof(pos0_e) / sizeof(int);
    /* printf("size of poso_e is %d\n",sizet); */

    int ix;
    for (int k = 0; k < L2; k++)
    {
        for (int i = 0; i < iL0; i++)
        {
            for (int j = 0; j < iL1; j++)
            {
                ix = i + j * iL0 + k * iL0 * iL1; /* Current lexicographic index*/
                if (j == iL1 - 1 && i < iL0 - 1)  /* Last column*/
                {
                    if ((i + j + k) % 2 == 0)
                    {
                        grid->iup[ix][1] = pos1_e[count[0]];
                        grid->iup[ix][0] = ix + 1;
                        count[0]++;
                    }
                    else
                    {
                        grid->iup[ix][1] = pos1_o[count[1]];
                        grid->iup[ix][0] = ix + 1;
                        count[1]++;
                    }
                }
                else if (i == iL0 - 1 && j < iL1 - 1) /* Last row */
                {
                    if ((i + j + k) % 2 == 0)
                    {
                        grid->iup[ix][0] = pos0_e[count[2]];
                        grid->iup[ix][1] = ix + iL0;
                        count[2]++;
                    }
                    else
                    {
                        grid->iup[ix][0] = pos0_o[count[3]];
                        grid->iup[ix][1] = ix + iL0;
                        count[3]++;
                    }
                }
                else if (i == iL0 - 1 && j == iL0 - 1) /* top right corner element in the local lattice which has both iup values outside the local lattice */
                {
                    if ((i + j + k) % 2 == 0)
                    {
                        grid->iup[ix][0] = pos0_e[count[2]];
                        grid->iup[ix][1] = pos1_e[count[0]];
                        count[2]++;
                        count[0]++;
                    }
                    else
                    {
                        grid->iup[ix][0] = pos0_o[count[3]];
                        grid->iup[ix][1] = pos1_o[count[1]];
                        count[3]++;
                        count[1]++;
                    }
                }
                else /* values inside the lattice which have iup within the local lattice */
                {
                    grid->iup[ix][0] = ix + 1;
                    grid->iup[ix][1] = ix + iL0;
                }
            }
        }
    }
    free(pos0_o);
    free(pos0_e);
    free(pos1_e);
    free(pos1_o);
}

void define_idn(GRID_INFO_T *grid)
{
    int vol;
    vol = iL0 * iL1 * L2;
    int bndry = 2 * (grid->face0 + grid->face1);
    int b = 0;
    int count[4] = {0, 0, 0, 0}; /* for updating every element in iup for all 4 bounday conditions*/

    int *neg0_o, *neg0_e;
    int *neg1_o, *neg1_e;

    neg0_o = (int *)malloc(grid->face0 / 2 * sizeof(int));
    neg0_e = (int *)malloc(grid->face0 / 2 * sizeof(int));
    neg1_o = (int *)malloc(grid->face1 / 2 * sizeof(int));
    neg1_e = (int *)malloc(grid->face1 / 2 * sizeof(int));

    int cnt = 0;
    for (int i = vol; i < vol + grid->face0 / 2; i++)
    {
        neg0_e[cnt] = i;
        cnt++;
    }
    cnt = 0;
    for (int i = vol + grid->face0; i < vol + grid->face0 + grid->face1 / 2; i++)
    {
        neg1_e[cnt] = i;
        cnt++;
    }
    cnt = 0;
    for (int i = vol + bndry / 2; i < vol + bndry / 2 + grid->face0 / 2; i++)
    {
        neg0_o[cnt] = i;
        cnt++;
    }
    cnt = 0;
    for (int i = vol + bndry / 2 + grid->face0; i < vol + bndry / 2 + grid->face0 + grid->face1 / 2; i++)
    {
        neg1_o[cnt] = i;
        cnt++;
    }

    int ix;
    for (int k = 0; k < L2; k++)
    {
        for (int i = 0; i < iL0; i++)
        {
            for (int j = 0; j < iL1; j++)
            {
                ix = i + j * iL0 + k * iL0 * iL1; /* Current lexicographic index*/
                if (j == 0 && i > 0)              /* first column*/
                {
                    if ((i + j + k) % 2 == 0)
                    {
                        grid->idn[ix][1] = neg1_e[count[0]];
                        grid->idn[ix][0] = ix - 1;
                        count[0]++;
                    }
                    else
                    {
                        grid->idn[ix][1] = neg1_o[count[1]];
                        grid->idn[ix][0] = ix - 1;
                        count[1]++;
                    }
                }
                else if (i == 0 && j > 0) /* first row */
                {
                    if ((i + j + k) % 2 == 0)
                    {
                        grid->idn[ix][0] = neg0_e[count[2]];
                        grid->idn[ix][1] = ix - iL0;
                        count[2]++;
                    }
                    else
                    {
                        grid->idn[ix][0] = neg0_o[count[3]];
                        grid->idn[ix][1] = ix - iL0;
                        count[3]++;
                    }
                }
                else if (i == 0 && j == 0) /* bottom-left element in the local lattice which has both idn values outside the local lattice */
                {
                    if ((i + j + k) % 2 == 0)
                    {
                        grid->idn[ix][0] = neg0_e[count[2]];
                        grid->idn[ix][1] = neg1_e[count[0]];
                        count[2]++;
                        count[0]++;
                    }
                    else
                    {
                        grid->idn[ix][0] = neg0_o[count[3]];
                        grid->idn[ix][1] = neg1_o[count[1]];
                        count[3]++;
                        count[1]++;
                    }
                }
                else /* values inside the lattice which have iup within the local lattice */
                {
                    grid->idn[ix][0] = ix - 1;
                    grid->idn[ix][1] = ix - iL0;
                }
            }
        }
    }
    free(neg0_o);
    free(neg0_e);
    free(neg1_e);
    free(neg1_o);
}

void define_map(GRID_INFO_T *grid)
{
    int loc_vol = iL0 * iL1 * L2;
    int bndry = 2 * (grid->face0 + grid->face1); // 400 points
    int ix;
    for (int k = 0; k < L2; k++)
    {
        for (int i = 0; i < iL0; i++)
        {
            for (int j = 0; j < iL1; j++)
            {
                ix = i + j * iL0 + k * iL0 * iL1; /* Current lexicographic index*/

                /* Verifying iup and idn to map local indices in neighbouring process*/

                /* Calculates the local indices in neighbouring process which will be a copy of subsequent element(first element) in the same row*/
                if (grid->iup[ix][0] >= loc_vol)
                {
                    // grid->map[grid->iup[ix][0]-loc_vol] = ix - (iL0-1);
                    grid->map[grid->iup[ix][0] - loc_vol] = ix - (iL0 - 1);
                }

                /* Calculates the local indices in neighbouring process which will be a copy of subsequent element(first element) in the same column*/
                if (grid->iup[ix][1] >= loc_vol)
                {
                    grid->map[grid->iup[ix][1] - loc_vol] = ix - ((iL0 - 1) * iL1);
                }
                /* Similar calculation for idn*/
                if (grid->idn[ix][0] >= loc_vol)
                {
                    grid->map[grid->idn[ix][0] - loc_vol] = ix + (iL0 - 1);
                }

                if (grid->idn[ix][1] >= loc_vol)
                {
                    grid->map[grid->idn[ix][1] - loc_vol] = ix + ((iL0 - 1) * iL1);
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    /* Initialize the MPI environment */
    MPI_Init(&argc, &argv);
    /* Get the rank and size of the process */
    int my_rank, all_p;
    /* Variables for user input and bcast */
    /* int NPROC0, NPROC1;   */ 
    /* int L0, L1, L2;       */ 
    /* int iL0, iL1, VOLUME; */ 
    double begin, end, passed_time, local_passed_time;
    int vol;
    vol = iL0*iL1*L2;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &all_p);
    
    /* Getting the user input */
    /* user_input(my_rank);   */

    /* Grid and array setup */
    struct GRID_INFO_T *grid = malloc(sizeof(struct GRID_INFO_T));
    
    Setup_grid(grid);
    define_ipt(grid);
    define_iup(grid);
    define_idn(grid);
    define_map(grid);

    struct field *fi = (field*)malloc((vol + grid->boundary) * sizeof(field));
    define_fi(grid,fi);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Begin timing */
    begin=MPI_Wtime();
    
    /* Communication between points */
    point_communications(grid,fi);

    /* End timing */
    end=MPI_Wtime();
    local_passed_time=end-begin;
    MPI_Reduce(&local_passed_time,&passed_time,1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank==0)
    {
        printf("Total time of communication: %f",passed_time);
    }
    free(fi);
    free(grid->map);
    free(grid);
    /* Finalize the MPI environment */
    MPI_Finalize();
    return 0;
}
