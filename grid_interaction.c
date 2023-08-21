/* Project part 2*/

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

    // /* Get the ranks of the neighboring processes */ //NEW
    // MPI_Cart_shift(grid->comm, 0, 1, &(grid->npr[LEFT]), &(grid->npr[RIGHT]));
    // MPI_Cart_shift(grid->comm, 1, 1, &(grid->npr[UP]), &(grid->npr[DOWN]));
    
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

    //field *fi = (field*)malloc((vol + bndry) * sizeof(field));
    srand(time(NULL)*(grid->grid_rank));
    /* Setting random field values based on even or odd indices in the lattice*/
    for(i=0;i<vol;i++)
    {
    /* Values inside local lattice*/
        if(grid->ipt[i] < vol/2) /* ipt is used*/  //even
        {
            fi[even_l].phi1 = grid->grid_rank; 
            fi[even_l].phi2 = grid->grid_rank;

            // fi[even_l].phi1 = (double) rand() / vol; 
            // fi[even_l].phi2 = (double) rand() / vol;
            even_l++;
        }
        else
        {
            // fi[odd_l].phi1 = (double) rand() / vol;
            // fi[odd_l].phi2 = (double) rand() / vol;

            fi[odd_l].phi1 = grid->grid_rank;
            fi[odd_l].phi2 = grid->grid_rank;
            odd_l++;   
        }
        if (grid->idn[i][0]>=vol+bndry/2 && grid->idn[i][0]<vol+bndry/2+grid->face0/2) // -0 odd
        {
            // fi[odd_b].phi1 = (double) rand() / vol;  //-0
            // fi[odd_b].phi2 = (double) rand() / vol;  //-0  

            fi[odd_b].phi1 = grid->grid_rank;
            fi[odd_b].phi2 = grid->grid_rank; 
            odd_b++;
        }
        if (grid->iup[i][0]>=vol+bndry/2 + grid->face0/2 && grid->iup[i][0]<vol+ bndry/2 +grid->face0) //+0 odd
        { 
            // fi[odd_b].phi1 = (double) rand() / vol; //+0  
            // fi[odd_b].phi2 = (double) rand() / vol; //+0

            fi[odd_b].phi1 = grid->grid_rank;  
            fi[odd_b].phi2 = grid->grid_rank;
            odd_b++;
        }
        if (grid->idn[i][1]>=vol+bndry/2+grid->face0 && grid->idn[i][1]<vol+bndry/2+grid->face0+grid->face1/2) //-1 odd
        {
            // fi[odd_b].phi1 = (double) rand() / vol; //-1
            // fi[odd_b].phi2 = (double) rand() / vol; //-1

            fi[odd_b].phi1 = grid->grid_rank;
            fi[odd_b].phi2 = grid->grid_rank;
            odd_b++;
        }
        if (grid->iup[i][1]>=vol+bndry/2 + grid->face0 + grid->face1/2 && grid->iup[i][1]<vol+bndry) //+1 odd
        {
            // fi[odd_b].phi1 = (double) rand() / vol; //+1
            // fi[odd_b].phi2 = (double) rand() / vol; //+1

            fi[odd_b].phi1 = grid->grid_rank;
            fi[odd_b].phi2 = grid->grid_rank;
            odd_b++;
        }
        if (grid->idn[i][0]>=vol && grid->idn[i][0]<vol+grid->face0/2 ) //-0 even
        {
            // fi[even_b].phi1 = (double) rand() / vol; 
            // fi[even_b].phi2 = (double) rand() / vol;

            fi[even_b].phi1 = grid->grid_rank;
            fi[even_b].phi2 = grid->grid_rank;
            even_b++;
        }
        if (grid->iup[i][0]>=vol + grid->face0/2 && grid->iup[i][0]<vol + grid->face0 )  //+0 even
        { 
            // fi[even_b].phi1 = (double) rand() / vol; 
            // fi[even_b].phi2 = (double) rand() / vol;

            fi[even_b].phi1 = grid->grid_rank; 
            fi[even_b].phi2 = grid->grid_rank;
            even_b++;
        }
        if (grid->idn[i][1]>=vol+grid->face0 &&  grid->idn[i][1]<vol+grid->face0+grid->face1/2) //-1 even
        {
            // fi[even_b].phi1 = (double) rand() / vol; 
            // fi[even_b].phi2 = (double) rand() / vol;

            fi[even_b].phi1 = grid->grid_rank; 
            fi[even_b].phi2 = grid->grid_rank;
            even_b++;
        }
        if (grid->iup[i][1]>=vol + grid->face0+ grid->face1/2 && grid->iup[i][1]<vol + bndry/2) //+1 even
        {
            // fi[even_b].phi1 = (double) rand() / vol; 
            // fi[even_b].phi2 = (double) rand() / vol;

            fi[even_b].phi1 = grid->grid_rank; 
            fi[even_b].phi2 = grid->grid_rank; 
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

    double *temp;
    temp = malloc(iL1*L2*sizeof(double));
    // double* temporary_bndry0;
    // double* temporary_bndry1;
    // double* temporary_bndry2;
    // double* temporary_bndry3;

    struct field *temporary_bndry0=(field*)malloc(iL1*L2 *sizeof(field));
    struct field *temporary_bndry1=(field*)malloc(iL1*L2 *sizeof(field));
    struct field *temporary_bndry2=(field*)malloc(iL0*L2 *sizeof(field));
    struct field *temporary_bndry3=(field*)malloc(iL0*L2 *sizeof(field));

    MPI_Status status;
    loc_vol=iL0*iL1*L2;

    /* MPI communications */
    for (j=0;j<loc_vol; j++)
    {
        if(grid->iup[j][0]>=loc_vol)  /* right boundary (9 -> 5(npr[+1])) */
        {
            temporary_bndry3[bnd3].phi1=fi[grid->ipt[j]].phi1;
            temporary_bndry3[bnd3].phi2=fi[grid->ipt[j]].phi2;
            // temp[bnd3]=fi[grid->ipt[j]].phi1;
            bnd3++;
            // nl = grid->map[grid->iup[j][0]-loc_vol]; //5
            // MPI_Sendrecv(&(fi[grid->ipt[j]].phi1), 1, MPI_DOUBLE, grid->npr[3], 0, &(fi[grid->idn[nl][0]].phi1), 1, MPI_DOUBLE, grid->grid_rank, 0, MPI_COMM_WORLD, &status);
            // printf("Rank %d msg0 Receving rank %d\n",grid->grid_rank,grid->npr[3]);
            // MPI_Sendrecv(&(fi[grid->ipt[j]].phi2), 1, MPI_DOUBLE, grid->npr[3], 1, &(fi[grid->idn[nl][0]].phi2), 1, MPI_DOUBLE, grid->grid_rank, 1, MPI_COMM_WORLD, &status);
            // printf("Rank %d msg1 Receving rank %d\n",grid->grid_rank,grid->npr[3]);
        }
        if(grid->idn[j][0]>=loc_vol) /* left boundary */
        {
            temporary_bndry2[bnd2].phi1=fi[grid->ipt[j]].phi1;
            temporary_bndry2[bnd2].phi2=fi[grid->ipt[j]].phi2;
            bnd2++;
            // nl= grid->map[grid->idn[j][0]-loc_vol];
            // MPI_Sendrecv(&(fi[grid->ipt[j]].phi1), 1, MPI_DOUBLE, grid->npr[2], 2, &(fi[grid->iup[nl][0]].phi1), 1, MPI_DOUBLE, grid->grid_rank, 2, MPI_COMM_WORLD, &status);
            // printf("Rank %d msg2 Receving rank %d\n",grid->grid_rank,grid->npr[2]);
            // MPI_Sendrecv(&(fi[grid->ipt[j]].phi2), 1, MPI_DOUBLE, grid->npr[2], 3, &(fi[grid->iup[nl][0]].phi2), 1, MPI_DOUBLE, grid->grid_rank, 3, MPI_COMM_WORLD, &status);
            // printf("Rank %d msg3 Receving rank %d\n",grid->grid_rank,grid->npr[2]);
        }
        if(grid->iup[j][1]>=loc_vol) /* up boundary */
        {
            temporary_bndry0[bnd0].phi1=fi[grid->ipt[j]].phi1;
            temporary_bndry0[bnd0].phi2=fi[grid->ipt[j]].phi2;
            bnd0++;
            // nl=grid->map[grid->iup[j][1]-loc_vol];
            // MPI_Sendrecv(&(fi[grid->ipt[j]].phi1), 1, MPI_DOUBLE, grid->npr[0], 4, &(fi[grid->idn[nl][1]].phi1), 1, MPI_DOUBLE, grid->grid_rank, 4, MPI_COMM_WORLD, &status);
            // printf("Rank %d msg4 Receving rank %d\n",grid->grid_rank,grid->npr[0]);
            // MPI_Sendrecv(&(fi[grid->ipt[j]].phi2), 1, MPI_DOUBLE, grid->npr[0], 5, &(fi[grid->idn[nl][1]].phi2), 1, MPI_DOUBLE, grid->grid_rank, 5, MPI_COMM_WORLD, &status);
            // printf("Rank %d msg5 Receving rank %d\n",grid->grid_rank,grid->npr[0]);
        }
        if(grid->idn[j][1]>=loc_vol) /* down boundary */
        {
            temporary_bndry1[bnd1].phi1=fi[grid->ipt[j]].phi1;
            temporary_bndry1[bnd1].phi2=fi[grid->ipt[j]].phi2;
            bnd1++;
            // nl=grid->map[grid->idn[j][1]-loc_vol];
            // MPI_Sendrecv(&(fi[grid->ipt[j]].phi1), 1, MPI_DOUBLE, grid->npr[1], 6, &(fi[grid->iup[nl][1]].phi1), 1, MPI_DOUBLE, grid->grid_rank, 6, MPI_COMM_WORLD, &status);
            // printf("Rank %d msg6 Receving rank %d\n",grid->grid_rank,grid->npr[1]);
            // MPI_Sendrecv(&(fi[grid->ipt[j]].phi2), 1, MPI_DOUBLE, grid->npr[1], 7, &(fi[grid->iup[nl][1]].phi2), 1, MPI_DOUBLE, grid->grid_rank, 7, MPI_COMM_WORLD, &status); 
            // printf("Rank %d msg7 Receving rank %d\n",grid->grid_rank,grid->npr[1]);   
        }
    }
    MPI_Barrier(grid->comm);

    if (grid->grid_rank==0)
    {
        printf("%f\n",temporary_bndry3[1].phi1);   
        printf("I with rank %d UP rank: %d and DOWN rank: %d\n",grid->grid_rank,grid->npr[0],grid->npr[1]);
    }
    MPI_Sendrecv_replace(temporary_bndry3, iL1*L2, MPI_DOUBLE, grid->npr[0], 0, grid->npr[1], 0, grid->comm, &status); //UP
    if (grid->grid_rank==0)
    {
        printf("%f\n",temporary_bndry3[1].phi1);   
        printf("I with rank %d UP rank: %d and DOWN rank: %d\n",grid->grid_rank,grid->npr[0],grid->npr[1]);
    }

    // MPI_Sendrecv_replace(temporary_bndry0.phi1, iL1*L2, MPI_DOUBLE, grid->npr[0], 0, grid->grid_rank, 0, MPI_COMM_WORLD, &status); //UP
    // MPI_Sendrecv_replace(temporary_bndry0.phi2, iL1*L2, MPI_DOUBLE, grid->npr[0], 1, grid->grid_rank, 1, MPI_COMM_WORLD, &status); //UP

    // MPI_Sendrecv_replace(temporary_bndry1.phi1, iL1*L2, MPI_DOUBLE, grid->npr[1], 2, grid->grid_rank, 2, MPI_COMM_WORLD, &status); //DOWN
    // MPI_Sendrecv_replace(temporary_bndry1.phi2, iL1*L2, MPI_DOUBLE, grid->npr[1], 3, grid->grid_rank, 3, MPI_COMM_WORLD, &status); //DOWN

    // MPI_Sendrecv_replace(temporary_bndry2.phi1, iL0*L2, MPI_DOUBLE, grid->npr[2], 4, grid->grid_rank, 4, MPI_COMM_WORLD, &status); //LEFT
    // MPI_Sendrecv_replace(temporary_bndry2.phi2, iL0*L2, MPI_DOUBLE, grid->npr[2], 5, grid->grid_rank, 5, MPI_COMM_WORLD, &status); //LEFT

    // MPI_Sendrecv_replace(temporary_bndry3.phi1, iL0*L2, MPI_DOUBLE, grid->npr[3], 6, grid->grid_rank, 6, MPI_COMM_WORLD, &status); //RIGHT
    // MPI_Sendrecv_replace(temporary_bndry3.phi2, iL0*L2, MPI_DOUBLE, grid->npr[3], 7, grid->grid_rank, 7, MPI_COMM_WORLD, &status); //RIGHT

    free(temporary_bndry0);
    free(temporary_bndry1);
    free(temporary_bndry2);
    free(temporary_bndry3);
}
// void update_fi_boundaries(GRID_INFO_T *grid, field *fi)
// {
//     int j;
//     int loc_vol;
//     int nl;
//     int bnd0=0;
//     int bnd1=0;
//     int bnd2=0;
//     int bnd3=0;
//     loc_vol=iL0*iL1*L2;
//
//     for(j=0; j<loc_vol;j++)
//     {
//         if(grid->iup[j][0]>=loc_vol) /* right */
//         {
//             nl = grid->map[grid->iup[j][0]-loc_vol];
//             fi[grid->idn[nl][0]].phi1= temporary_bndry3[bnd3].phi1;
//             fi[grid->idn[nl][0]].phi2= temporary_bndry3[bnd3].phi2;
//             bnd3++;
//         }
//         if(grid->idn[j][0]>=loc_vol) /* left boundary */
//         {
//             nl= grid->map[grid->idn[j][0]-loc_vol];
//             fi[grid->iup[nl][0]].phi1=temporary_bndry2[bnd2].phi1;
//             fi[grid->iup[nl][0]].phi2=temporary_bndry2[bnd2].phi2;
//             bnd2++;
//         }
//         if(grid->iup[j][1]>=loc_vol) /* up boundary */
//         {
//             nl=grid->map[grid->iup[j][1]-loc_vol];
//             fi[grid->idn[nl][1]].phi1=temporary_bndry0[bnd0].phi1;
//             fi[grid->idn[nl][1]].phi2=temporary_bndry0[bnd0].phi2;
//             bnd0++;
//         }
//         if(grid->idn[j][1]>=loc_vol) /* down boundary */
//         {
//             nl=grid->map[grid->idn[j][1]-loc_vol];
//             fi[grid->iup[nl][1]].phi1=temporary_bndry1[bnd1].phi1;
//             fi[grid->iup[nl][1]].phi2=temporary_bndry1[bnd1].phi2;
//             bnd1++;
//         }
//     }
// }

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
    // int NPROC0, NPROC1;
    // int L0, L1, L2;
    // int iL0, iL1, VOLUME;
    double begin, end, passed_time, local_passed_time;
    int vol;
    vol = iL0*iL1*L2;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &all_p);
    
    /* Getting the user input */
    //user_input(my_rank);
    
    /* Seed the random number generator */
    /* srand(time(NULL)); */

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
    //update_fi_boundaries(grid,fi);

    /* End timing */
    end=MPI_Wtime();
    local_passed_time=end-begin;
    MPI_Reduce(&local_passed_time,&passed_time,1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank==0)
    {
        printf("Total time of communication: %f\n",passed_time);
    }

    free(fi);
    free(grid->map);
    free(grid);
    /* Finalize the MPI environment */
    MPI_Finalize();
    return 0;
}
