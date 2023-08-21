/* Project */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"mpi.h"


#define eps  0.00001
#define max_iter 100000
double omega = 1.85;

#define square_val(x) ((x)*(x))

int main(int argc, char* argv[])
{
     
   int i,j;
   int n;
   int my_rank,p;
   int dim[2],periods[2];
   int m;
   double starttime,endtime;
   double local_time, tot_time;
   double** u;
   double* result;
   int n_bar,coordinates[2];
   double **temp;
   double err;
   int row,col;

   n = 256;

   void initialize_function(double** u, int row, int col, int m, int n, int n_bar);
   void gauss_seidel(int n_bar, double** u_old, int m, int row, int col, double** u_new);
   int bounday_point_check(int row, int col, int i, int j, int n_bar, int m);
   double** set_domain(int m,int n);
   void print_data(int n_bar, double** x, int m, int row, int col);
   
   MPI_Comm comm_2Dcart;

   /* MPI Start up*/
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &p);

   m = sqrt(p);

   n_bar = n/m + 2;

   /* Setting up the grid*/
   periods[0] = periods[1] = 0; /* Not periodic in nature*/
   dim[0] = dim[1] = m; /* number of block rows = number of block columns */

   MPI_Cart_create(MPI_COMM_WORLD, 2, dim, periods, 1, &comm_2Dcart);
   
   MPI_Comm_rank(comm_2Dcart, &my_rank);  
   MPI_Cart_coords(comm_2Dcart, my_rank, 2, coordinates);
   
   row = coordinates[0];  
   col = coordinates[1];
   
   if ( my_rank == 0 ) 
  {
    printf ( "\n" );
    printf ( "  POISSON MPI:\n" );
    printf ( "  2-D Poisson's equation using Gauss seidel algorithm\n" );
    printf ( "  *******************************************\n" );
    printf ( "  Size of the problem         = %d\n", n );
    printf ( "  Number of processes         = %d\n", p );
    printf ( "  Desired fractional accuracy = %f\n", eps );
    printf ( "\n" );
  }

   

   /* Allocating the 2D matrix*/
   u = set_domain(n_bar, n_bar);
   temp = set_domain(n_bar, n_bar);

   initialize_function(u, row, col, m, n, n_bar);

   MPI_Barrier(MPI_COMM_WORLD);
   starttime = MPI_Wtime();
  
   /* Gauss seidel iteration in parallel*/
   gauss_seidel(n_bar, u, m, row, col, temp);

   MPI_Barrier(MPI_COMM_WORLD);
   endtime = MPI_Wtime() - starttime;

   /* To compare Analytical and Numerical Solution */
   // print_data(n_bar, u, m, row, col);
   
   if (my_rank == 0)
       printf("  Jacobi Time taken: %e\n", endtime);

   /* Freeing up space*/
   MPI_Comm_free(&comm_2Dcart);

   MPI_Finalize();
}

/* Setting up x-coordinate and y-coordinates */
double x_coord(int row, int i, int n_bar, int m)
{
    int n;
    double h, x;

    n = m * (n_bar - 2);
    h = 1 / (double)(n + 1);

    return x = h * (row * (n_bar - 2) + i);
}

double y_coord(int col, int j, int n_bar, int m)
{
    int n;
    double h, y;

    n = m * (n_bar - 2);
    h = 1 / (double)(n + 1);

    return y = h * (col * (n_bar - 2) + j);
}


/* Function to define f */

double func_f(int row, int col, int i, int j, int n_bar, int m)
{
    double x, y;

    x = x_coord(row, i, n_bar, m);
    y = y_coord(col, j, n_bar, m);

    return 2 * (((1 + x) * sin(x + y)) - cos(x + y));
}

/* Function to define g*/
double func_g(double x, double y)
{
    return (1 + x) * sin(x + y);
}


int bounday_point_check(int row, int col, int i, int j, int n_bar, int m)
{
    if ((row == 0 && i == 0) || (row == m - 1 && i == n_bar - 1))
        return 1;
    if ((col == 0 && j == 0) || (col == m - 1 && j == n_bar - 1))
        return 1;

    return 0;
}

void initialize_function(double** u, int row, int col, int m, int n, int n_bar)
{
    int i, j;
    double x, y;

    for (i = 0; i < n_bar; i++)
    {
        for (j = 0; j < n_bar; j++)
        {
            if (bounday_point_check(row, col, i, j, n_bar, m))
            {
                x = x_coord(row, i, n_bar, m);
                y = y_coord(col, j, n_bar, m);

                u[i][j] = func_g(x, y);
            }
            else
                u[i][j] = 0.0;
        }
    }
}


void gauss_seidel(int local_n, double** u_old, int m, int row, int col, double** u_new)
{
    int i, j;
    int iterations = 0, h, n;
    int tag1 = 1, tag2 = 2;
    double local_sum, sum;
    double norm_calc;
    

    double func_f(int row, int col, int i, int j, int local_n, int m);    

    n = m * (local_n - 2);
    h = 1 / (double)(n + 1);

    /* Initializing custom dataype for col values*/
    MPI_Datatype colmpi;
    MPI_Status status;

    /* Defining colmpi */
    MPI_Type_vector(local_n, 1, local_n, MPI_DOUBLE, &colmpi);
    MPI_Type_commit(&colmpi);

    /* I will be indicating the checkerboard in the form of red and black points*/

    do
    {
        iterations++; /* Number of iterations*/
        local_sum = 0; /* Local sum */

        /* For Red Points */
        for (i = 1; i < local_n - 1; i++)
        {
            for (j = 1; j < local_n - 1; j++)
            {
                if ((i + j) % 2 == 0)
                {
                    /* Find new values. */
                    u_new[i][j] = (1 - omega) * u_old[i][j] + ((h * h * func_f(row, col, i, j, local_n, n) + (u_old[i - 1][j] + u_old[i + 1][j]
                        + u_old[i][j - 1] + u_old[i][j + 1])) * 0.25 * omega);

                   

                    /* Updates local_sum. */
                    local_sum = local_sum + square_val(u_new[i][j] - u_old[i][j]);
                }
            }
        }

        /* Now make sure that the local boundary is updated with the old values */

        for (i = 0; i < local_n; i++)
        {
            u_new[i][0] = u_old[i][0];
            u_new[i][local_n - 1] = u_old[i][local_n - 1];
        }
        for (j = 0; j < local_n; j++)
        {
            u_new[0][j] = u_old[0][j];
            u_new[local_n - 1][j] = u_old[local_n - 1][j];
        }

        /* Send data to adjacent processors */
        if (row < m - 1)
        {
            /* up */
            MPI_Send(&u_new[local_n - 2][0], local_n, MPI_DOUBLE,
                ((row + 1) * m + col), tag1, MPI_COMM_WORLD);
            MPI_Recv(&u_old[local_n - 1][0], local_n, MPI_DOUBLE,
                ((row + 1) * m + col), tag2, MPI_COMM_WORLD, &status);
        }

        if (row > 0)
        {
            /* down */
            MPI_Send(&u_new[1][0], local_n, MPI_DOUBLE,
                ((row - 1) * m + col), tag2, MPI_COMM_WORLD);
            MPI_Recv(&u_old[0][0], local_n, MPI_DOUBLE,
                ((row - 1) * m + col), tag1, MPI_COMM_WORLD, &status);
        }

        if (col < m - 1)
        {
            /* right */
            MPI_Send(&u_new[0][local_n - 2], 1, colmpi,
                (row * m + (col+1)), tag1, MPI_COMM_WORLD);
            MPI_Recv(&u_old[0][local_n - 1], 1, colmpi,
                (row * m + (col+1)), tag2, MPI_COMM_WORLD, &status);
        }

        if (col > 0)
        {
            /* left */
            MPI_Send(&u_new[0][1], 1, colmpi,
                (row * m + (col-1)), tag2, MPI_COMM_WORLD);
            MPI_Recv(&u_old[0][0], 1, colmpi,
                (row * m + (col-1)), tag1, MPI_COMM_WORLD, &status);
        }

        /* Copy u_new into u_old and do iterations again. */

        for (i = 1; i < local_n - 1; i++)
            for (j = 1; j < local_n - 1; j++)
                u_old[i][j] = u_new[i][j];


        /* For Black Points */

        for (i = 1; i < local_n - 1; i++)
        {
            for (j = 1; j < local_n - 1; j++)
            {
                if ((i + j) % 2 != 0)
                {
                    /* Find new values. */
                    u_new[i][j] = (1 - omega) * u_old[i][j] + ((h * h * func_f(row, col, i, j, local_n, n) + (u_old[i - 1][j] + u_old[i + 1][j]
                        + u_old[i][j - 1] + u_old[i][j + 1])) * 0.25 * omega);

                    /* Updates local_sum. */
                    local_sum = local_sum + square_val(u_new[i][j] - u_old[i][j]);
                }
            }
        }

        /* Now we need to make sure that the local boundary is updated with the old values */

        for (i = 0; i < local_n; i++)
        {
            u_new[i][0] = u_old[i][0];
            u_new[i][local_n - 1] = u_old[i][local_n - 1];
        }
        for (j = 0; j < local_n; j++)
        {
            u_new[0][j] = u_old[0][j];
            u_new[local_n - 1][j] = u_old[local_n - 1][j];
        }

        /* Transferring msgs to neighbours*/

        if (row < m - 1)
        {
            /* up */
            MPI_Send(&u_new[local_n - 2][0], local_n, MPI_DOUBLE,
                ((row + 1) * m + col), tag1, MPI_COMM_WORLD);
            MPI_Recv(&u_old[local_n - 1][0], local_n, MPI_DOUBLE,
                ((row + 1) * m + col), tag2, MPI_COMM_WORLD, &status);
        }

        if (row > 0)
        {
            /* down */
            MPI_Send(&u_new[1][0], local_n, MPI_DOUBLE,
                ((row - 1) * m + col), tag2, MPI_COMM_WORLD);
            MPI_Recv(&u_old[0][0], local_n, MPI_DOUBLE,
                ((row - 1) * m + col), tag1, MPI_COMM_WORLD, &status);
        }

        if (col < m - 1)
        {
            /* right */
            MPI_Send(&u_new[0][local_n - 2], 1, colmpi,
                (row * m + (col+1)), tag1, MPI_COMM_WORLD);
            MPI_Recv(&u_old[0][local_n - 1], 1, colmpi,
                (row * m + (col+1)), tag2, MPI_COMM_WORLD, &status);
        }

        if (col > 0)
        {
            /* left */
            MPI_Send(&u_new[0][1], 1, colmpi,
                (row * m + (col-1)), tag2, MPI_COMM_WORLD);
            MPI_Recv(&u_old[0][0], 1, colmpi,
                (row * m + (col-1)), tag1, MPI_COMM_WORLD, &status);
        }

        /* Copy u_new into u_old and do iterations again. */

        for (i = 1; i < local_n - 1; i++)
            for (j = 1; j < local_n - 1; j++)
                u_old[i][j] = u_new[i][j];


        /* Reduce local_sum in each proc to sum and broadcast sum. */

        MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        /* Calculate the norm*/

        norm_calc = sqrt(sum) / n;

    } while (iterations < max_iter && eps < norm_calc); /* end of do-while */


    if (((row * m) + col) == 0 && eps > norm_calc)
        printf("  Iterations : %d for the norm: %f\n", iterations, norm_calc);
    else if (((row * m) + col) == 0 && eps <= norm_calc)
        printf("  Gauss seidel does not converge");

    MPI_Type_free(&colmpi);

}


double** set_domain(int m, int n)
{
    double** u;
    int i;

    u = (double**)malloc(m * sizeof(double*));
    u[0] = (double*)malloc(m * n * sizeof(double));
    
    for (i = 1; i < m; i++)
        u[i] = &u[0][i * n];

    return u;
}

void print_data(int n_bar, double** x, int m, int row, int col)
{
    int i, j, k, l;
    double Calc_soln(int row, int col, int i, int j, int n_bar, int m);
    
    FILE *fptr;
    FILE *fpt;
    double soln;

    int n = m * (n_bar - 2);
    
    fptr = fopen("AnalyticalSoln.txt","w");
    fprintf(fptr,"%d\n %d\n",m,n);
    
    fpt = fopen("SORSoln.txt","w");
    fprintf(fpt,"%d\n %d\n",m,n);


    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == row && j == col)
            {
                for (k = 1; k < n_bar - 1; k++)
                {
                    for (l = 1; l < n_bar - 1; l++)
                    {
                        soln = Calc_soln(row, col, k, l, n_bar, m);
                        printf("Exact solution(%d,%d): %f \t", k, l, soln);
                        printf("Matrix value(%d,%d): %f\n", k, l, x[k][l]);
			fprintf(fpt,"%f \n",x[k][l]);
			fprintf(fptr,"%f \n",soln);
                    }
                }
            }
        }
}

double Calc_soln(int row, int col, int i, int j, int n_bar, int m)
{
    double x, y;

    x = x_coord(row, i, n_bar, m);
    y = y_coord(col, j, n_bar, m);

    return (1 + x) * sin(x + y);
}
