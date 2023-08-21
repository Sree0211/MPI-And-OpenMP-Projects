/* Project */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"mpi.h"


#define eps  0.00001
#define max_iter  100000

#define square_val(x) ((x)*(x))

int main(int argc, char* argv[])
{
     
   int i,j;
   int n;
   int my_rank,p;
   int dim[2],periods[2];
   int row,col;
   double starttime,endtime;
   double local_time, tot_time;
   double** u;
   double* result;
   int n_bar,coordinates[2];
   double **temp;
   double err;
   int m;

   n = 16;

   void initialize_function(double** u, int row, int col, int m, int n, int n_bar);
   void jacobi_iteration(int local_n, double **x_old, int m, int row, int col, double **x_new);

   int bounday_point_check(int row, int col, int i, int j, int n_bar, int m);
   double** set_domain(int m,int n);
   void print_data(int n_bar, double** x, int m, int row, int col);
   double func_f(int row, int col, int i, int j, int n_bar, int m);
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
    printf ( "  2-D Poisson's equation using Jacobi algorithm\n" );
    printf ( "  *******************************************\n" );
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
   
   jacobi_iteration(n_bar, u, m, row, col, temp);

   MPI_Barrier(MPI_COMM_WORLD);
   endtime = MPI_Wtime() - starttime;

   print_data(n_bar, u, m,  row, col);
   
   if (my_rank == 0)
       printf("Jacobi Time taken: %e\n", endtime);

   /* Freeing up space*/
   MPI_Comm_free(&comm_2Dcart);


   MPI_Finalize();
}

int bounday_point_check(int row, int col, int i, int j, int n_bar, int m)
{
    if ((row == 0 && i == 0) || (row == m - 1 && i == n_bar - 1))
        return 1;
    if ((col == 0 && j == 0) || (col == m - 1 && j == n_bar - 1))
        return 1;

    return 0;
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

/* Function to define g*/
double func_g(double x, double y)
{
    return (1 + x) * sin(x + y);
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

/* Function to define f */

double func_f(int row, int col, int i, int j, int n_bar, int m)
{
    double x, y;

    x = x_coord(row, i, n_bar, m);
    y = y_coord(col, j, n_bar, m);

    return 2 * (((1 + x) * sin(x + y)) - cos(x + y));
}


int  rank_of(int row, int col, int m)
{
  return m*row + col;
}

void jacobi_iteration(int local_n, double **x_old, int m, int row, int col, double **x_new)
{
  int i, j, n, tag1 = 1, tag2 = 2, num_iter = 0,h;
  double local_sum, sum, diff_norm, exact, delta, max, global_delta;
  double f(int row, int col, int i, int j, int local_n, int m);
  
  MPI_Datatype col_mpi_t;
  MPI_Status status;  

  n = m*(local_n-2);
  h = 1/(double)(n+1);

  /* Define col_mpi_t datatype. */
  MPI_Type_vector(local_n, 1, local_n, MPI_DOUBLE, &col_mpi_t);
  MPI_Type_commit(&col_mpi_t);
  
  do 
  { /* Start iterations */
    num_iter ++;

    /* Set up the initial local_sum(Sum of 2_norm of (x_new-x_old)). */
    local_sum = 0;  
              
                                 
    for(i = 1; i < local_n-1; i++)
    {
      for(j = 1; j < local_n-1; j++)
      {
        /* Find new values. */
        	x_new[i][j] = (x_old[i-1][j] + x_old[i+1][j] + x_old[i][j-1] + x_old[i][j+1] + (h*h*func_f(row, col, i, j, local_n, n)) ) * 0.25;

      	/* Updates local_sum. */
        local_sum = local_sum + square_val(x_new[i][j]-x_old[i][j]);            

        if(i == 1 && j == 1)
          max = fabs(x_new[i][j]-x_old[i][j]);

        delta =  fabs(x_new[i][j]-x_old[i][j]);

        if (delta > max)
          max = delta;

      }
    }


    /* Send data to adjacent processors */
    if(row < m-1)
    { 
      /* up */
      MPI_Sendrecv(&x_new[local_n-2][0], local_n, MPI_DOUBLE, rank_of(row+1,col,m), tag1,&x_old[local_n-1][0], local_n, MPI_DOUBLE, rank_of(row+1,col,m), tag2, MPI_COMM_WORLD, &status);
    }

    if(row > 0)
    { 
      /* down */
      MPI_Send(&x_new[1][0], local_n, MPI_DOUBLE, 
	       rank_of(row-1,col,m), tag2, MPI_COMM_WORLD);
      MPI_Recv(&x_old[0][0], local_n, MPI_DOUBLE, 
	       rank_of(row-1,col,m), tag1, MPI_COMM_WORLD, &status);
    }

    if(col < m-1)
    { 
      /* right */
      MPI_Send(&x_new[0][local_n-2], 1, col_mpi_t, 
	       rank_of(row,col+1,m), tag1, MPI_COMM_WORLD);
      MPI_Recv(&x_old[0][local_n-1], 1, col_mpi_t,
	       rank_of(row,col+1,m), tag2, MPI_COMM_WORLD, &status);
    }

    if(col > 0)
    { 
      /* left */
      MPI_Send(&x_new[0][1], 1, col_mpi_t, 
	       rank_of(row,col-1,m), tag2, MPI_COMM_WORLD);
      MPI_Recv(&x_old[0][0], 1, col_mpi_t, 
	       rank_of(row,col-1,m), tag1, MPI_COMM_WORLD, &status);
    }

    /* Copy x_new into x_old and do iterations again. */
    for(i = 1; i < local_n-1; i++)
      for(j = 1; j < local_n-1; j++)
        x_old[i][j] = x_new[i][j];

     
    /* Reduce local_sum in each proc to sum and broadcast sum. */
    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    /*Norm calculation */
    diff_norm = sqrt(sum)/n;     
       
  } while(num_iter < max_iter && eps < global_delta); /* end of do-while */
 
  
  if(rank_of(row,col,m) == 0 && eps > diff_norm )
    printf("num_iter: %d with diff_norm: %f\n", num_iter, diff_norm);
  else if(rank_of(row,col,m) == 0 && eps <= diff_norm)
    printf("Jacobi iteration does NOT converges\n");

  MPI_Type_free(&col_mpi_t);
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

    fpt = fopen("JacobiSoln.txt","w");
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
