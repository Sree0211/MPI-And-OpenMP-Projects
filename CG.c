/* Projecct*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"


#define eps 0.0001
#define max_iter 10000

#define square_calc(x,y) ((x)*(y))

int rank;

void show_matrix(double** a, int m_l, int n_l);
void show_vector(double* a, int m);
void set_stencils(double* stencils, int size);
void fill_values(double** values, double** prev_values, int m_l, int n_l);
double f(double x, double y);
double g(double x, double y);
double u(double x, double y);
void matrix_u(int xstart, int xend, int ystart, int yend, double h, double** uu);
void err_calc(double** values, int xstart, int xend, int ystart, int yend, double h, double* err);
void update_boundary(int p_start, int p_end, int left, int right, int top, int bottom, double h, double* ghost);
void read_b(double** b, int m_l, int n_l, int xstart, int ystart, double h);
MPI_Comm Grid_initialize();
double** array2d_create(int m, int n);
void update_ghosts(double** values, int xstart, int xend, int ystart, int yend, double* left_ghost, double* right_ghost, double* top_ghost, double* bottom_ghost, double* left_border, double* right_border, double* top_border, double* bottom_border, int m_l, int n_l, MPI_Comm grid_comm, double h);
void free_2d_array(double** temp, int m, int n);
void Grid_setup(int N, MPI_Comm grid_comm, int* x_start, int* x_end, int* y_start, int* y_end, int* m_l, int* n_l);
void CG_method(int N, MPI_Comm grid_comm, const int MAIN_PROCESSOR,double* u0);

int main(int argc, char** argv) {

    const int current_rank = 0;
    int p;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&p);
    int N = 16;
    
    MPI_Comm grid_comm = Grid_initialize();
    MPI_Comm_rank(grid_comm, &rank);

    if (rank == 0)
    {
        printf("\n");
        printf("  POISSON MPI:\n");
        printf("  2-D Poisson's equation using Conjugate gradient algorithm\n");
        printf("  *******************************************\n");
        printf("  Number of processes         = %d\n", p);
        printf("  Desired fractional accuracy = %f\n", eps);
        printf("\n");
    }
    
    double* u0 = 0;
    
    CG_method(N, grid_comm, current_rank,u0);


    MPI_Finalize();

}


void show_matrix(double** a, int m_l, int n_l) {
    int i, j;
    for (i = 0; i < m_l; ++i) {
        for (j = 0; j < n_l; ++j) {
            printf("%lf ", a[i][j]);
        }
        printf("\n");
    }
}

void show_vector(double* a, int m) {
    int i;
    for (i = 0; i < m; ++i) {
        printf("%lf ", a[i]);
    }
    printf("\n");
}

void set_stencils(double* stencils, int size) {
    int i;
    for (i = 1; i < size; ++i) {
        stencils[i] = -1.0;
    }
    stencils[0] = 4.0;
}

void fill_values(double** values, double** prev_values, int m_l, int n_l) {
    int i, j;
    for (i = 0; i < n_l; ++i) {
        for (j = 0; j < m_l; ++j) {
            prev_values[j][i] = values[j][i];
        }
    }
}

double f(double x, double y) {
    return 2.0 * ((1.0 + x) * sin(x + y) - cos(x + y));
}

double g(double x, double y) {
    return (1.0 + x) * sin(x + y);
}

double u(double x, double y) {
    return (1.0 + x) * sin(x + y);
}

void matrix_u(int xstart, int xend, int ystart, int yend, double h, double** uu) {
    int i, j;
    double x_global, y_global;
    for (i = xstart; i <= xend; ++i) {
        x_global = (i + 1) * h;
        for (j = ystart; j <= yend; ++j) {
            y_global = (j + 1) * h;
            uu[j - ystart][i - xstart] = u(x_global, y_global);
        }
    }
}

void err_calc(double** values, int xstart, int xend, int ystart, int yend, double h, double* err) {
    int i, j;
    double x_global, y_global, temp_err;
    double e_inf = 0.0;
    for (i = xstart; i <= xend; ++i) {
        x_global = (i + 1) * h;
        for (j = ystart; j <= yend; ++j) {
            y_global = (j + 1) * h;
            temp_err = fabs(values[j - ystart][i - xstart] - u(x_global, y_global));
            if (temp_err > e_inf) e_inf = temp_err;
        }
    }
    MPI_Allreduce(&e_inf, err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

void update_boundary(int p_start, int p_end, int left, int right, int top, int bottom, double h, double* ghost) {
    
    int i;
    double x_global, y_global;
   
    for (i = p_start + 1; i <= p_end + 1; ++i) {
        if (left || right) {
            y_global = i * h;
            if (left) {
                x_global = 0.0;
            }
            else {
                x_global = 1.0;
            }
        }
        else {
            x_global = i * h;
            if (top) {
                y_global = 0.0;
            }
            else {
                y_global = 1.0;
            }
        }
        ghost[i - p_start - 1] = g(x_global, y_global);
    }
}

void read_b(double** b, int m_l, int n_l, int xstart, int ystart, double h) {
    int i, j;
    double x, y;
    for (i = 0; i < n_l; ++i) {
        x = (xstart + i + 1) * h;
        for (j = 0; j < m_l; ++j) {
            y = (ystart + j + 1) * h;
            b[j][i] = f(x, y);
        }
    }
}


/* Generalizing the Setup of the grid just incase of rectangular matrices*/
// We are using mxm matrix

void set_m_n(int size, int* m, int* n) {
    
    int i, temp;
    for (i = 1; i < size; ++i) {
        temp = size / i;
        if (temp * i == size) {
            *m = i;
            *n = temp;
        }
    }
    if (size == 1) *m = *n = 1;
}

MPI_Comm Grid_initialize() 
{
    
    MPI_Comm grid_comm;
    int size, m;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    set_m_n(size, &m, &n);
    int dims[2];
    int period[2];
    dims[0] = m;
    dims[1] = m;
    period[0] = period[1] = 0;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 1, &grid_comm);

    return grid_comm;
}

double** array2d_create(int m, int n) 
{
    
    double** temp;
    temp = (double**)malloc(m * sizeof(double*));
    int i, j;
    for (i = 0; i < m; ++i) {
        temp[i] = (double*)malloc(n * sizeof(double));
    }
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            temp[i][j] = 0.0;
        }
    }
    return temp;
}

void free_2d_array(double** temp, int m, int n) 
{
    
    int i;
    for (i = 0; i < m; ++i) {
        free(temp[i]);
    }
    free(temp);

}

void Grid_setup(int N, MPI_Comm grid_comm, int* x_start, int* x_end, int* y_start, int* y_end, int* m_l, int* n_l) 
{
    
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int m, n;
    set_m_n(size, &m, &n);
    int coordinates[2];
    MPI_Cart_coords(grid_comm, rank, 2, coordinates);
    int y_step = N / m;
    int x_step = N / n;
    *x_start = coordinates[1] * x_step;
    *x_end = (coordinates[1] + 1) * x_step - 1;
    *y_start = coordinates[0] * y_step;
    *y_end = (coordinates[0] + 1) * y_step - 1;
    *m_l = *y_end - *y_start + 1;
    *n_l = *x_end - *x_start + 1;

}

void update_ghosts(double** values, int xstart, int xend, int ystart, int yend, double* left_ghost, double* right_ghost, double* top_ghost, double* bottom_ghost, double* left_border, double* right_border, double* top_border, double* bottom_border, int m_l, int n_l, MPI_Comm grid_comm, double h) 
{
    
    MPI_Status status;
    int i, left_index, right_index, top_index, bottom_index, size, m, n, my_x, my_y, my_index;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    set_m_n(size, &m, &n);
    my_x = rank % n;
    my_y = rank / n;
   
    my_index = my_y * n + my_x;
    left_index = (my_x == 0) ? (my_y + 1) * n - 1 : my_y * n + my_x - 1;
    right_index = (my_x == n - 1) ? my_y * n : my_y * n + my_x + 1;
    top_index = (my_y == 0) ? (m - 1) * n + my_x : (my_y - 1) * n + my_x;
    bottom_index = (my_y == m - 1) ? my_x : (my_y + 1) * n + my_x;
    
    for (i = 0; i < m_l; ++i) {
        left_border[i] = values[i][0];
        right_border[i] = values[i][n_l - 1];
    }
    for (i = 0; i < n_l; ++i) {
        top_border[i] = values[0][i];
        bottom_border[i] = values[m_l - 1][i];
    }

    // Transferring data left
    MPI_Sendrecv(left_border, m_l, MPI_DOUBLE, left_index, 0, right_ghost, m_l, MPI_DOUBLE, right_index, 0, grid_comm, &status);
    if (my_x == n - 1) {
        update_boundary(ystart, yend, 0, 1, 0, 0, h, right_ghost);
    }
    MPI_Barrier(grid_comm);

    // Transferring data right
    MPI_Sendrecv(right_border, m_l, MPI_DOUBLE, right_index, 0, left_ghost, m_l, MPI_DOUBLE, left_index, 0, grid_comm, &status);
    if (my_x == 0) {
        update_boundary(ystart, yend, 1, 0, 0, 0, h, left_ghost);
    }
    MPI_Barrier(grid_comm);

    // Transferring data top
    MPI_Sendrecv(top_border, n_l, MPI_DOUBLE, top_index, 0, bottom_ghost, n_l, MPI_DOUBLE, bottom_index, 0, grid_comm, &status);
    if (my_y == m - 1) {
        update_boundary(xstart, xend, 0, 0, 0, 1, h, bottom_ghost);
    }
    MPI_Barrier(grid_comm);

    // Transferring data bottom
    MPI_Sendrecv(bottom_border, n_l, MPI_DOUBLE, bottom_index, 0, top_ghost, n_l, MPI_DOUBLE, top_index, 0, grid_comm, &status);
    if (my_y == 0) {
        update_boundary(xstart, xend, 0, 0, 1, 0, h, top_ghost);
    }
}


void CG_method(int N, MPI_Comm grid_comm, const int MAIN_PROCESSOR, double* u_0)
{

    int* x_offsets;
    int* y_offsets;
    double* stencils;
    int size, m_l, i, j, l, p;

    MPI_Comm_size(MPI_COMM_WORLD, &p);

    size = 5;

    x_offsets = (int*)malloc(size * sizeof(int));
    y_offsets = (int*)malloc(size * sizeof(int));
    stencils = (double*)malloc(size * sizeof(double));
    set_stencils(stencils, size);

    x_offsets[0] = 0; y_offsets[0] = 0;
    x_offsets[1] = 0; y_offsets[1] = -1;
    x_offsets[2] = 0; y_offsets[2] = 1;
    x_offsets[3] = -1; y_offsets[3] = 0;
    x_offsets[4] = 1; y_offsets[4] = 0;

    int x_start, x_end, y_start, y_end;
    double h;

    h = 1.0 / (double)(N + 1);
    
    MPI_Status status;

    double err = 1.0;
    double** values;
    double** u;
    double** prev_values;
    double** b;
    double* top_ghost, * bottom_ghost, * right_ghost, * left_ghost;
    double* top_border, * bottom_border, * right_border, * left_border;
    double* values_l;

    Grid_setup(N, grid_comm, &x_start, &x_end, &y_start, &y_end, &m_l, &m_l);
    values = array2d_create(m_l, m_l);
    prev_values = array2d_create(m_l, m_l);
    b = array2d_create(m_l, m_l);
    u = array2d_create(m_l, m_l);

    read_b(b, m_l, m_l, x_start, y_start, h);

    top_border = (double*)malloc(m_l * sizeof(double));
    bottom_border = (double*)malloc(m_l * sizeof(double));
    left_border = (double*)malloc(m_l * sizeof(double));
    right_border = (double*)malloc(m_l * sizeof(double));
    top_ghost = (double*)malloc(m_l * sizeof(double));
    bottom_ghost = (double*)malloc(m_l * sizeof(double));
    left_ghost = (double*)malloc(m_l * sizeof(double));
    right_ghost = (double*)malloc(m_l * sizeof(double));
    values_l = (double*)malloc(size * sizeof(double));

    double sum, max_err, temp_err, t;

    double* r0, *r, *p0, *p;
    double** s;
    double beta, rho,*rho_new;
    double rho_sum = 0;
    int k=0; /*Iteration for the CG method*/
    double* pTa_local, *pTa;

    /* Define r0*/
    /* r0 = b - Au*/
    r0 = (double*)malloc(m_l * sizeof(double));
    p0 = (double*)malloc(m_l * sizeof(double));
    pTa_local = (double*)malloc(m_l * sizeof(double));
    pTa = (double*)malloc(m_l * sizeof(double));
    r = (double*)malloc(m_l * sizeof(double));
    p = (double*)malloc(m_l * sizeof(double));
    rho_new = (double*)malloc(m_l * sizeof(double));


    p0 = r0;

    for (i = 0; i < sizeof(r); i++)
    {
        rho_sum += square_calc(r[i],r[i]);
    }
    
    MPI_Allreduce(&rho_sum, &rho, 1, MPI_DOUBLE, MPI_MAX, grid_comm);

    MPI_Barrier(grid_comm);

    if (rank == MAIN_PROCESSOR) {
        t = MPI_Wtime();
    }

        while (rho > eps && k < max_iter)
        {
            max_err = 0.0;
            MPI_Sendrecv(&p, m_l, MPI_DOUBLE, ((x_end + 1) * m_l + y_end), tag1, prev_values[m_l - 1][0], m_l, MPI_DOUBLE, ((x_end + 1) * m_l + y_end), tag2, MPI_COMM_WORLD, &status); 
            update_ghosts(prev_values, x_start, x_end, y_start, y_end, left_ghost, right_ghost, top_ghost, bottom_ghost, left_border, right_border, top_border, bottom_border, m_l, m_l, grid_comm, h);
            for (i = 0; i < m_l; ++i) {
                for (j = 0; j < m_l; ++j) {
                    sum = 0.0;
                    for (l = 1; l < size; ++l) {
                        if (x_offsets[l] == 1) {
                            values_l[l] = ((i == m_l - 1) ? right_ghost[j] : prev_values[j][1 + i]);
                        }
                        if (x_offsets[l] == -1) {
                            values_l[l] = ((i == 0) ? left_ghost[j] : prev_values[j][-1 + i]);
                        }
                        if (y_offsets[l] == 1) {
                            values_l[l] = ((j == m_l - 1) ? bottom_ghost[i] : prev_values[j + 1][i]);
                        }
                        if (y_offsets[l] == -1) {
                            values_l[l] = ((j == 0) ? top_ghost[i] : prev_values[j - 1][i]);
                        }
                        sum -= values_l[l] * (double)stencils[l];
                    }
                    
                    s[i][j] = sum;
                    for (int te = 0; te < sizeof(s); te++)
                    {
                        rho_sum += square_calc(p,s);
                    }
                    pTa_local = rho_sum;
                    MPI_Allreduce(&pTa_local, &pTa, 1, MPI_DOUBLE, MPI_MAX, grid_comm);
                    beta = rho / pTa;
                    values[i][j] += beta * p;
                    r -= beta * s;
                    for (int te = 0; te < sizeof(s); te++)
                    {
                        rho_sum += square_calc(r[i], r[i]);
                    }
                    MPI_Allreduce(&rho_sum, &rho_new, 1, MPI_DOUBLE, MPI_MAX, grid_comm);
                    values[i][j] = (rho_new / rho) * values[i][j] + r[i][j];
                    
                    k = k + 1;

                    temp_err = fabs(values[j][i] - prev_values[j][i]);
                    if (max_err < temp_err) max_err = temp_err;

                }
            }
            MPI_Allreduce(&max_err, &err, 1, MPI_DOUBLE, MPI_MAX, grid_comm);
            fill_values(values, prev_values, m_l, m_l);
        }

    MPI_Barrier(grid_comm);
    if (rank == MAIN_PROCESSOR) {
        t = MPI_Wtime() - t;
        printf(" For %d nodes and %d processors CG Method takes %lf seconds\n", N, p, t);
    }

    double e_inf;

    err_calc(values, x_start, x_end, y_start, y_end, h, &e_inf);
    if (rank == MAIN_PROCESSOR) {
        printf("||e||_inf = %.10lf\n", e_inf);
    }

    free(right_ghost);
    free(left_ghost);
    free(top_ghost);
    free(bottom_ghost);
    free(right_border);
    free(left_border);
    free(top_border);
    free(bottom_border);
    free(values_l);
    free(x_offsets);
    free(y_offsets);
    free(stencils);
    free_2d_array(values, m_l, m_l);
    free_2d_array(prev_values, m_l, m_l);
    free_2d_array(b, m_l, m_l);
}

