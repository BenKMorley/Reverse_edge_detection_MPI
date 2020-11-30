#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <stdbool.h>
#include <unistd.h>
#include "../c/pgmio.c"
#include "../c/arralloc.c"

#define tolerance 0.02

int find_jdim(int nj, int ni, int size);

void find_edges(int *coords, int jsize, int isize, bool *edges);

int main(int argc, char *argv[]){
  MPI_Init(NULL, NULL);
  int rank;
  int size;
  MPI_Comm cart_comm;
  MPI_Request request;
  MPI_Status status;

  // Use the first argument to take in the image name which is to be edited
  char *filename = argv[1];

  int nj, ni;

  // Construct the input filename
  char infile[100] = "./im/";
  strcat(infile, filename);

  // Find the size of the image in question
  pgmsize(infile, &ni, &nj);

  // Find the number of processes in the program
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Construct the output filename
  char outfile[100] = "./im/out_";
  char size_string[10];
  sprintf(size_string, "%d_", size);
  strcat(outfile, size_string);
  strcat(outfile, filename);
 
  // Create a Cartesian topology
  int periods_[2] = {0, 0};  //Non-periodic boundaries
  const int *periods = (const int *) periods_;

  // Find the dims most appropriate for the number of processes
  int jdim = find_jdim(nj, ni, size);
  int idim = size / jdim;


  // Find the x and y size of the subarrays which will be distributed to the
  // various processes
  int jsize = nj / jdim;
  int isize = ni / idim;

  if(rank == 0){
    printf("jdim = %d \n", jdim);
    printf("jsize = %d\n", jsize);
    printf("idim = %d \n", idim);
    printf("isize = %d\n", isize);
    printf("size = %d\n", size);
  }
    
  int dims_[2] = {idim, jdim};
  const int *dims = (const int *) dims_;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

  // Extract the rank of this process
  MPI_Comm_rank(cart_comm, &rank);

  // Extract the cartesian coordinates of this process in the communicator
  int coords[2];
  MPI_Cart_coords(cart_comm, rank, 2, coords);
  printf("Process %d here with coords : [%d, %d]\n", rank, coords[0], coords[1]);

  // Create a vector derived data-type to split up the image into
  // blocks of size (jsize, isize)
  MPI_Datatype block;
  MPI_Type_vector(isize, jsize, nj, MPI_DOUBLE, &block);
  MPI_Type_commit(&block);

  // Initialize the 4 arrays we will need for this algoritm
  double **old = arralloc(sizeof(double), 2, isize+2, jsize+2);
  double **new = arralloc(sizeof(double), 2, isize+2, jsize+2);

  // Give the edge array halos for clarity in calculation later
  double **edge = arralloc(sizeof(double), 2, isize+2, jsize+2);
  double **buf = arralloc(sizeof(double), 2, isize, jsize);

  double **masterbuf = arralloc(sizeof(double), 2, ni, nj);


  // Read in the image to the buf array only on rank 0
  if(rank == 0){

    pgmread(infile, &masterbuf[0][0], ni, nj);
  }

  // Initialise old, edge and new to white
  for(int i=0; i < isize + 2; i++){
    for(int j=0; j < jsize + 2; j++){
      old[i][j] = 255;
      edge[i][j] = 255;
      new[i][j] = 255;
    }
  }

  // Process 0 will distribute data to the other processes 
  if(rank == 0){
    int send_rank;

    // Loop over the cartesian coordinates
    for(int i = 0; i<idim; i++){
      for(int j = 0; j<jdim; j++){
        const int other_coords[2] = {i, j};

        // Find the rank of the process at this cart coordinate
        MPI_Cart_rank(cart_comm, other_coords, &send_rank);

        // Send a block of data by using derived datatypes
        MPI_Issend(&masterbuf[i*isize][j*jsize], 1, block, send_rank, 0, cart_comm, &request);
      }
    }
  }

  // Since the recieving array is of the correct size, data is contiguous
  MPI_Irecv(&buf[0][0], jsize * isize, MPI_DOUBLE, 0, 0, cart_comm, &request);

  MPI_Wait(&request, &status);
  MPI_Barrier(cart_comm);

  // if(rank == 0){
  //   pgmwrite("test_0.pgm", &buf[0][0], isize, jsize);  
  // }

  // if(rank == 1){
  //   pgmwrite("test_1.pgm", &buf[0][0], isize, jsize);  
  // }

  // Make a derived datatype for transfering the row part of the halos
  MPI_Datatype row;
  MPI_Type_vector(isize, 1, jsize + 2, MPI_DOUBLE, &row);
  MPI_Type_commit(&row);

  // Copy buf into edge, remembering the halos
  for(int i=1; i< isize + 1; i++){
    for(int j=1; j< jsize + 1; j++){
      edge[i][j] = buf[i-1][j-1];
    }
  }

  // Find the ranks of this processes neighbours
  int up, down, left, right;
  MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
  MPI_Cart_shift(cart_comm, 1, 1, &down, &up);
  printf("process %d has neighbours up %d, down %d, left %d, right %d\n", rank, up, down, left, right);

  // Determine if this process is on a edge/which edges is it on
  // find_edges will return true if buf is on the edge of the image
  // with the order of edges being up, down, left, right.
  bool edges[4] = {false, false, false, false};

  find_edges(coords, idim, jdim, edges);
  bool left_edge = edges[0];
  bool right_edge = edges[1];
  bool down_edge = edges[2];
  bool up_edge = edges[3];

  // Find number of requests needed
  int no_requests = 2 * ((int) !up_edge + (int) !down_edge + (int) !left_edge + (int) !right_edge);

  // Construct an array of requests
  MPI_Request requests[8];
  
  // Use a tolerance parameter to decide when enough iterations have been done
  double temp = 0;
  double temp2 = 0;
  double delta = 1;
  int iter = 0;
  int counter = 0;
  double total = 0;
  double total2 = 0;

  MPI_Barrier(cart_comm);

  // Find the time per iteration
  double start = MPI_Wtime();

  // Perform the algorithm to reverse the edge detection
  while(delta > tolerance){
    // Send and recieve halos from the neighbours
    // Use non-blocking communication for better performance
    counter = 0;

    // Send and recieve up
    if(!up_edge){
      // printf("Process %d sending/recieving up to process %d\n", rank, up);
      // Send up
      MPI_Issend(&old[1][jsize], 1, row, up, 0, cart_comm, &requests[counter]);

      // Recieve up
      MPI_Irecv(&old[1][jsize + 1], 1, row, up, 0, cart_comm, &requests[counter + 1]);
      
      counter += 2;
    }

    // Send and recieve down
    if(!down_edge){
      // printf("Process %d sending/recieving down to process %d\n", rank, down);

      MPI_Issend(&old[1][1], 1, row, down, 0, cart_comm, &requests[counter]);

      MPI_Irecv(&old[1][0], 1, row, down, 0, cart_comm, &requests[counter + 1]);

      counter += 2;
    }

    // Send and recieve left
    if(!left_edge){
      // printf("Process %d sending/recieving left to process %d\n", rank, left);

      MPI_Issend(&old[1][1], jsize, MPI_DOUBLE, left, 0, cart_comm, &requests[counter]);

      MPI_Irecv(&old[0][1], jsize, MPI_DOUBLE, left, 0, cart_comm, &requests[counter + 1]);

      counter += 2;
    }

    // Send and recieve right
    if(!right_edge){
      // printf("Process %d sending/recieving right to process %d\n", rank, right);

      MPI_Issend(&old[isize][1], jsize, MPI_DOUBLE, right, 0, cart_comm, &requests[counter]);

      MPI_Irecv(&old[isize + 1][1], jsize, MPI_DOUBLE, right, 0, cart_comm, &requests[counter + 1]);
    }

    // Wait for recieves and sends before proceeding
    MPI_Status statuses[no_requests];
    MPI_Waitall(no_requests, requests, statuses);

    // Make sure all processes are aligned before proceeding
    MPI_Barrier(cart_comm);

    // Perform an iteraction of the edge-detection reversal algorithm
    for(int i=1; i<isize+1; i++){
      for(int j=1; j<jsize+1; j++){
        new[i][j] = 0.25 * (old[i-1][j] + old[i+1][j] + old[i][j-1] + old[i][j+1] - edge[i][j]);
      }
    }

    // Write new into old ready for the next iteration, and calculate delta
    // Do this only every 100 iterations is if statements are expensive
    if(iter%100 == 0){
      total2 = 0;
      total = 0;
      temp = 0;
      temp2 = 0;
      for(int i=1; i<isize+1; i++){
        for(int j=1; j<jsize+1; j++){

          // Find the largest change to a pixel
          temp2 = fabs(old[i][j] - new[i][j]);
          if(temp2 > temp){
            temp = temp2;
          }

          // Add to total so the average can be found
          total2 += new[i][j];
        }
      }
      MPI_Barrier(cart_comm);

      // Calculate delta, the maximum change of any pixel in the image
      MPI_Allreduce(&temp, &delta, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

      // Calculate the average pixal size
      MPI_Allreduce(&total2, &total, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

      if(rank == 0){
        printf("Completed %d iterations\n", iter);
        printf("delta = %f\n", delta);
        printf("Average pixal value = %f\n", (total / (ni * nj)));
      }
    }

    // Set old equal to new
    for(int i=1; i<isize+1; i++){
      for(int j=1; j<jsize+1; j++){
        old[i][j] = new[i][j];
      }
    }

    iter ++;
  }
  
  double end = MPI_Wtime();

  // Copy the resultant image back into buf for writing out
  for(int i=1; i<isize+1; i++){
    for(int j=1; j<jsize+1; j++){
      buf[i-1][j-1] = new[i][j];
    }
  }

  // All processes will send data to process with rank 0
  MPI_Isend(&buf[0][0], isize * jsize, MPI_DOUBLE, 0, 0, cart_comm, &request);

  // Process 0 will collect data from the other processes 
  if(rank == 0){
    int send_rank;

    // Loop over the cartesian coordinates
    for(int i = 0; i<idim; i++){
      for(int j = 0; j<jdim; j++){
        const int other_coords[2] = {i, j};
        
        // Find the rank of the process at this cart coordinate
        MPI_Cart_rank(cart_comm, other_coords, &send_rank);

        // Recieve a block of data by using derived datatypes
        MPI_Recv(&masterbuf[i*isize][j*jsize], 1, block, send_rank, 0, cart_comm, &status);
      }
    }

    // Write out the image
    pgmwrite(outfile, &masterbuf[0][0], ni, nj);

    // Write out the time taken to results.csv
    FILE *results = fopen("results.csv", "a");

    fprintf(results, "%d,%d,%d,%.10f,%d,%d,%d,%d,%d\n", ni, nj, size, (end - start) * 1000/iter, iter, idim, jdim, isize, jsize);

    fclose(results);
  }

  MPI_Type_free(&row);
  MPI_Type_free(&block);
  
  free(edge);
  free(buf);
  free(masterbuf);
  free(new);
  free(old);

  MPI_Finalize();
}

int find_jdim(int nj, int ni, int size){
  // Determine a decomposition of the image that is as close to square as
  // possible
  int temp = 1;
  int jdim;
  int max;
  bool solution = false;

  while( (temp <= sqrt(size)) || !solution ){
    if(nj%temp == 0 & // The j-axis divides exactly
       size%temp == 0 & // temp divides the number of processes exactly
       ni%(size/temp) == 0 // The i-axis divides exactly
    ){
      solution = true;
      max = temp;
    }
    temp++;
  }
  return max;
}

void find_edges(int *coords, int idim, int jdim, bool *edges){
  if(coords[0] == 0){  //if at the left
    edges[0] = true;
  }

  if(coords[0] == idim - 1){ // if on the right
    edges[1] = true;
  }

  if(coords[1] == 0){ // if on the bottom
    edges[2] = true;
  }

  if(coords[1] == jdim - 1){ // if on top
    edges[3] = true;
  }
}
