#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cuda_runtime_api.h>

#include "writeToCSVFileHeader.h"
#include "userInputHeader.h"

void printArray(int *elements);


int deviceBlocks;
int threadsPerBlock;
int elementsToSort;
int threadCount;

int phases;

//Max times we cann run the process
int executionCount;


const int randMax = 10000;

void createUnsortedArray(int* elements){

	for (int i = 0; i < elementsToSort; ++i){
		elements[i] = rand() % randMax - rand() % 5;
	}

}

bool isSorted(int *elements){

	bool sorted = true;
	for (int i = 0; i < (elementsToSort - 1); ++i){
		if (elements[i] > elements[i + 1]){
			sorted = false;
		}
	}
	return sorted;
}


double getElapsedTime(clock_t start, clock_t stop)
{
	double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
	printf("Elapsed time: %.3fs\n", elapsed);

	return elapsed;
}

int random_int()
{
	return (int)rand() / (int)2048;
}


__global__ void bitonicSort(int* deviceElements, int subSequenceSize, int steps){


	//1printf("Kernal Called!!!!");
	/*
	Here we get our first thread var i and j.
	we get j by knowing the size of the subsequence and then halfing it, this gives us the rang that values should be comapired for this step.
	As we go down the steps, we'll be halfing j until step = 1;
	*/

	int firstIndex = threadIdx.x + blockDim.x * blockIdx.x;
	int rangeOfComparison = (subSequenceSize / 2);
	for (int step = steps; step >= 1; step--){

		//This xor op checks that our second value is bigger than our firstIndex value
		if ((firstIndex ^ rangeOfComparison) > firstIndex){

			//assending
			if ((firstIndex / subSequenceSize) % 2 == 0){

				if (deviceElements[firstIndex] > deviceElements[firstIndex ^ rangeOfComparison]) {
					printf("Even element assending %d: %d -> %d \n", firstIndex, deviceElements[firstIndex], deviceElements[firstIndex ^ rangeOfComparison]);
					int temp = deviceElements[firstIndex];
					deviceElements[firstIndex] = deviceElements[firstIndex ^ rangeOfComparison];
					deviceElements[firstIndex ^ rangeOfComparison] = temp;
				}

			}
			else{

				if (deviceElements[firstIndex] < deviceElements[firstIndex ^ rangeOfComparison]) {
					printf("Even element desending %d : %d -> %d \n", firstIndex, deviceElements[firstIndex], deviceElements[firstIndex ^ rangeOfComparison]);
					int temp = deviceElements[firstIndex];
					deviceElements[firstIndex] = deviceElements[firstIndex ^ rangeOfComparison];
					deviceElements[firstIndex ^ rangeOfComparison] = temp;
				}

			}

		}
		__syncthreads();

		rangeOfComparison = rangeOfComparison / 2;
	}
}
/*
Main function call. Created array and calls stepskernel based of the size of the bitonic sequences and step.
*/
void bitonic_Sort(int* elements){

	int* d_elements;

	//get "phases" so we know how many times we need to send array over to device  
	phases = int(log2(double(elementsToSort)));

	//General cuda managment here : Allocate on device, array isn't going to change  in size
	cudaMalloc(&d_elements, elementsToSort*sizeof(int));
	cudaMemcpy(d_elements, elements, elementsToSort*sizeof(int), cudaMemcpyHostToDevice);
	dim3 blocks(deviceBlocks, 1);    /* Number of blocks   */
	dim3 threads(threadsPerBlock, 1);  /* Number of threads  */

	for (int currentPhase = 1; currentPhase <= phases; currentPhase++){

		//Get the  size of each sub sequence and the amount of "Steps" in the individual sub sequences 
		int subSequenceSize = int(pow(double(2), double(currentPhase)));

		int steps = int(log2((double)subSequenceSize));

		cudaDeviceSynchronize();
		printf("Phase: %d \n", currentPhase);
		cudaDeviceSynchronize();
		bitonicSort << <blocks, threads>> >(d_elements, subSequenceSize, steps);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(elements, d_elements, elementsToSort*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_elements);
}


void preExecution(){

	int values[7];
	values[0] = 10;
	values[1] = 13;
	values[2] = 9;
	values[3] = 18;
	values[4] = 26;
	values[4] = 100;
	values[6] = 3;

	bitonic_Sort(values);
}

int main(void)
{
	executionCount = getMaxProcessCount();
	int fixedExecutionCount = executionCount;

	preExecution();

	bool runSort = true;

	//Pointers to store our results that we're writing to CSV files, allocate space entered buy the user
	int* threadCounts = (int*)malloc(executionCount*sizeof(int));
	int* allBlocks = (int*)malloc(executionCount*sizeof(int));;
	double* timeResults = (double*)malloc(executionCount*sizeof(double));;
	char* arrayStates = (char*)malloc(executionCount*sizeof(char));

	double time;
	clock_t start, stop;
	//Counter so we can assine values to the array in the execution loop

	while (runSort && executionCount != 0){

		runSort = runSortAgain();

		//Get thread, blocks and  element count

		//Get total elements and suggested block thread configurations
		blockAndThreadCounts inputCountandSuggestedThreadBlockCount;
		inputCountandSuggestedThreadBlockCount = getElementCounts();
		elementsToSort = inputCountandSuggestedThreadBlockCount.elementCount;

		//wirte possible thread and block configurations to text file
		printf("Writing suggested block thread configuration...");
		writeSuggestedBlockThreadConfigToCsv(inputCountandSuggestedThreadBlockCount.threadCounts,
			inputCountandSuggestedThreadBlockCount.blockCounts,
			inputCountandSuggestedThreadBlockCount.combinationsCount
			);
		printf("Done \n");
		//elementsToSort = inputCountandSuggestedThreadBlockCount.elementCount;
		deviceBlocks = getBlockCount();
		threadsPerBlock = getThreadCount();

		threadCount = threadsPerBlock * deviceBlocks;

		//Malloc array, add values to it and write unsorted array to csv file
		int* values = (int*)malloc(elementsToSort*sizeof(int));
		createUnsortedArray(values);
		writeBlockElementCsvFile(values, "preSorted", threadCount, deviceBlocks);

		//Do Sort and time it
		start = clock();
		bitonic_Sort(values);
		stop = clock();

		time = getElapsedTime(start, stop);

		char* arrayState;
		char arrayStateChar;

		if (isSorted(values)){

			printf("Is Sorted \n");
			arrayState = "sorted";
			arrayStateChar = 's';
		}
		else{

			printf("Not Sorted \n");
			arrayState = "unsorted";
			arrayStateChar = 'u';
		}

		writeBlockElementCsvFile(values, arrayState, threadCount, deviceBlocks);

		//Allocate results values to pointers 
		*threadCounts = threadCount;
		*allBlocks = deviceBlocks;
		*timeResults = time;
		*arrayStates = arrayStateChar;

		//Increment Result pointers
		threadCounts++;
		allBlocks++;
		timeResults++;
		arrayStates++;

		free(values);

		//Check again for user input

		executionCount--;
	}

	printf("Execution ended. Writing results to C:\BitonicSortArrayCSVFiles /n");

	writeSortResultsToCsv(timeResults, "ParallelBitonicSort", arrayStates, threadCounts, allBlocks, fixedExecutionCount);

	getchar();
}
