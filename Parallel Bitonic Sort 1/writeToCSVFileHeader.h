#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

//Function Prototypes 
void writeSortResultsToCsv(double* timeResults, char* sortType, char* arrayStates, int* threadCounts, int* deviceBlocks, int executionCount);
void writeArrayAsCsvFile(char* filename, char* arrayState, int* array, int arrayLength);
void writeBlockElementCsvFile(int* values, char* arrayState, int threadCount, int deviceBlocks);
void writeSuggestedBlockThreadConfigToCsv(int* suggestedThreads, int* suggestedBlocks, int combinationsCount);


void incrementFileId(char* fileDirAndName);
int fileExists(const char *fileName);