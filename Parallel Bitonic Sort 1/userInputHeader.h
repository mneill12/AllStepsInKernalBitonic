#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

struct blockAndThreadCounts{

	int* blockCounts;
	int* threadCounts;
	int elementCount;
	int combinationsCount;
};

int getBlockCount();
int getThreadCount();
blockAndThreadCounts getElementCounts();
int getMaxProcessCount();
bool isPowerOfTwo(int n);
bool runSortAgain();
blockAndThreadCounts getSuggestedThreadCounts(int elementcount);
