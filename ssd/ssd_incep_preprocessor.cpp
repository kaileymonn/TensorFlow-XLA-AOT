#include <cstdio>
#include <algorithm>
#include <assert.h>
#include "out.h"

int main() {
    Preprocessor computation;

    int32_t args[1][300][300][3];
    for (int i = 0; i<300; i++) {
    	for (int j = 0; j<300; j++) {
	    for (int k = 0; k<3; k++) {
		args[0][i][j][k] = i+j+k*4;		
	    }
	}
    }
    computation.set_arg0_data(&args);
    if(!computation.Run()) {
   	puts(computation.error_msg().c_str());
	return 1;
    } 
  
    assert(computation.result0(0,10,200,1) == 211.0);
    assert(computation.result0(0,250,50,2) == 302.0);
    //printf("%f\n", computation.result0(0,10,200,1));
    //printf("%f\n", computation.result0(0,250,50,2));

    printf("Size of input: %zu\n", sizeof(args)/sizeof(args[0][0][0][0]));
    printf("Size of fetch0: %zu\n", sizeof(*computation.result0_data()));
    for(size_t i = 0; i<sizeof(*computation.result0_data()); i++) {
	printf("data: %f\n", computation.result0_data()[i]);
    }

    //printf(computation.result0_data())

    for(size_t i = 0; i < 3; i++) {
        //printf("%d\n", computation.result1(0,i));
    }

    return 0;
}
