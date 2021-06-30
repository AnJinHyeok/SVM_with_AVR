#include <avr/io.h>
#include <string.h>
#include <stdlib.h>
#include <util/delay.h>



#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>



int seg_t[4] = {0, 0, 0, 0};



/**

/**
* Compute kernel between feature vector and support vector.
* Kernel type: linear
*/
float compute_kernel(int args, ...) {
    va_list w;
    va_start(w, args);
    float kernel = 0.0;

    int *x = va_arg(w, int*);

    for (int i = 0; i < 4; i++) {
        kernel += x[i] * va_arg(w, double);
    }

    return kernel;
}
                    /**
                    * Predict class for features vector
                    */
int predict(float *x) {
    float kernels[179] = { 0 };
    float decisions[6] = { 0 };
    int votes[4] = { 0 };
    kernels[0] = compute_kernel(5, x,   6.0  , 20.4  , 29.6  , 70.4 );
    kernels[1] = compute_kernel(5, x,   6.0  , 21.5  , 30.1  , 70.5 );
    kernels[2] = compute_kernel(5, x,   7.0  , 18.1  , 24.3  , 79.3 );
    kernels[3] = compute_kernel(5, x,   7.0  , 20.4  , 29.8  , 74.5 );
    kernels[4] = compute_kernel(5, x,   7.0  , 19.9  , 27.6  , 68.6 );
    kernels[5] = compute_kernel(5, x,   7.0  , 22.1  , 28.1  , 75.5 );
    kernels[6] = compute_kernel(5, x,   7.0  , 21.3  , 30.4  , 71.9 );
    kernels[7] = compute_kernel(5, x,   7.0  , 22.8  , 31.0  , 71.6 );
    kernels[8] = compute_kernel(5, x,   7.0  , 21.4  , 30.5  , 77.0 );
    kernels[9] = compute_kernel(5, x,   7.0  , 22.3  , 29.0  , 69.3 );
    kernels[10] = compute_kernel(5, x,   7.0  , 21.6  , 29.6  , 69.3 );
    kernels[11] = compute_kernel(5, x,   8.0  , 22.6  , 26.1  , 85.3 );
    kernels[12] = compute_kernel(5, x,   8.0  , 23.5  , 31.3  , 83.9 );
    kernels[13] = compute_kernel(5, x,   8.0  , 26.3  , 30.3  , 82.4 );
    kernels[14] = compute_kernel(5, x,   8.0  , 25.3  , 30.8  , 82.6 );
    kernels[15] = compute_kernel(5, x,   8.0  , 26.1  , 32.2  , 77.9 );
    kernels[16] = compute_kernel(5, x,   8.0  , 24.0  , 31.9  , 76.3 );
    kernels[17] = compute_kernel(5, x,   8.0  , 23.6  , 31.4  , 76.8 );
    kernels[18] = compute_kernel(5, x,   8.0  , 25.0  , 31.7  , 76.0 );
    kernels[19] = compute_kernel(5, x,   8.0  , 21.5  , 29.9  , 78.4 );
    kernels[20] = compute_kernel(5, x,   8.0  , 23.7  , 32.0  , 71.6 );
    kernels[21] = compute_kernel(5, x,   8.0  , 23.8  , 30.0  , 77.9 );
    kernels[22] = compute_kernel(5, x,   9.0  , 23.8  , 29.5  , 72.4 );
    kernels[23] = compute_kernel(5, x,   9.0  , 20.6  , 26.5  , 71.5 );
    kernels[24] = compute_kernel(5, x,   9.0  , 17.6  , 24.9  , 79.3 );
    kernels[25] = compute_kernel(5, x,   9.0  , 18.9  , 27.7  , 76.4 );
    kernels[26] = compute_kernel(5, x,   9.0  , 20.1  , 23.6  , 70.3 );
    kernels[27] = compute_kernel(5, x,   10.0  , 15.6  , 23.6  , 74.9 );
    kernels[28] = compute_kernel(5, x,   10.0  , 16.3  , 21.9  , 79.6 );
    kernels[29] = compute_kernel(5, x,   10.0  , 18.3  , 23.2  , 72.5 );
    kernels[30] = compute_kernel(5, x,   10.0  , 8.6  , 15.4  , 64.3 );
    kernels[31] = compute_kernel(5, x,   10.0  , 11.8  , 17.3  , 69.4 );
    kernels[32] = compute_kernel(5, x,   10.0  , 9.0  , 19.4  , 74.1 );
    kernels[33] = compute_kernel(5, x,   11.0  , 8.5  , 17.6  , 71.5 );
    kernels[34] = compute_kernel(5, x,   11.0  , 8.2  , 14.9  , 75.0 );
    kernels[35] = compute_kernel(5, x,   11.0  , 0.3  , 7.0  , 74.3 );
    kernels[36] = compute_kernel(5, x,   11.0  , -3.6  , 2.6  , 59.6 );
    kernels[37] = compute_kernel(5, x,   12.0  , 1.2  , 9.1  , 66.9 );
    kernels[38] = compute_kernel(5, x,   12.0  , 3.2  , 9.3  , 69.9 );
    kernels[39] = compute_kernel(5, x,   12.0  , 2.6  , 9.4  , 77.1 );
    kernels[40] = compute_kernel(5, x,   12.0  , -0.4  , 8.5  , 76.6 );
    kernels[41] = compute_kernel(5, x,   1.0  , -9.8  , 1.6  , 64.0 );
    kernels[42] = compute_kernel(5, x,   1.0  , -8.4  , 0.3  , 51.4 );
    kernels[43] = compute_kernel(5, x,   1.0  , -9.9  , -2.1  , 52.8 );
    kernels[44] = compute_kernel(5, x,   1.0  , -16.5  , -8.4  , 49.9 );
    kernels[45] = compute_kernel(5, x,   1.0  , -12.8  , -2.7  , 54.4 );
    kernels[46] = compute_kernel(5, x,   1.0  , -9.6  , -4.0  , 62.1 );
    kernels[47] = compute_kernel(5, x,   1.0  , -3.0  , 8.3  , 80.3 );
    kernels[48] = compute_kernel(5, x,   1.0  , -8.7  , -0.7  , 59.3 );
    kernels[49] = compute_kernel(5, x,   1.0  , -10.8  , -1.3  , 58.5 );
    kernels[50] = compute_kernel(5, x,   1.0  , 4.8  , 12.2  , 66.0 );
    kernels[51] = compute_kernel(5, x,   1.0  , 2.4  , 13.9  , 63.9 );
    kernels[52] = compute_kernel(5, x,   1.0  , 3.0  , 13.9  , 62.0 );
    kernels[53] = compute_kernel(5, x,   2.0  , -4.4  , 6.7  , 71.8 );
    kernels[54] = compute_kernel(5, x,   2.0  , 4.3  , 9.5  , 80.4 );
    kernels[55] = compute_kernel(5, x,   2.0  , -2.6  , 7.7  , 67.0 );
    kernels[56] = compute_kernel(5, x,   2.0  , -1.5  , 7.2  , 75.1 );
    kernels[57] = compute_kernel(5, x,   2.0  , 1.3  , 10.9  , 71.5 );
    kernels[58] = compute_kernel(5, x,   2.0  , -10.4  , -5.2  , 44.1 );
    kernels[59] = compute_kernel(5, x,   2.0  , 5.5  , 14.9  , 60.4 );
    kernels[60] = compute_kernel(5, x,   3.0  , -1.2  , 9.9  , 64.4 );
    kernels[61] = compute_kernel(5, x,   3.0  , 3.0  , 16.2  , 70.3 );
    kernels[62] = compute_kernel(5, x,   3.0  , 3.2  , 11.6  , 69.5 );
    kernels[63] = compute_kernel(5, x,   3.0  , 2.1  , 12.2  , 62.9 );
    kernels[64] = compute_kernel(5, x,   3.0  , 7.2  , 14.9  , 57.0 );
    kernels[65] = compute_kernel(5, x,   3.0  , 4.6  , 14.2  , 62.8 );
    kernels[66] = compute_kernel(5, x,   3.0  , 4.9  , 14.7  , 71.4 );
    kernels[67] = compute_kernel(5, x,   4.0  , 5.6  , 15.3  , 64.3 );
    kernels[68] = compute_kernel(5, x,   5.0  , 11.7  , 14.6  , 67.9 );
    kernels[69] = compute_kernel(5, x,   6.0  , 17.0  , 25.7  , 68.1 );
    kernels[70] = compute_kernel(5, x,   6.0  , 19.4  , 24.0  , 71.5 );
    kernels[71] = compute_kernel(5, x,   6.0  , 20.9  , 29.7  , 72.6 );
    kernels[72] = compute_kernel(5, x,   6.0  , 19.6  , 25.0  , 71.4 );
    kernels[73] = compute_kernel(5, x,   6.0  , 17.5  , 27.4  , 74.0 );
    kernels[74] = compute_kernel(5, x,   6.0  , 20.1  , 23.4  , 91.0 );
    kernels[75] = compute_kernel(5, x,   6.0  , 19.2  , 25.7  , 82.1 );
    kernels[76] = compute_kernel(5, x,   6.0  , 20.6  , 26.5  , 77.0 );
    kernels[77] = compute_kernel(5, x,   7.0  , 21.6  , 30.6  , 63.5 );
    kernels[78] = compute_kernel(5, x,   7.0  , 20.4  , 29.5  , 63.9 );
    kernels[79] = compute_kernel(5, x,   7.0  , 22.7  , 27.7  , 75.0 );
    kernels[80] = compute_kernel(5, x,   7.0  , 21.7  , 26.0  , 71.9 );
    kernels[81] = compute_kernel(5, x,   7.0  , 17.2  , 24.3  , 79.4 );
    kernels[82] = compute_kernel(5, x,   7.0  , 18.7  , 26.8  , 74.3 );
    kernels[83] = compute_kernel(5, x,   7.0  , 23.4  , 28.4  , 85.5 );
    kernels[84] = compute_kernel(5, x,   7.0  , 22.8  , 29.6  , 81.9 );
    kernels[85] = compute_kernel(5, x,   8.0  , 25.0  , 29.4  , 89.3 );
    kernels[86] = compute_kernel(5, x,   8.0  , 25.5  , 29.1  , 92.1 );
    kernels[87] = compute_kernel(5, x,   8.0  , 22.7  , 26.1  , 89.5 );
    kernels[88] = compute_kernel(5, x,   8.0  , 25.2  , 29.2  , 83.1 );
    kernels[89] = compute_kernel(5, x,   8.0  , 27.0  , 34.5  , 68.3 );
    kernels[90] = compute_kernel(5, x,   8.0  , 25.6  , 28.6  , 84.6 );
    kernels[91] = compute_kernel(5, x,   8.0  , 26.0  , 29.7  , 89.1 );
    kernels[92] = compute_kernel(5, x,   8.0  , 25.5  , 29.0  , 90.1 );
    kernels[93] = compute_kernel(5, x,   9.0  , 19.5  , 24.4  , 80.0 );
    kernels[94] = compute_kernel(5, x,   9.0  , 19.0  , 25.3  , 83.4 );
    kernels[95] = compute_kernel(5, x,   9.0  , 18.8  , 27.0  , 75.8 );
    kernels[96] = compute_kernel(5, x,   9.0  , 19.2  , 27.1  , 72.6 );
    kernels[97] = compute_kernel(5, x,   9.0  , 18.1  , 23.2  , 83.8 );
    kernels[98] = compute_kernel(5, x,   9.0  , 21.3  , 24.9  , 79.9 );
    kernels[99] = compute_kernel(5, x,   9.0  , 17.6  , 26.2  , 59.3 );
    kernels[100] = compute_kernel(5, x,   11.0  , 11.6  , 15.0  , 84.3 );
    kernels[101] = compute_kernel(5, x,   11.0  , 7.9  , 15.6  , 58.3 );
    kernels[102] = compute_kernel(5, x,   11.0  , 2.7  , 8.6  , 47.5 );
    kernels[103] = compute_kernel(5, x,   11.0  , 12.2  , 18.1  , 68.8 );
    kernels[104] = compute_kernel(5, x,   11.0  , 11.7  , 20.0  , 67.5 );
    kernels[105] = compute_kernel(5, x,   11.0  , 14.8  , 18.7  , 91.4 );
    kernels[106] = compute_kernel(5, x,   11.0  , -0.3  , 12.8  , 67.9 );
    kernels[107] = compute_kernel(5, x,   11.0  , 1.5  , 9.1  , 80.5 );
    kernels[108] = compute_kernel(5, x,   12.0  , -6.0  , 2.0  , 74.9 );
    kernels[109] = compute_kernel(5, x,   12.0  , -6.4  , 1.8  , 57.0 );
    kernels[110] = compute_kernel(5, x,   12.0  , 4.2  , 11.4  , 72.1 );
    kernels[111] = compute_kernel(5, x,   12.0  , -6.2  , 4.3  , 70.8 );
    kernels[112] = compute_kernel(5, x,   1.0  , -12.0  , -1.9  , 54.6 );
    kernels[113] = compute_kernel(5, x,   1.0  , -9.3  , -0.5  , 79.5 );
    kernels[114] = compute_kernel(5, x,   1.0  , -0.7  , 9.9  , 67.4 );
    kernels[115] = compute_kernel(5, x,   1.0  , -9.1  , 0.2  , 64.3 );
    kernels[116] = compute_kernel(5, x,   1.0  , 2.5  , 7.0  , 64.4 );
    kernels[117] = compute_kernel(5, x,   1.0  , -9.7  , 1.4  , 52.3 );
    kernels[118] = compute_kernel(5, x,   1.0  , -8.1  , 5.2  , 60.0 );
    kernels[119] = compute_kernel(5, x,   2.0  , -2.3  , 9.5  , 74.5 );
    kernels[120] = compute_kernel(5, x,   2.0  , -9.4  , 1.4  , 64.6 );
    kernels[121] = compute_kernel(5, x,   2.0  , -7.4  , 0.7  , 50.1 );
    kernels[122] = compute_kernel(5, x,   2.0  , -5.2  , 8.2  , 65.1 );
    kernels[123] = compute_kernel(5, x,   2.0  , -7.0  , -1.3  , 57.9 );
    kernels[124] = compute_kernel(5, x,   3.0  , 0.2  , 6.9  , 75.9 );
    kernels[125] = compute_kernel(5, x,   3.0  , 5.1  , 15.7  , 68.8 );
    kernels[126] = compute_kernel(5, x,   3.0  , 4.9  , 13.1  , 52.5 );
    kernels[127] = compute_kernel(5, x,   3.0  , 7.4  , 14.7  , 74.6 );
    kernels[128] = compute_kernel(5, x,   3.0  , 3.1  , 8.6  , 65.9 );
    kernels[129] = compute_kernel(5, x,   3.0  , 7.7  , 13.0  , 95.8 );
    kernels[130] = compute_kernel(5, x,   3.0  , 7.3  , 14.6  , 62.3 );
    kernels[131] = compute_kernel(5, x,   4.0  , 9.0  , 15.8  , 65.8 );
    kernels[132] = compute_kernel(5, x,   4.0  , 5.7  , 13.5  , 74.0 );
    kernels[133] = compute_kernel(5, x,   4.0  , 8.1  , 14.8  , 60.0 );
    kernels[134] = compute_kernel(5, x,   4.0  , 12.5  , 20.0  , 62.4 );
    kernels[135] = compute_kernel(5, x,   4.0  , 11.2  , 18.2  , 64.1 );
    kernels[136] = compute_kernel(5, x,   4.0  , 10.9  , 20.1  , 54.1 );
    kernels[137] = compute_kernel(5, x,   5.0  , 8.5  , 15.3  , 80.8 );
    kernels[138] = compute_kernel(5, x,   5.0  , 7.3  , 18.7  , 67.9 );
    kernels[139] = compute_kernel(5, x,   5.0  , 10.3  , 18.4  , 64.4 );
    kernels[140] = compute_kernel(5, x,   5.0  , 10.7  , 17.2  , 60.3 );
    kernels[141] = compute_kernel(5, x,   5.0  , 20.8  , 26.2  , 79.5 );
    kernels[142] = compute_kernel(5, x,   5.0  , 14.2  , 16.7  , 93.0 );
    kernels[143] = compute_kernel(5, x,   5.0  , 15.6  , 22.9  , 59.8 );
    kernels[144] = compute_kernel(5, x,   5.0  , 13.8  , 23.8  , 69.3 );
    kernels[145] = compute_kernel(5, x,   5.0  , 11.9  , 19.3  , 72.3 );
    kernels[146] = compute_kernel(5, x,   5.0  , 12.4  , 19.7  , 86.0 );
    kernels[147] = compute_kernel(5, x,   5.0  , 12.8  , 19.1  , 79.8 );
    kernels[148] = compute_kernel(5, x,   5.0  , 10.8  , 26.3  , 73.0 );
    kernels[149] = compute_kernel(5, x,   5.0  , 16.5  , 23.8  , 78.9 );
    kernels[150] = compute_kernel(5, x,   6.0  , 15.9  , 23.9  , 74.6 );
    kernels[151] = compute_kernel(5, x,   6.0  , 14.6  , 22.7  , 80.8 );
    kernels[152] = compute_kernel(5, x,   6.0  , 18.8  , 29.5  , 69.6 );
    kernels[153] = compute_kernel(5, x,   6.0  , 20.5  , 29.8  , 68.5 );
    kernels[154] = compute_kernel(5, x,   6.0  , 19.9  , 24.4  , 84.5 );
    kernels[155] = compute_kernel(5, x,   6.0  , 21.5  , 25.3  , 79.1 );
    kernels[156] = compute_kernel(5, x,   6.0  , 18.8  , 27.9  , 70.5 );
    kernels[157] = compute_kernel(5, x,   6.0  , 19.9  , 26.3  , 83.5 );
    kernels[158] = compute_kernel(5, x,   7.0  , 20.6  , 26.6  , 81.6 );
    kernels[159] = compute_kernel(5, x,   8.0  , 24.3  , 26.9  , 90.9 );
    kernels[160] = compute_kernel(5, x,   8.0  , 23.5  , 26.4  , 96.3 );
    kernels[161] = compute_kernel(5, x,   9.0  , 22.6  , 26.4  , 91.3 );
    kernels[162] = compute_kernel(5, x,   9.0  , 20.0  , 25.6  , 82.0 );
    kernels[163] = compute_kernel(5, x,   9.0  , 16.2  , 25.7  , 78.0 );
    kernels[164] = compute_kernel(5, x,   3.0  , 10.2  , 15.0  , 70.9 );
    kernels[165] = compute_kernel(5, x,   4.0  , 11.9  , 17.8  , 70.4 );
    kernels[166] = compute_kernel(5, x,   4.0  , 8.6  , 14.6  , 81.8 );
    kernels[167] = compute_kernel(5, x,   6.0  , 18.4  , 23.1  , 83.0 );
    kernels[168] = compute_kernel(5, x,   7.0  , 22.6  , 24.9  , 95.8 );
    kernels[169] = compute_kernel(5, x,   8.0  , 22.9  , 26.0  , 92.8 );
    kernels[170] = compute_kernel(5, x,   8.0  , 23.7  , 25.8  , 95.4 );
    kernels[171] = compute_kernel(5, x,   8.0  , 24.7  , 29.5  , 91.6 );
    kernels[172] = compute_kernel(5, x,   8.0  , 25.0  , 27.0  , 96.0 );
    kernels[173] = compute_kernel(5, x,   8.0  , 21.7  , 27.6  , 87.4 );
    kernels[174] = compute_kernel(5, x,   8.0  , 24.4  , 30.6  , 88.8 );
    kernels[175] = compute_kernel(5, x,   9.0  , 17.5  , 20.6  , 93.5 );
    kernels[176] = compute_kernel(5, x,   11.0  , 6.9  , 19.1  , 90.5 );
    kernels[177] = compute_kernel(5, x,   3.0  , 0.5  , 9.0  , 91.0 );
    kernels[178] = compute_kernel(5, x,   4.0  , 10.9  , 18.0  , 80.8 );
    decisions[0] = 3.157951645122
    + kernels[0]
    + kernels[1]
    + kernels[2]
    + kernels[3]
    + kernels[4]
    + kernels[5]
    + kernels[6]
    + kernels[7]
    + kernels[8]
    + kernels[9]
    + kernels[10]
    + kernels[11]
    + kernels[12]
    + kernels[13]
    + kernels[14]
    + kernels[15]
    + kernels[16]
    + kernels[17]
    + kernels[18]
    + kernels[19]
    + kernels[20] * 0.8135783862
    + kernels[21]
    + kernels[22]
    + kernels[23]
    + kernels[24]
    + kernels[25]
    + kernels[26]
    + kernels[27]
    + kernels[28]
    + kernels[29]
    + kernels[30]
    + kernels[31]
    + kernels[32]
    + kernels[33] * 0.409694981732
    + kernels[34]
    + kernels[35]
    + kernels[36]
    + kernels[37]
    + kernels[38]
    + kernels[39]
    + kernels[40]
    + kernels[41]
    + kernels[42]
    + kernels[43]
    + kernels[44]
    + kernels[45]
    + kernels[46]
    + kernels[47]
    + kernels[48]
    + kernels[49]
    + kernels[50]
    + kernels[51]
    + kernels[52]
    + kernels[53]
    + kernels[54]
    + kernels[55]
    + kernels[56]
    + kernels[57]
    + kernels[58]
    + kernels[59]
    + kernels[60]
    + kernels[61] * 0.061659460608
    + kernels[62]
    + kernels[63]
    + kernels[64] * 0.799152066857
    + kernels[65]
    + kernels[66]
    + kernels[67]
    + kernels[68]
    + kernels[69]
    + kernels[70]
    + kernels[71]
    + kernels[72]
    + kernels[73]
    - kernels[75]
    - kernels[76]
    - kernels[77]
    - kernels[78]
    - kernels[79]
    - kernels[80]
    - kernels[81]
    - kernels[82]
    - kernels[83]
    - kernels[84]
    - kernels[88]
    - kernels[89]
    - kernels[90]
    - kernels[93]
    - kernels[94]
    - kernels[95]
    - kernels[96]
    - kernels[97]
    - kernels[98]
    - kernels[99]
    - kernels[100]
    - kernels[101]
    - kernels[102]
    - kernels[103]
    - kernels[104]
    - kernels[106]
    - kernels[107]
    - kernels[108]
    - kernels[109]
    - kernels[110]
    - kernels[111]
    - kernels[112]
    - kernels[114]
    - kernels[115]
    - kernels[116]
    - kernels[117]
    - kernels[118]
    - kernels[119]
    - kernels[120]
    - kernels[121]
    - kernels[122]
    - kernels[123]
    - kernels[124]
    - kernels[125]
    - kernels[126]
    - kernels[127]
    - kernels[128]
    - kernels[130]
    - kernels[131]
    - kernels[132]
    - kernels[133]
    - kernels[134]
    - kernels[135]
    - kernels[136]
    - kernels[137]
    - kernels[138]
    - kernels[139]
    - kernels[140]
    - kernels[141]
    - kernels[143]
    - kernels[144]
    - kernels[145]
    - kernels[146]
    - kernels[147]
    - kernels[148]
    - kernels[149]
    - kernels[150]
    - kernels[151]
    - kernels[152]
    - kernels[153]
    + kernels[154] * -0.084084895397
    - kernels[155]
    - kernels[156]
    ;
    decisions[1] = 6.569181730597
    + kernels[2]
    + kernels[11]
    + kernels[12]
    + kernels[13]
    + kernels[14]
    + kernels[15] * 0.123033845248
    + kernels[28] * 0.99032335488
    + kernels[47] * 0.852587153909
    + kernels[54]
    + kernels[68] * 0.629218568622
    - kernels[157]
    - kernels[158]
    + kernels[159] * -0.595162922659
    - kernels[161]
    - kernels[162]
    - kernels[163]
    - kernels[164]
    - kernels[165]
    - kernels[166]
    ;
    decisions[2] = 46.068773018707
    + kernels[11]
    + kernels[12] * 0.919541994091
    + kernels[47] * 0.247069123968
    + kernels[54] * 0.631267115691
    - kernels[167]
    + kernels[173] * -0.772247620478
    + kernels[176] * -0.025630613271
    - kernels[178]
    ;
    decisions[3] = 1.002372742013
    + kernels[74] * 0.922247589871
    + kernels[77] * 0.416353029398
    + kernels[84] * 0.026553867937
    + kernels[85]
    + kernels[89]
    + kernels[91]
    + kernels[92]
    + kernels[93]
    + kernels[102] * 0.170955746172
    + kernels[105] * 0.441345913334
    + kernels[111] * 0.002978858828
    + kernels[113] * 0.117431331039
    + kernels[119] * 0.010129997627
    + kernels[121] * 0.243488001877
    + kernels[123] * 0.247977512788
    + kernels[127] * 0.309573884754
    + kernels[129]
    + kernels[144] * 2.490579e-06
    + kernels[146]
    + kernels[153] * 0.004209535811
    + kernels[155] * 0.086752239985
    - kernels[157]
    - kernels[158]
    - kernels[159]
    - kernels[160]
    - kernels[161]
    - kernels[162]
    - kernels[163]
    - kernels[164]
    - kernels[165]
    - kernels[166]
    ;
    decisions[4] = 43.779601451256
    + kernels[74]
    + kernels[85]
    + kernels[86]
    + kernels[87] * 0.919116435558
    + kernels[91]
    + kernels[92]
    + kernels[105]
    + kernels[129]
    + kernels[142] * 0.722873253223
    + kernels[146]
    + kernels[148] * 0.353677394059
    - kernels[167]
    - kernels[168]
    - kernels[169]
    - kernels[170]
    - kernels[171]
    + kernels[172] * -0.165377601557
    - kernels[173]
    - kernels[174]
    - kernels[175]
    + kernels[177] * -0.830289481283
    - kernels[178]
    ;
    decisions[5] = 28.516598956262
    + kernels[157]
    + kernels[158] * 0.228596918608
    + kernels[159]
    + kernels[160]
    + kernels[161]
    + kernels[163] * 0.530287404309
    + kernels[166]
    - kernels[167]
    - kernels[169]
    - kernels[173]
    + kernels[174] * -0.982623833887
    + kernels[175] * -0.80187710854
    + kernels[178] * -0.97438338049
    ;
    votes[decisions[0] > 0 ? 0 : 1] += 1;
    votes[decisions[1] > 0 ? 0 : 2] += 1;
    votes[decisions[2] > 0 ? 0 : 3] += 1;
    votes[decisions[3] > 0 ? 1 : 2] += 1;
    votes[decisions[4] > 0 ? 1 : 3] += 1;
    votes[decisions[5] > 0 ? 2 : 3] += 1;
    int val = votes[0];
    int idx = 0;

    for (int i = 1; i < 4; i++) {
        if (votes[i] > val) {
            val = votes[i];
            idx = i;
        }
    }

    return idx;
}


/**
* Convert class idx to readable name
*/
const char idxToLabel(int classIdx) {
    switch (classIdx) {
        case 0:
        return 0;
        case 1:
        return 1;
        case 2:
        return 2;
        case 3:
        return 3;
        default:
        return -1;
    }
}

/**
* Predict readable class name
*/
const char predictLabel(float *x) {
    return idxToLabel(predict(x));
}


void init_uart(){
   DDRE = 0x0E; //PORT E에서 1번 bit를 output으로 설정
   UBRR0H = 0; //상위 비트는 0으로 설정
   UBRR0L = 103; //하위 비트는 103으로 설정, baud rate 9600bps로 설정
   UCSR0B = 0x18; //3번과 4번 비트인 수신부와 송신부 enable
}

void init_port()
{
   DDRA = 0xFF; //모두 output으로 설정
   DDRE = 0x0E; //PE2와 PE3를 output으로 설정
   PORTE = 0x04; //FND 위치 설정으로 변경
   PORTA = 0x0F; //모든 FND를 다 출력
   DDRF = 0xF0;
}

unsigned char FND_SEGNP[10] = {0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x27, 0x7F, 0x6F};
//7 Segment에 숫자를 간편하게 입력하기 위해서 선언한 것이다.

unsigned char FND_SEGPOS[4] = {0x01, 0x02, 0x04, 0x08};
//7 Segment 중에서 출력할 위치를 선정하는 배열이다.





void print() {

    PORTA = FND_SEGPOS[0];
    PORTE = 0X04;   // LED Ctl
    PORTE = 0X00;

    PORTA = FND_SEGNP[seg_t[3]];
    PORTE = 0X08;	// LED Data를 건듦으로서 Port A의 신호를 받아 FND 출력에 이용
    PORTE = 0X00;

    _delay_ms(0.05);

    PORTA = FND_SEGPOS[1];
    PORTE = 0X04;   // LED Ctl
    PORTE = 0X00;

    PORTA = 0x80;
    PORTE = 0X08;	// LED Data를 건듦으로서 Port A의 신호를 받아 FND 출력에 이용
    PORTE = 0X00;

    _delay_ms(0.05);

    PORTA = FND_SEGPOS[2];
    PORTE = 0X04;   // LED Ctl
    PORTE = 0X00;

    PORTA = FND_SEGNP[seg_t[0]];
    PORTE = 0X08;	// LED Data를 건듦으로서 Port A의 신호를 받아 FND 출력에 이용
    PORTE = 0X00;

    _delay_ms(0.05);

    PORTA = FND_SEGPOS[3];
    PORTE = 0X04;   // LED Ctl
    PORTE = 0X00;

    PORTA = FND_SEGNP[seg_t[1]];
    PORTE = 0X08;	// LED Data를 건듦으로서 Port A의 신호를 받아 FND 출력에 이용
    PORTE = 0X00;

    _delay_ms(0.05);


}





int main(){
   init_port(); //uart 초기설정
   init_uart();
   char rx, tx; //송신 수신 문자 변수 선언
   int num[4] = {8,22,27,90};
   int car = 0;
   int count = 0;
   
   int temp[3] = {0, };
   int flag = 1;   


   while(1){


      for(int i = 0; i < 3; i++) {
         while((UCSR0A & 0x80)==0x00); //수신을 받으면 while문 벗어남
         rx = UDR0; //rx에 수신받은 값 저장
         temp[i] = atoi(&rx);
      }

	  
	  if(flag == 1) {
	    seg_t[0] = temp[1];
		seg_t[1] = temp[2];
	    car = temp[1] * 10 + temp[2];
		flag = 0;
	  }
	  else {
		  num[count++] = temp[0] * 100 + temp[1] * 10 + temp[2];  
	  
	  }

      	   
       for(int i = 0; i < 3; i++) {
		  while((UCSR0A & 0x20)==0x00); //송신 가능할 때까지
		  char a = temp[i] + 48;
		  UDR0 = a;
	   }
		while((UCSR0A & 0x20)==0x00); //송신 가능할 때까지
		UDR0 = '\n';

		while((UCSR0A & 0x20)==0x00); //송신 가능할 때까지
		UDR0 = '\r';

      if (count == 4)
          break;

   }
   if(car == 77)
      PORTF = 0xF0;
   else if(car == 47)
      PORTF = 0x70;
   else if(car == 25)
      PORTF = 0xC0;
   else if(car == 75)
      PORTF = 0xD0;
   else if(car == 68)
      PORTF = 0x30;
   else if(car == 57)
      PORTF = 0x90;
   else if(car == 43)
      PORTF = 0x50;
   else if(car == 92)
      PORTF = 0x10;


   /*--입력 완료 점등--*/
   /*PORTF = 0xFF;
   _delay_ms(1000);
   PORTF = 0x00;*/


   char a = predictLabel(num);
   
   if (a == 0)
      seg_t[3] = 0;
   else if(a == 1)
      seg_t[3] = 1;
   else if(a == 2)
      seg_t[3] = 2;
   else if(a == 3)
      seg_t[3] = 3;
   else if(a == 4)
      seg_t[3] = 4;
   /*else if(a == 5)
      PORTF = 0x50;
   else if(a == 6)
      PORTF = 0x60;
   else if(a == 7)
      PORTF = 0x70;
   else if(a == 8)
      PORTF = 0x80;
   else if(a == 9)
      PORTF = 0x90;
   else
      PORTF = 0xF0;
*/

	while(1) {
	  print();
	}
   return 0;
}
